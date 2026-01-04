import ee
import pandas as pd
import numpy as np
import time
import concurrent.futures
import logging
from scipy import stats
from pandas.tseries.offsets import DateOffset
import os
import json
import tempfile

# --- 0. CONSTANTS ---
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

# EE Objects (hoisted)
DEM_IMG = ee.Image("NASA/NASADEM_HGT/001")
WRI_BASELINE_FC = ee.FeatureCollection("WRI/Aqueduct_Water_Risk/V4/baseline_annual")
WRI_FUTURE_FC = ee.FeatureCollection("WRI/Aqueduct_Water_Risk/V4/future_annual")

# Hazard sensitivity constants
WILDFIRE_TEMP_SENS = 0.16
CYCLONE_TEMP_SENS = 0.08

EPOCHS = {'2030s': (2025, 2034), '2050s': (2045, 2054), '2080s': (2075, 2084)}
EPOCH_MIDPOINTS = {'2030s': 2030, '2050s': 2050, '2080s': 2080}
ERA5_RANGE = ('1980-01-01', '2024-12-31')
CMIP6_HIST_RANGE = ('1980-01-01', '2014-12-31')
MODEL = 'MPI-ESM1-2-HR'

# ================= ERROR TYPES =================
class HazardError(Exception):
    def __init__(self, stage: str, message: str):
        self.stage = stage
        self.message = message
        super().__init__(f"[{stage}] {message}")

EE_INITIALIZED = False

def ensure_ee_initialized():
    global EE_INITIALIZED
    if EE_INITIALIZED:
        return

    try:
        if "GCP_SERVICE_ACCOUNT" not in os.environ:
            raise HazardError(
                stage="EE_INIT",
                message="GCP_SERVICE_ACCOUNT environment variable not set"
            )

        service_account_info = json.loads(os.environ["GCP_SERVICE_ACCOUNT"])

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            json.dump(service_account_info, f)
            key_path = f.name

        try:
            credentials = ee.ServiceAccountCredentials(
                service_account_info["client_email"],
                key_path
            )

            ee.Initialize(
                credentials,
                project=service_account_info["project_id"],
                url="https://earthengine-highvolume.googleapis.com"
            )

            EE_INITIALIZED = True

        finally:
            os.remove(key_path)

    except HazardError:
        raise
    except Exception as e:
        raise HazardError(
            stage="EE_INIT",
            message=str(e)
        )

def safe_to_float(val, default=0.3):
    """Safely convert values to float with fallback"""
    if val is None:
        return default
    try:
        v = float(val)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default

def is_coastal(geom, max_dist_km=50):
    """
    Coastal if:
    - elevation ≤ 30 m
    - within max_dist_km of ocean
    """
    try:
        elev = DEM_IMG.select("elevation") \
            .reduceRegion(
                ee.Reducer.mean(),
                geom,
                scale=90,
                bestEffort=True
            ).get("elevation").getInfo()
    except Exception as e:
        logging.exception("is_coastal elevation check failed")
        return False

    if elev is None or elev > 30:
        return False

    # Distance to ocean (landmask inversion)
    ocean = ee.Image("JRC/GSW1_4/GlobalSurfaceWater") \
        .select("occurrence") \
        .lt(1)  # land = 1, ocean = 0

    dist = ocean.fastDistanceTransform(1000).sqrt() \
        .multiply(30) \
        .reduceRegion(
            ee.Reducer.min(),
            geom,
            scale=1000,
            bestEffort=True
        ).values().get(0).getInfo()

    return dist is not None and dist <= max_dist_km * 1000

# -------------------------------------------------------------------------
# 1. ULTRA-FAST ANNUAL FEATURE EXTRACTOR
# -------------------------------------------------------------------------
def get_annual_features(
    collection_id,
    geom,
    start_year,
    end_year,
    bands,
    model=None,
    scenario=None,
    reducer='mean'
):
    """Extract annual aggregates directly from EE - 50x faster than daily processing"""
    try:
        coll = ee.ImageCollection(collection_id)

        if model:
            coll = coll.filterMetadata('model', 'equals', model)
        if scenario:
            coll = coll.filterMetadata('scenario', 'equals', scenario)

        coll = coll.select(bands)

        def year_feature(y):
            y = ee.Number(y)
            start = ee.Date.fromYMD(y, 1, 1)
            end = start.advance(1, 'year')

            img = coll.filterDate(start, end)

            img = ee.Image(
                ee.Algorithms.If(
                    reducer == 'total_precipitation_sum',
                    img.sum(),
                    img.mean()
                )
            )

            stats = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=25000,
                bestEffort=True
            )

            return ee.Feature(None, stats).set('year', y)

        years = ee.List.sequence(start_year, end_year)
        fc = ee.FeatureCollection(years.map(year_feature))
        
        features = fc.getInfo()['features']
        if not features:
            return pd.DataFrame()
            
        df = pd.DataFrame([f['properties'] for f in features])
        df['year'] = df['year'].astype(int)
        return df
    except Exception as e:
        logging.exception(
            f"get_annual_features failed | collection={collection_id} "
            f"model={model} scenario={scenario}"
        )
        return pd.DataFrame()

def score(val, min_v, max_v, inverted=False):
    if pd.isna(val): return np.nan
    n = 1 - (val - min_v) / (max_v - min_v) if inverted else (val - min_v) / (max_v - min_v)
    return round(np.clip(n, 0, 1) * 5, 3)

def perform_qm(train_df, hist_df, fut_df, t_col, h_col, f_col):
    """
    Annual quantile mapping (NO monthly logic).
    Correct for annual-mean / annual-extreme climate data.
    """
    if train_df.empty or hist_df.empty or fut_df.empty:
        return fut_df[f_col].values

    # Observed (ERA5) reference distribution
    obs = np.sort(train_df[t_col].values)

    # Historical model distribution
    hist = np.sort(hist_df[h_col].values)
    
    # Guard against zero variance
    if np.std(hist) < 1e-6:
        return fut_df[f_col].values

    # Future model values
    fut = fut_df[f_col].values

    # Percentile rank of future values in historical model space
    ranks = np.searchsorted(hist, fut) / len(hist)

    # Map those ranks onto observed distribution
    corrected = np.quantile(obs, np.clip(ranks, 0, 0.9999))

    return corrected

# -------------------------------------------------------------------------
# 2. IPCC SEA LEVEL RISE MULTIPLIER
# -------------------------------------------------------------------------
def get_ipcc_slr_multiplier(year, scenario, geom):
    try:
        local_elev = DEM_IMG.select('elevation').reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=30,
            bestEffort=True
        ).get('elevation').getInfo()
    except Exception:
        local_elev = None

    if local_elev is not None and local_elev > 30:
        return 0.0

    slr_factors = {
        'ssp245': {2030: 1.05, 2050: 1.15, 2080: 1.35},
        'ssp585': {2030: 1.08, 2050: 1.25, 2080: 1.65}
    }

    epoch = min(slr_factors[scenario].keys(), key=lambda x: abs(x - year))
    return slr_factors[scenario][epoch]

# -------------------------------------------------------------------------
# 3. STRICT FUTURE WRI EXTRACTOR (NO FALLBACK)
# -------------------------------------------------------------------------
def safe_float(props, key):
    v = props.get(key)
    if v is None or str(v).lower() in ['none', 'null', 'nan']:
        return np.nan
    try:
        v = float(v)
    except:
        return np.nan
    return float(np.clip(v, 0.0, 5.0))

def extract_wri_future_strict(geom, geom_coords):
    """
    Strict extraction of Aqueduct V4 future_annual values.
    No fallback. No baseline overwrite. Dataset truth only.
    """
    try:
        lon, lat = geom_coords
        pt = ee.Geometry.Point([lon, lat])

        # Find nearest polygon ONCE
        nearest = (
            WRI_FUTURE_FC
            .map(lambda f: f.set('distance', ee.Feature(pt).distance(f.geometry())))
            .sort('distance')
            .first()
            .getInfo()
        )

        if not nearest:
            return {}

        props = nearest['properties']
        future = {}

        for scenario in ['Trend', 'ssp245', 'ssp585']:
            pref = (
                'pes' if scenario == 'ssp585'
                else 'opt' if scenario == 'ssp245'
                else 'bau'
            )

            for ep, mid in EPOCH_MIDPOINTS.items():
                y_key = '30' if mid == 2030 else '50' if mid == 2050 else '80'
                col = f'{scenario}_{ep}'

                future[col] = {
                    'Water Stress': safe_float(props, f'{pref}{y_key}_ws_x_s'),
                    'Seasonal Variability': safe_float(props, f'{pref}{y_key}_sv_x_s'),
                    'Interannual Variability': safe_float(props, f'{pref}{y_key}_iv_x_s'),
                    'Drought Risk': None  # Future dataset doesn't provide this
                }

        return future
    except Exception as e:
        logging.exception("extract_wri_future_strict failed")
        return {}

# -------------------------------------------------------------------------
# 4. OBSERVED WRI BASELINE
# -------------------------------------------------------------------------
WRI_SYSTEM_KEYS = ['Water Stress', 'Seasonal Variability', 'Interannual Variability']
WRI_EVENT_KEYS  = ['Drought Risk', 'Riverine Flood Risk', 'Coastal Flood Risk']

def system_metrics_invalid(result):
    return any(pd.isna(result.get(k)) for k in WRI_SYSTEM_KEYS)

def sample_nearest_wri(coll, point, use_baseline_bau30, year=None, scenario_name=None):
    try:
        knn = coll.map(lambda f: f.set('distance', ee.Feature(point).distance(f.geometry())))
        nearest = knn.sort('distance').limit(1).getInfo()['features']

        if not nearest:
            return None

        props = nearest[0]['properties']

        if use_baseline_bau30:
            return {
                'Water Stress': safe_float(props, 'bws_score'),
                'Drought Risk': safe_float(props, 'drr_score'),
                'Seasonal Variability': safe_float(props, 'sev_score'),
                'Interannual Variability': safe_float(props, 'iav_score'),
                'Riverine Flood Risk': safe_float(props, 'rfr_score'),
                'Coastal Flood Risk': safe_float(props, 'cfr_score')
            }
        else:
            pref = 'pes' if '585' in (scenario_name or '') else 'opt' if '245' in (scenario_name or '') else 'bau'
            y_key = '30' if year == 2030 else '50' if year == 2050 else '80'
            return {
                'Water Stress': safe_float(props, f'{pref}{y_key}_ws_x_s'),
                'Seasonal Variability': safe_float(props, f'{pref}{y_key}_sv_x_s'),
                'Interannual Variability': safe_float(props, f'{pref}{y_key}_iv_x_s'),
                'Drought Risk': None
            }
    except Exception as e:
        logging.exception("sample_nearest_wri failed")
        return None

def get_wri_4_directions_parallel(geom, geom_coords, year=None, scenario_name=None, use_baseline_bau30=False):
    start_time = time.time()
    lon, lat = geom_coords

    if use_baseline_bau30:
        coll = WRI_BASELINE_FC
    else:
        coll = WRI_FUTURE_FC

    center_start = time.time()
    center_pt = ee.Geometry.Point([lon, lat])
    center_result = sample_nearest_wri(coll, center_pt, use_baseline_bau30, year, scenario_name)
    center_time = time.time() - center_start

    system_locked = False
    if use_baseline_bau30 and center_result and system_metrics_invalid(center_result):
        try:
            bau_2030 = sample_nearest_wri(
                WRI_FUTURE_FC,
                center_pt,
                use_baseline_bau30=False,
                year=2030,
                scenario_name='Trend_2030s'
            )

            if bau_2030:
                for k in WRI_SYSTEM_KEYS:
                    if pd.isna(center_result.get(k)):
                        center_result[k] = bau_2030.get(k)
                center_result['_system_fallback'] = 'BAU_2030_CENTER'
        except Exception:
            pass
        if center_result and '_system_fallback' in center_result:
            system_locked = True

    # Safer WRI validation (require 2+ valid events)
    valid_events = [
        k for k in WRI_EVENT_KEYS
        if not pd.isna(center_result.get(k))
    ]

    if center_result and len(valid_events) >= 2:
        center_result['_source'] = 'CENTER'
        total_time = time.time() - start_time
        center_result['_method'] = 'CENTER'
        return center_result, {
            'method': 'CENTER',
            'total_time': total_time
        }

    parallel_start = time.time()
    earth_radius = 6371000
    lat_rad = np.radians(lat)
    all_points = {}
    radii_m = [10000, 50000, 100000, 200000]

    for radius in radii_m:
        delta_lat = (radius / earth_radius) * (180 / np.pi)
        delta_lon = (radius / earth_radius) * (180 / np.pi) / np.cos(lat_rad)

        all_points[f'{radius//1000}km_N'] = ee.Geometry.Point([lon, lat + delta_lat])
        all_points[f'{radius//1000}km_E'] = ee.Geometry.Point([lon + delta_lon, lat])
        all_points[f'{radius//1000}km_S'] = ee.Geometry.Point([lon, lat - delta_lat])
        all_points[f'{radius//1000}km_W'] = ee.Geometry.Point([lon - delta_lon, lat])

    # SERIAL: Process all directions
    all_results = {}
    for name, pt in all_points.items():
        result = sample_nearest_wri(
            coll, pt, use_baseline_bau30, year, scenario_name
        )
        if result:
            all_results[name] = result

    parallel_time = time.time() - parallel_start

    if not all_results:
        total_time = time.time() - start_time
        return {k: 1.0 for k in ['Water Stress', 'Drought Risk', 'Seasonal Variability',
                                 'Interannual Variability', 'Riverine Flood Risk', 'Coastal Flood Risk']}, {'method': 'FAILED', 'total_time': total_time}

    valid_candidates = []
    for name, result in all_results.items():
        valid_count = sum(not pd.isna(result.get(k)) for k in WRI_EVENT_KEYS)
        if valid_count >= 1:
            radius_km = int(name.split('km')[0])
            valid_candidates.append((radius_km, result, name))

    if valid_candidates:
        valid_candidates.sort(key=lambda x: x[0])
        nearest = valid_candidates[0][1]

        if system_locked:
            for k in WRI_SYSTEM_KEYS:
                nearest[k] = center_result[k]

        radius_used, name_used = valid_candidates[0][0], valid_candidates[0][2]
        total_time = time.time() - start_time
        nearest['_method'] = f'PARALLEL_{radius_used}km'
        nearest['_source'] = f'PARALLEL_{radius_used}km'
        return nearest, {
            'method': f'PARALLEL_{radius_used}km',
            'center_time': center_time,
            'parallel_time': parallel_time,
            'total_time': total_time
        }

    total_time = time.time() - start_time
    return {k: 1.0 for k in ['Water Stress', 'Drought Risk', 'Seasonal Variability',
                             'Interannual Variability', 'Riverine Flood Risk', 'Coastal Flood Risk']}, {'method': 'NO_VALID', 'total_time': total_time}

def get_fire_cyclone_baselines(geom):
    landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
        .filterBounds(geom) \
        .filterDate('2023-01-01', '2023-12-31') \
        .median()

    ndvi = landsat.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    ndvi_val = ndvi.reduceRegion(
        ee.Reducer.mean(), geom, 1000, bestEffort=True
    ).get('NDVI')

    firms = ee.ImageCollection("FIRMS") \
        .filterBounds(geom) \
        .filterDate('2023-01-01', '2023-12-31') \
        .filter(ee.Filter.gt('confidence', 75)) \
        .filter(ee.Filter.eq('type', 0)) \
        .filter(ee.Filter.gt('frp', 15))

    wf_count = firms.size()

    tracks = ee.FeatureCollection("NOAA/IBTrACS/v4") \
        .filterBounds(geom) \
        .filter(ee.Filter.gte('SEASON', 1990))

    storm_count = tracks.distinct('SID').size()

    def clean_wind(f):
        wind_cols = [
            'WMO_WIND', 'USA_WIND', 'TOKYO_WIND',
            'CMA_WIND', 'HKO_WIND', 'NEWDELHI_WIND',
            'REUNION_WIND', 'BOM_WIND', 'NADI_WIND', 'WELLINGTON_WIND'
        ]
        max_v = ee.Number(0)
        for col in wind_cols:
            val = ee.Number(ee.Algorithms.If(f.get(col), f.get(col), 0))
            max_v = max_v.max(val)
        return f.set('peak_wind', max_v)
    
    max_wind = tracks.map(clean_wind).aggregate_max('peak_wind')

    return ee.Dictionary({
        'NDVI': ndvi_val,
        'wf_count': wf_count,
        'storm_count': storm_count,
        'max_wind_knots': max_wind
    }).getInfo()

# -------------------------------------------------------------------------
# 5. MAIN API FUNCTION (REFACTORED)
# -------------------------------------------------------------------------
def run_for_point(lat: float, lon: float):
    try:
        ensure_ee_initialized()
    except HazardError:
        raise
    
    total_start = time.time()
    final_rows = []
    WRI_WATER_HAZARDS = ['Water Stress', 'Drought Risk', 'Seasonal Variability', 'Interannual Variability']

    logging.info(
        "Processing point",
        extra={"lat": round(lat, 4), "lon": round(lon, 4)}
    )
    
    geom = ee.Geometry.Point([lon, lat])
    geom_coords = (lon, lat)  # Cache coordinates
    
    coastal_flag = is_coastal(geom)
    
    # ULTRA-FAST ERA5 + HISTORICAL (annual aggregation in EE)
    climate_start = time.time()
    era5 = get_annual_features(
        "ECMWF/ERA5_LAND/DAILY_AGGR",
        geom,
        1980, 2024,
        ['temperature_2m', 'temperature_2m_max', 'temperature_2m_min', 'total_precipitation_sum']
    )
    hist_cmip = get_annual_features(
        "NASA/GDDP-CMIP6",
        geom,
        1980, 2014,
        ['tas', 'tasmax', 'tasmin', 'pr'],
        MODEL, 'historical'
    )
    
    # Unit conversions
    if not era5.empty:
        era5['pr'] = era5['total_precipitation_sum'] * 1000
    if not hist_cmip.empty:
        hist_cmip['tas'] -= 273.15
        hist_cmip['tasmax'] -= 273.15
        hist_cmip['tasmin'] -= 273.15
        hist_cmip['pr'] *= 86400  # daily to mm
    
    if era5.empty:
        raise HazardError(
            stage="ERA5_FETCH",
            message="ERA5 returned empty dataframe"
        )
    climate_time = time.time() - climate_start

    # SEPARATED WRI FLOWS
    wri_start = time.time()
    
    # 1️⃣ OBSERVED BASELINE ONLY
    wri_base, _ = get_wri_4_directions_parallel(
        geom, geom_coords, use_baseline_bau30=True
    )
    
    if not wri_base:
        raise HazardError(
            stage="WRI_FETCH",
            message="WRI baseline returned no data"
        )
        
    # 2️⃣ STRICT FUTURE ONLY (no fallback)
    wri_future = extract_wri_future_strict(geom, geom_coords)
    
    drr_base_score = wri_base.get('Drought Risk', np.nan)
    baseline_method = wri_base.get('_method', 'UNKNOWN')
    wri_time = time.time() - wri_start

    rfr_base = wri_base.get('Riverine Flood Risk')
    if rfr_base is None or pd.isna(rfr_base):
        rfr_base = np.nan

    cfr_base = wri_base.get('Coastal Flood Risk')
    if pd.isna(cfr_base):
        cfr_base = np.nan
        
    buffer_m = 50_000 if coastal_flag else 30_000
    geom_fc = geom.buffer(buffer_m)
    
    fire_cyclone_base = get_fire_cyclone_baselines(geom_fc)
    
    if fire_cyclone_base is None:
        raise HazardError(
            stage="FIRE_CYCLONE",
            message="Fire/Cyclone baseline failed"
        )

    ndvi = safe_to_float(fire_cyclone_base.get('NDVI'), default=0.3)
    wf_count = safe_to_float(fire_cyclone_base.get('wf_count'), default=0.0)
    storm_count = fire_cyclone_base.get('storm_count')
    try:
        storm_count = float(storm_count)
    except Exception:
        storm_count = np.nan
    max_wind_knots = safe_to_float(fire_cyclone_base.get('max_wind_knots'), default=20.0)

    fuel_mult = np.clip(ndvi / 0.3, 0.5, 1.5)
    wf_base = np.clip((wf_count / 100) * 5, 0.5, 5.0) * fuel_mult
    
    # CYCLONE BASELINE (STRICTLY COASTAL)
    if not coastal_flag:
        cy_base = 0.0
    else:
        if storm_count is None or pd.isna(storm_count) or storm_count < 1:
            cy_base = 0.5
        else:
            cy_wind_kmh = max_wind_knots * 1.852
            rp_years = np.clip(34 / max(1, storm_count), 5, 100)

            damage_potential = (cy_wind_kmh / 180.0) ** 3

            cy_base = (
                np.clip(damage_potential * 5.0, 0.5, 4.5) * 0.8 +
                np.clip(15.0 / rp_years, 0.5, 4.5) * 0.2
            )
    
    # ERA5 processing (now annual data)
    era5_base = era5[(era5['year'] >= 1980) & (era5['year'] <= 2014)]
    era5_temp_base = era5_base['temperature_2m'].mean()
    
    era5_pr_base = era5_base['pr'].mean()
    if pd.isna(era5_pr_base) or era5_pr_base == 0:
        era5_pr_base = 1e-6

    hist_tmax = era5_base['temperature_2m_max']
    hist_tmin = era5_base['temperature_2m_min']

    heat_threshold = hist_tmax.quantile(0.95)
    cold_threshold = hist_tmin.quantile(0.05)

    trends = {}
    for m_name, col, func in [('max_t','temperature_2m_max','max'),('min_t','temperature_2m_min','min'),('mean_t','temperature_2m','mean'),
                              ('pr_sum','pr','sum'),('max_pr','pr','max')]:
        ann = era5.groupby('year')[col].agg(func).reset_index()
        slope, icept, _, _, _ = stats.linregress(ann['year'], ann[col])
        trends[m_name] = {'slope': slope, 'intercept': icept}

    def add_row(col_name, metrics, year_proj=None):
        is_obs = 'Observed' in col_name
        pr_pct, t_anom = metrics.get('pr_change_pct', 0.0), metrics.get('anom', 0.0)

        # Climate hazards (temperature/precipitation)
        for h, s in [('Extreme Heat', metrics['s_max_t']), ('Chronic Heat Stress', metrics['s_days_35']),
                     ('Extreme Cold', metrics['s_min_t']), ('Chronic Cold Stress', metrics['s_days_0']),
                     ('Temperature Anomaly', metrics['s_anom']), ('Precipitation Change', metrics['s_pr_change']),
                     ('Extreme Precipitation', metrics['s_max_pr'])]:
            final_rows.append({'Hazard': h, 'Column': col_name, 'Score': round(s, 3)})

        # CLEAN WRI SEPARATION
        if is_obs:
            wri_vals = wri_base
        else:
            wri_vals = wri_future.get(col_name, {})

        # WRI Water Hazards
        for h in WRI_WATER_HAZARDS:
            # Drought Risk (special handling)
            if h == 'Drought Risk':
                if pd.isna(drr_base_score):
                    s = None
                elif is_obs:
                    s = drr_base_score
                else:
                    drought_mult = max(
                        0.7,
                        (1 - pr_pct / 100) + (t_anom / 4.0)
                    )
                    s = min(5.0, drr_base_score * drought_mult)
            # Other WRI hazards (direct lookup)
            else:
                s = wri_vals.get(h)

            final_rows.append({
                'Hazard': h,
                'Column': col_name,
                'Score': s
            })

        # Riverine Flood Risk (always applicable)
        if pd.isna(rfr_base):
            riverine_s = np.nan
        elif is_obs:
            riverine_s = rfr_base
        else:
            riverine_s = min(
                5.0,
                rfr_base * max(0.8, 1 + (pr_pct / 100))
            )

        # Coastal Flood Risk (ONLY if coastal)
        if not coastal_flag:
            coastal_s = np.nan
        else:
            if is_obs:
                coastal_s = cfr_base
            else:
                scenario = 'ssp585' if '585' in col_name else 'ssp245'
                slr_mult = get_ipcc_slr_multiplier(year_proj, scenario, geom)

                if pd.isna(cfr_base) or slr_mult is None:
                    coastal_s = np.nan
                else:
                    coastal_s = min(5.0, cfr_base * slr_mult)

        final_rows.append({'Hazard': 'Riverine Flood Risk', 'Column': col_name, 'Score': round(riverine_s, 3)})
        final_rows.append({'Hazard': 'Coastal Flood Risk', 'Column': col_name, 'Score': round(coastal_s, 3)})

        # Fire & Cyclone
        if is_obs:
            s_wf = wf_base
            s_cy = cy_base
        else:
            s_wf = wf_base * (1 + t_anom * WILDFIRE_TEMP_SENS)
            if cy_base == 0.0:
                s_cy = 0.0
            else:
                s_cy = cy_base * (1 + t_anom * CYCLONE_TEMP_SENS)

        s_wf = round(min(5.0, s_wf), 3)
        s_cy = round(min(5.0, s_cy), 3)

        final_rows.append({'Hazard': 'Wildfire', 'Column': col_name, 'Score': s_wf})
        final_rows.append({'Hazard': 'Cyclone', 'Column': col_name, 'Score': s_cy})

    # Execute Observed 2024
    obs_start = time.time()
    obs_dec = era5[(era5['year'] >= 2015) & (era5['year'] <= 2024)]
    
    if not obs_dec.empty:
        # Annual observed precipitation (mm/year)
        obs_ann_pr = obs_dec['pr'].mean()
        
        # Heat / cold day counts (approximated from annual max/min)
        dec_heat = (obs_dec['temperature_2m_max'] > heat_threshold).sum() * 365
        dec_cold = (obs_dec['temperature_2m_min'] < cold_threshold).sum() * 365
        
        # Temperature anomaly (raw, °C)
        t_anom_obs = obs_dec['temperature_2m'].mean() - era5_temp_base
        
        m = {
            # Temperature extremes
            's_max_t': score(obs_dec['temperature_2m_max'].max(), 30, 50),
            
            # Approx annualized intensity from decade counts
            's_days_35': score(dec_heat / 10, 5, 150),
            's_min_t': score(obs_dec['temperature_2m_min'].min(), -30, 5, inverted=True),
            's_days_0': score(dec_cold / 10, 1, 90),
            
            # Temperature anomaly
            'anom': t_anom_obs,
            's_anom': score(t_anom_obs, 0, 4),
            
            # Precipitation change (annual mean vs baseline)
            'pr_change_pct': ((obs_ann_pr - era5_pr_base) / era5_pr_base) * 100,
            's_pr_change': score(
                abs((obs_ann_pr - era5_pr_base) / era5_pr_base * 100),
                10, 50
            ),
            
            # Extreme precipitation (annual max)
            's_max_pr': score(obs_dec['pr'].max(), 30, 500)
        }
    else:
        m = {
            's_max_t': 1.0, 's_days_35': 1.0, 's_min_t': 1.0, 's_days_0': 1.0,
            's_anom': 0.0, 'pr_change_pct': 0.0, 's_pr_change': 0.0, 's_max_pr': 1.0
        }

    add_row('Observed_2024', m)
    obs_time = time.time() - obs_start

    # Execute scenarios - Annual CMIP6 epoch data only
    scenarios_start = time.time()
    cmip_cache = {}  # key = (scenario, epoch)
    
    for scenario in ['Trend', 'ssp245', 'ssp585']:
        for ep, (sy, ey) in EPOCHS.items():
            mid = EPOCH_MIDPOINTS[ep]
            if scenario == 'Trend':
                t_mean = trends['mean_t']['slope'] * mid + trends['mean_t']['intercept']
                heat_shift = (trends['max_t']['slope'] * mid + trends['max_t']['intercept']) - heat_threshold
                cold_shift = cold_threshold - (trends['min_t']['slope'] * mid + trends['min_t']['intercept'])

                dec_heat = max(0, 0.02 * 365 * (1 + heat_shift / 2.0))
                dec_cold = max(0, 0.02 * 365 * (1 + cold_shift / 2.0))

                min_t_proj = (trends['min_t']['slope'] * mid + trends['min_t']['intercept'])
                pr_sum_proj = trends['pr_sum']['slope'] * mid + trends['pr_sum']['intercept']
                max_pr_proj = trends['max_pr']['slope'] * mid + trends['max_pr']['intercept']
                pr_change_pct = ((pr_sum_proj - era5_pr_base) / era5_pr_base) * 100
                pr_change_pct = np.clip(pr_change_pct, -30, 30)

                dec_heat = max(dec_heat * 10, 0)
                dec_cold = max(dec_cold * 10, 0)
                m = {
                        's_max_t': score(trends['max_t']['slope'] * mid + trends['max_t']['intercept'], 30, 50),
                        's_min_t': score(min_t_proj, -30, 5, inverted=True),
                        's_days_35': score(dec_heat / 10, 5, 150),
                        's_days_0': score(dec_cold / 10, 1, 90),
                        'anom': t_mean - era5_temp_base,
                        's_anom': score(t_mean - era5_temp_base, 0, 5),
                        'pr_change_pct': pr_change_pct,
                        's_pr_change': score(abs(pr_change_pct), 10, 50),
                        's_max_pr': score(max_pr_proj, 30, 500)
                    }
            else:
                # Annual epoch data only (10 years)
                cache_key = (scenario, ep)
                if cache_key not in cmip_cache:
                    df_future = get_annual_features(
                        "NASA/GDDP-CMIP6",
                        geom,
                        sy, ey,
                        ['tas', 'tasmax', 'tasmin', 'pr'],
                        MODEL,
                        scenario
                    )
                    if not df_future.empty:
                        df_future['tas'] -= 273.15
                        df_future['tasmax'] -= 273.15
                        df_future['tasmin'] -= 273.15
                        df_future['pr'] *= 86400
                    cmip_cache[cache_key] = df_future

                sdf = cmip_cache[cache_key]
                if sdf.empty: continue
                
                mx = perform_qm(era5_base, hist_cmip, sdf, 'temperature_2m_max', 'tasmax', 'tasmax')
                mn = perform_qm(era5_base, hist_cmip, sdf, 'temperature_2m_min', 'tasmin', 'tasmin')
                av = perform_qm(era5_base, hist_cmip, sdf, 'temperature_2m', 'tas', 'tas')
                pr = perform_qm(era5_base, hist_cmip, sdf, 'pr', 'pr', 'pr')
                t_anom = np.mean(av) - era5_temp_base
                pr_change = ((np.mean(pr) - era5_pr_base) / era5_pr_base) * 100
                dec_heat = np.sum(mx > heat_threshold) * 36.5
                dec_cold = np.sum(mn < cold_threshold) * 36.5

                m = {'s_max_t': score(np.max(mx), 30, 50),
                     's_min_t': score(np.min(mn), -30, 5, True),
                     's_days_35': score(dec_heat / 10, 5, 150),
                     's_days_0': score(dec_cold / 10, 1, 90),
                     'anom': t_anom, 's_anom': score(t_anom, 0, 5),
                     'pr_change_pct': pr_change, 's_pr_change': score(abs(pr_change), 10, 50),
                     's_max_pr': score(np.max(pr), 30, 500)}
            add_row(f'{scenario}_{ep}', m, year_proj=mid)
    scenarios_time = time.time() - scenarios_start

    # Final matrix
    matrix_start = time.time()
    if not final_rows:
        raise HazardError(
            stage="FINAL_MATRIX",
            message="No hazard rows generated"
        )

    df_final = pd.DataFrame(final_rows).pivot(index='Hazard', columns='Column', values='Score')
    cols = ['Observed_2024'] + [f'{s}_{e}' for s in ['Trend', 'ssp245', 'ssp585'] for e in EPOCHS]
    df_final = df_final[[c for c in cols if c in df_final.columns]].reset_index()
    
    # NaN diagnostic
    nan_count = df_final.isna().sum().sum()
    if nan_count > 0:
        logging.warning(f"Final matrix contains {nan_count} NaN values")
    
    matrix_time = time.time() - matrix_start

    total_time = time.time() - total_start
    logging.info(f"✅ Hazards computed | Shape: {df_final.shape} | Time: {total_time:.1f}s")

    df_final = df_final.replace([np.inf, -np.inf, np.nan], None)
    return df_final
