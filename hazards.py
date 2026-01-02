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

# --- 0. SETUP ---
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

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
# --- CONFIGURATION ---
EPOCHS = {'2030s': (2025, 2034), '2050s': (2045, 2054), '2080s': (2075, 2084)}
EPOCH_MIDPOINTS = {'2030s': 2030, '2050s': 2050, '2080s': 2080}

ERA5_RANGE = ('1960-01-01', '2024-12-31')
CMIP6_HIST_RANGE = ('1960-01-01', '2014-12-31')
CMIP6_FUT_RANGE = ('2025-01-01', '2085-12-31')
MODEL = 'MPI-ESM1-2-HR'
CHUNK_SIZE_YEARS = 12
MAX_WORKERS = 4

def is_coastal(geom, max_dist_km=50):
    """
    Coastal if:
    - elevation ≤ 30 m
    - within max_dist_km of ocean
    """
    try:
        elev = ee.Image("NASA/NASADEM_HGT/001") \
            .select("elevation") \
            .reduceRegion(
                ee.Reducer.mean(),
                geom,
                scale=90,
                bestEffort=True
            ).get("elevation").getInfo()
    except:
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
# 1. SPATIAL CLIMATE ENGINE
# -------------------------------------------------------------------------
def fetch_chunk(collection_id, geom, start, end, bands, model=None, scenario=None):
    try:
        coll = ee.ImageCollection(collection_id).filterDate(start, end)
        if model: coll = coll.filterMetadata('model', 'equals', model)
        if scenario: coll = coll.filterMetadata('scenario', 'equals', scenario)
        coll = coll.select(bands)

        def reduce_to_point(img):
            stats = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom.buffer(5000),
                scale=25000,
                bestEffort=True
            )
            return ee.Feature(None, stats).set('system:time_start', img.get('system:time_start'))

        data_list = coll.map(reduce_to_point).getInfo()['features']
        if not data_list: return pd.DataFrame()

        rows = []
        for feat in data_list:
            row = feat['properties'].copy()
            row['time'] = feat['properties'].get('system:time_start')
            rows.append(row)

        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['time'], unit='ms')

        for b in ['temperature_2m', 'temperature_2m_max', 'temperature_2m_min', 'tas', 'tasmax', 'tasmin']:
            if b in df.columns:
                df[b] = pd.to_numeric(df[b], errors='coerce')
                if df[b].mean() > 200: df[b] = df[b] - 273.15
        if 'total_precipitation_sum' in df.columns:
            df['pr'] = pd.to_numeric(df['total_precipitation_sum'], errors='coerce') * 1000
        elif 'pr' in df.columns:
            df['pr'] = pd.to_numeric(df['pr'], errors='coerce') * 86400
        return df
    except: 
        return pd.DataFrame()

def get_full_series(collection_id, geom, start_date, end_date, bands, model=None, scenario=None):
    start = pd.to_datetime(start_date); end = pd.to_datetime(end_date)
    tasks = []; curr = start
    while curr < end:
        next_step = curr + DateOffset(years=CHUNK_SIZE_YEARS)
        chunk_end = min(next_step, end)
        tasks.append((curr.strftime('%Y-%m-%d'), chunk_end.strftime('%Y-%m-%d')))
        curr = next_step
    dfs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(fetch_chunk, collection_id, geom, s, e, bands, model, scenario): (s,e) for s,e in tasks}
        for f in concurrent.futures.as_completed(futures):
            df = f.result()
            if not df.empty: dfs.append(df)
    if not dfs: return pd.DataFrame()
    return pd.concat(dfs).sort_values('date').reset_index(drop=True)

def score(val, min_v, max_v, inverted=False):
    if pd.isna(val): return np.nan
    n = 1 - (val - min_v) / (max_v - min_v) if inverted else (val - min_v) / (max_v - min_v)
    return round(np.clip(n, 0, 1) * 5, 3)

def perform_qm(train_df, hist_df, fut_df, t_col, h_col, f_col):
    if train_df.empty or hist_df.empty or fut_df.empty: return fut_df[f_col].values
    obs_vals = train_df[t_col].values; obs_months = train_df['date'].dt.month.values
    hist_vals = hist_df[h_col].values; hist_months = hist_df['date'].dt.month.values
    fut_vals = fut_df[f_col].values; fut_months = fut_df['date'].dt.month.values
    corrected = np.zeros(len(fut_vals))
    for m in range(1, 13):
        o_m = np.sort(obs_vals[obs_months == m]); h_m = np.sort(hist_vals[hist_months == m])
        f_idx = (fut_months == m); f_m = fut_vals[f_idx]
        if len(o_m) == 0 or len(h_m) == 0 or len(f_m) == 0: corrected[f_idx] = f_m; continue
        ranks = np.searchsorted(h_m, f_m) / len(h_m)
        corrected[f_idx] = np.quantile(o_m, np.clip(ranks, 0, 0.9999))
    return corrected

# -------------------------------------------------------------------------
# 2. 🌊 IPCC SEA LEVEL RISE MULTIPLIER
# -------------------------------------------------------------------------
def get_ipcc_slr_multiplier(year, scenario, geom):
    try:
        elevation_img = ee.Image("NASA/NASADEM_HGT/001").select('elevation')
        local_elev = elevation_img.reduceRegion(
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
# 3. 🚀 ULTRA-PARALLEL WRI SAMPLING
# -------------------------------------------------------------------------
def safe_float(props, key):
    v = props.get(key)
    if v is None or str(v).lower() in ['none', 'null', 'nan'] or pd.isna(v):
        return np.nan
    try:
        v = float(v)
        if abs(v) > 5:
            return np.nan
        return v
    except:
        return np.nan

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
                'Drought Risk': 0.0
            }
    except:
        return None

def get_wri_4_directions_parallel(geom, year=None, scenario_name=None, use_baseline_bau30=False):
    start_time = time.time()
    lon, lat = geom.coordinates().getInfo()

    if use_baseline_bau30:
        coll = ee.FeatureCollection("WRI/Aqueduct_Water_Risk/V4/baseline_annual")
    else:
        coll = ee.FeatureCollection("WRI/Aqueduct_Water_Risk/V4/future_annual")

    center_start = time.time()
    center_pt = ee.Geometry.Point([lon, lat])
    center_result = sample_nearest_wri(coll, center_pt, use_baseline_bau30, year, scenario_name)
    center_time = time.time() - center_start

    system_locked = False
    if use_baseline_bau30 and center_result and system_metrics_invalid(center_result):
        try:
            bau_coll = ee.FeatureCollection("WRI/Aqueduct_Water_Risk/V4/future_annual")
            bau_2030 = sample_nearest_wri(
                bau_coll,
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

    if center_result and sum(not pd.isna(center_result.get(k)) for k in WRI_EVENT_KEYS) >= 1:
        center_result['_source'] = 'CENTER'
        total_time = time.time() - start_time
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(sample_nearest_wri, coll, pt, use_baseline_bau30, year, scenario_name):
                   name for name, pt in all_points.items()}

        all_results = {}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            result = future.result()
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

def get_all_wri_parallel(geom):
    total_start = time.time()

    wri_configs = []
    wri_configs.append(('baseline', None, None, True))

    for scenario in ['Trend', 'ssp245', 'ssp585']:
        for ep, mid in EPOCH_MIDPOINTS.items():
            scenario_name = f'{scenario}_{ep}'
            wri_configs.append((scenario_name, mid, scenario_name, False))

    all_wri_results = {}
    timings = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        for config_idx, config in enumerate(wri_configs):
            scenario_name, year, scen_name, is_base = config
            futures[executor.submit(get_wri_4_directions_parallel, geom, year, scen_name, is_base)] = config_idx

        for future in concurrent.futures.as_completed(futures):
            config_idx = futures[future]
            scenario_name, year, scen_name, is_base = wri_configs[config_idx]
            result, timing_info = future.result()

            result['_method'] = timing_info['method']
            all_wri_results[scenario_name] = result
            timings[scenario_name] = timing_info

    return all_wri_results, timings

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
# 4. MAIN API FUNCTION
# -------------------------------------------------------------------------
def run_for_point(lat: float, lon: float):
    try:
        ensure_ee_initialized()
    except HazardError:
        raise
    
    total_start = time.time()
    final_rows = []
    WRI_WATER_HAZARDS = ['Water Stress', 'Drought Risk', 'Seasonal Variability', 'Interannual Variability']

    print(f"🌍 Processing ({lat:.4f}, {lon:.4f})...")
    geom = ee.Geometry.Point([lon, lat])
    
    # Climate data fetching (parallel)
    climate_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(get_full_series, "ECMWF/ERA5_LAND/DAILY_AGGR", geom, ERA5_RANGE[0], ERA5_RANGE[1],
                             ['temperature_2m_max', 'temperature_2m_min', 'temperature_2m', 'total_precipitation_sum'])
        f2 = executor.submit(get_full_series, "NASA/GDDP-CMIP6", geom, CMIP6_HIST_RANGE[0], CMIP6_HIST_RANGE[1],
                             ['tasmax', 'tasmin', 'tas', 'pr'], MODEL, 'historical')
        f3 = executor.submit(get_full_series, "NASA/GDDP-CMIP6", geom, CMIP6_FUT_RANGE[0], CMIP6_FUT_RANGE[1],
                             ['tasmax', 'tasmin', 'tas', 'pr'], MODEL, 'ssp245')
        f4 = executor.submit(get_full_series, "NASA/GDDP-CMIP6", geom, CMIP6_FUT_RANGE[0], CMIP6_FUT_RANGE[1],
                             ['tasmax', 'tasmin', 'tas', 'pr'], MODEL, 'ssp585')
        datasets = {'era5': f1.result(), 'hist': f2.result(), 'ssp245': f3.result(), 'ssp585': f4.result()}
        if datasets['era5'].empty:
            raise HazardError(
                stage="ERA5_FETCH",
                message="ERA5 returned empty dataframe"
            )
    climate_time = time.time() - climate_start

    # ULTRA-PARALLEL: ALL 10 WRI scenarios at once
    wri_start = time.time()
    all_wri, wri_timings = get_all_wri_parallel(geom)
    
    if 'baseline' not in all_wri or not all_wri['baseline']:
        raise HazardError(
            stage="WRI_FETCH",
            message="WRI baseline returned no data"
        )
        
    wri_base = all_wri['baseline']

    drr_base_score = wri_base.get('Drought Risk', np.nan)
    baseline_method = wri_base.get('_method', 'UNKNOWN')
    wri_time = time.time() - wri_start

    rfr_base = wri_base.get('Riverine Flood Risk')

    if rfr_base is None or pd.isna(rfr_base):
        rfr_base = np.nan

    cfr_base = wri_base.get('Coastal Flood Risk')
    if pd.isna(cfr_base):
        cfr_base = np.nan
    
    geom_fc = geom.buffer(100_000)
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

    if not coastal_flag:
        cy_base = np.nan
    
    elif storm_count is None or pd.isna(storm_count) or storm_count < 1:
        cy_base = np.nan
    
    else:
        cy_wind_kmh = max_wind_knots * 1.852
        rp_years = 34 / storm_count
    
        damage_potential = (cy_wind_kmh / 150.0) ** 3
    
        cy_base = (
            np.clip(damage_potential * 5.0, 0.5, 5.0) * 0.8 +
            np.clip(15.0 / rp_years, 0.5, 5.0) * 0.2
        )
    # ERA5 processing
    era5 = datasets['era5']; era5['year'] = era5['date'].dt.year
    era5_base = era5[(era5['date'] >= '1960-01-01') & (era5['date'] <= '2014-12-31')]
    era5_temp_base = era5_base['temperature_2m'].mean()
    
    era5_pr_base = era5_base.groupby(
        era5_base['date'].dt.year
    )['pr'].sum().mean()
    
    if pd.isna(era5_pr_base) or era5_pr_base == 0:
        era5_pr_base = 1e-6

    hist_tmax = era5_base['temperature_2m_max']
    hist_tmin = era5_base['temperature_2m_min']

    heat_threshold = hist_tmax.quantile(0.95)
    cold_threshold = hist_tmin.quantile(0.05)

    trends = {}
    for m_name, col, func in [('max_t','temperature_2m_max','max'),('min_t','temperature_2m_min','min'),('mean_t','temperature_2m','mean'),
                              ('pr_sum','pr','sum'),('max_pr','pr','max'),('days_35','temperature_2m_max', lambda x: (x > 35).sum()),
                              ('days_0', 'temperature_2m_min', lambda x: (x < 0).sum())]:
        ann = era5.groupby('year')[col].agg(func).reset_index()
        slope, icept, _, _, _ = stats.linregress(ann['year'], ann[col])
        trends[m_name] = {'slope': slope, 'intercept': icept}

    coastal_flag = is_coastal(geom)

    def add_row(col_name, metrics, year_proj=None):
        is_obs = 'Observed' in col_name
        pr_pct, t_anom = metrics.get('pr_change_pct', 0.0), metrics.get('anom', 0.0)

        for h, s in [('Extreme Heat', metrics['s_max_t']), ('Chronic Heat Stress', metrics['s_days_35']),
                     ('Extreme Cold', metrics['s_min_t']), ('Chronic Cold Stress', metrics['s_days_0']),
                     ('Temperature Anomaly', metrics['s_anom']), ('Precipitation Change', metrics['s_pr_change']),
                     ('Extreme Precipitation', metrics['s_max_pr'])]:
            final_rows.append({'Hazard': h, 'Column': col_name, 'Score': round(s, 3)})

        scenario_key = 'baseline' if is_obs else col_name
        wri_scenario = all_wri.get(scenario_key, wri_base)

        for h in WRI_WATER_HAZARDS:

            # --- Drought Risk ---
            if h == 'Drought Risk':
                if pd.isna(drr_base_score):
                    s = np.nan
                elif is_obs:
                    s = drr_base_score
                else:
                    drought_mult = max(
                        0.7,
                        (1 - pr_pct / 100) + (t_anom / 4.0)
                    )
                    s = min(5.0, drr_base_score * drought_mult)

            # --- Other WRI hazards ---
            else:
                base_s = wri_base.get(h)
                if is_obs:
                    s = base_s if not pd.isna(base_s) else np.nan
                else:
                    fut_s = wri_scenario.get(h, base_s)
                    s = fut_s if not pd.isna(fut_s) else base_s

            final_rows.append({
                'Hazard': h,
                'Column': col_name,
                'Score': s
            })


        # --- Riverine Flood Risk (always applicable) ---
        if pd.isna(rfr_base):
            riverine_s = np.nan
        elif is_obs:
            riverine_s = rfr_base
        else:
            riverine_s = min(
                5.0,
                rfr_base * max(0.8, 1 + (pr_pct / 100))
        )

        # --- Coastal Flood Risk (ONLY if coastal) ---
        

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

        if is_obs:
            s_wf = wf_base
            s_cy = cy_base
        else:
            wf_sens = 0.16
            cy_sens = 0.08

            s_wf = wf_base * (1 + t_anom * wf_sens)
            s_cy = np.nan if pd.isna(cy_base) else cy_base * (1 + t_anom * cy_sens)


        s_wf = round(min(5.0, s_wf), 3)
        s_cy = round(min(5.0, s_cy), 3)

        final_rows.append({'Hazard': 'Wildfire', 'Column': col_name, 'Score': s_wf})
        final_rows.append({'Hazard': 'Cyclone', 'Column': col_name, 'Score': s_cy})

    # Execute Observed 2024
    obs_start = time.time()
    obs_dec = era5[(era5['year'] >= 2015) & (era5['year'] <= 2024)]

    if not obs_dec.empty:
        dec_heat = (obs_dec['temperature_2m_max'] > heat_threshold).sum()
        dec_cold = (obs_dec['temperature_2m_min'] < cold_threshold).sum()

        m = {
            's_max_t': score(obs_dec['temperature_2m_max'].max(), 30, 50),
            's_days_35': score(dec_heat / 10, 5, 150),
            's_min_t': score(obs_dec['temperature_2m_min'].min(), -30, 5, True),
            's_days_0': score(dec_cold / 10, 1, 90),
            's_anom': score(obs_dec['temperature_2m'].mean() - era5_temp_base, 0, 4),
            'pr_change_pct': ((obs_dec['pr'].sum()/10 - era5_pr_base) / era5_pr_base) * 100,
            's_pr_change': score(abs(((obs_dec['pr'].sum()/10 - era5_pr_base) / era5_pr_base) * 100), 10, 50),
            's_max_pr': score(obs_dec['pr'].max(), 30, 200)
        }
    else:
        m = {
            's_max_t': 1.0, 's_days_35': 1.0, 's_min_t': 1.0, 's_days_0': 1.0,
            's_anom': 0.0, 'pr_change_pct': 0.0, 's_pr_change': 0.0, 's_max_pr': 1.0
        }

    add_row('Observed_2024', m)
    obs_time = time.time() - obs_start

    # Execute scenarios
    scenarios_start = time.time()
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
                    's_max_pr': score(max_pr_proj, 30, 200)
                }
            else:
                df = datasets[scenario]
                sdf = df[(df['date'].dt.year >= sy) & (df['date'].dt.year <= ey)]
                if sdf.empty: continue
                mx = perform_qm(era5_base, datasets['hist'], sdf, 'temperature_2m_max', 'tasmax', 'tasmax')
                mn = perform_qm(era5_base, datasets['hist'], sdf, 'temperature_2m_min', 'tasmin', 'tasmin')
                av = perform_qm(era5_base, datasets['hist'], sdf, 'temperature_2m', 'tas', 'tas')
                pr = perform_qm(era5_base, datasets['hist'], sdf, 'pr', 'pr', 'pr')
                t_anom = np.mean(av) - era5_temp_base
                pr_change = ((np.mean(pr)*365.25 - era5_pr_base) / era5_pr_base) * 100
                dec_heat = np.sum(mx > heat_threshold)
                dec_cold = np.sum(mn < cold_threshold)

                m = {'s_max_t': score(np.max(mx), 30, 50),
                     's_min_t': score(np.min(mn), -30, 5, True),
                     's_days_35': score(dec_heat / 10, 5, 150),
                     's_days_0': score(dec_cold / 10, 1, 90),
                     'anom': t_anom, 's_anom': score(t_anom, 0, 5),
                     'pr_change_pct': pr_change, 's_pr_change': score(abs(pr_change), 10, 50),
                     's_max_pr': score(np.max(pr), 30, 200)}
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
    matrix_time = time.time() - matrix_start

    total_time = time.time() - total_start
    logging.info(f"✅ Hazards computed | Shape: {df_final.shape} | Time: {total_time:.1f}s")

    df_final = df_final.replace([np.inf, -np.inf, np.nan], None)
    return df_final

                                     








