from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import ee

# =================================================
# 1. INITIALIZE EARTH ENGINE (SERVICE ACCOUNT)
# =================================================

SERVICE_ACCOUNT = "gee-backend@citric-snow-424111-q0.iam.gserviceaccount.com"
KEY_FILE = "service_account.json"

credentials = ee.ServiceAccountCredentials(
    SERVICE_ACCOUNT,
    KEY_FILE
)

ee.Initialize(
    credentials,
    project="citric-snow-424111-q0",
    url="https://earthengine-highvolume.googleapis.com"
)

# =================================================
# 2. LOAD AQUEDUCT DATASET
# =================================================

AQUEDUCT_FC = ee.FeatureCollection(
    "WRI/Aqueduct_Water_Risk/V4/baseline_annual"
)

# =================================================
# 3. FASTAPI APP
# =================================================

app = FastAPI(title="Climate Risk Tool")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =================================================
# 4. SERVE FRONTEND
# =================================================

@app.get("/")
def home():
    return FileResponse("static/index.html")

# =================================================
# 5. HELPER: NEAREST VALID BASIN LOOKUP
# =================================================

def nearest_valid_value(lat, lon, field, search_radius_m=100_000):
    """
    Atlas-aligned lookup:
    - buffer search
    - ignore -9999
    - nearest valid basin
    """

    point = ee.Geometry.Point([lon, lat])

    nearby = AQUEDUCT_FC.filterBounds(
        point.buffer(search_radius_m)
    ).filter(
        ee.Filter.neq(field, -9999)
    )

    basin = nearby.map(
        lambda f: f.set(
            "dist", f.geometry().distance(point)
        )
    ).sort("dist").first()

    value = ee.Algorithms.If(basin, basin.get(field), None)
    return value.getInfo()

# =================================================
# 6. NEW AQUEDUCT SCORE GENERATOR (ATLAS-ALIGNED)
# =================================================

def get_aqueduct_risks(lat: float, lon: float):
    """
    Returns Aqueduct risks consistent with Aqueduct Atlas UI
    """

    return {
        # Baseline Water Stress (CATEGORY: 1–5)
        "water_stress_category": nearest_valid_value(lat, lon, "bws_cat"),

        # Drought Risk (0–5)
        "drought_risk_score": nearest_valid_value(lat, lon, "drr_score"),

        # River Flood Risk (0–5)
        "river_flood_risk_score": nearest_valid_value(lat, lon, "rfr_score"),

        # Coastal Flood Risk (0–5)
        "coastal_flood_risk_score": nearest_valid_value(lat, lon, "cfr_score"),

        # Metadata
        "source": "WRI Aqueduct v4 (Atlas-aligned)",
        "units": "0–5 scale (category for water stress)"
    }

# =================================================
# 7. API ENDPOINT
# =================================================

@app.get("/risk")
def risk(lat: float, lon: float):
    scores = get_aqueduct_risks(lat, lon)

    return {
        "latitude": lat,
        "longitude": lon,
        "scores": scores
    }
