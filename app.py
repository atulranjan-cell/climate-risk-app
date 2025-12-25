from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import ee
import os
import json
from google.oauth2 import service_account

# =================================================
# 1. INITIALIZE EARTH ENGINE (RENDER-SAFE)
# =================================================

# Load service account JSON from environment variable
sa_info = json.loads(os.environ["GCP_SERVICE_ACCOUNT"])

credentials = service_account.Credentials.from_service_account_info(
    sa_info,
    scopes=["https://www.googleapis.com/auth/earthengine"]
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
# 6. AQUEDUCT SCORE GENERATOR (ATLAS-ALIGNED)
# =================================================

def get_aqueduct_risks(lat: float, lon: float):
    """
    Returns Aqueduct risks consistent with Aqueduct Atlas UI
    """

    return {
        "water_stress_category": nearest_valid_value(lat, lon, "bws_cat"),
        "drought_risk_score": nearest_valid_value(lat, lon, "drr_score"),
        "river_flood_risk_score": nearest_valid_value(lat, lon, "rfr_score"),
        "coastal_flood_risk_score": nearest_valid_value(lat, lon, "cfr_score"),
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
