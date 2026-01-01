from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import numpy as np
import time
from contextlib import asynccontextmanager
from hazards import run_for_point
import os
import json
import ee
import tempfile

# =================================================
# LOGGING
# =================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("climate-api")

# =================================================
# EARTH ENGINE INITIALIZATION (RENDER-SAFE)
# =================================================
_EE_INITIALIZED = False

def init_ee():
    global _EE_INITIALIZED
    if _EE_INITIALIZED:
        return

    if "GCP_SERVICE_ACCOUNT" not in os.environ:
        raise RuntimeError("Missing GCP_SERVICE_ACCOUNT environment variable")

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

    logger.info("🌍 Earth Engine initialized")
    _EE_INITIALIZED = True

# =================================================
# FASTAPI LIFESPAN
# =================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Climate Hazard API starting")
    try:
        init_ee()
    except Exception as e:
        logger.exception("❌ Earth Engine init failed")
        raise RuntimeError("Earth Engine init failed — check Render env vars") from e
    yield
    logger.info("🛑 Climate Hazard API shutting down")

# =================================================
# RESPONSE MODELS
# =================================================
class HazardResponse(BaseModel):
    success: bool
    city: Optional[str]
    lat: float
    lon: float
    columns: List[str]
    data: List[Dict[str, Any]]
    shape: List[int]
    hazards_count: int
    computation_time_ms: float

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"

# =================================================
# FASTAPI APP
# =================================================
app = FastAPI(
    title="🌍 Climate Hazard Matrix API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount UI only if present (Render-safe)
if os.path.isdir("static"):
    app.mount("/ui", StaticFiles(directory="static", html=True), name="static")

# =================================================
# ROUTES
# =================================================
@app.get("/")
async def root():
    return {
        "message": "Climate Hazard Matrix API",
        "endpoints": {
            "health": "/health",
            "run": "/run?lat=25.59&lon=85.14&city=Patna",
            "ui": "/ui",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()

@app.get("/run", response_model=HazardResponse)
async def run_hazards(lat: float, lon: float, city: Optional[str] = None):
    if not (-90 <= lat <= 90):
        raise HTTPException(400, "Latitude must be between -90 and 90")
    if not (-180 <= lon <= 180):
        raise HTTPException(400, "Longitude must be between -180 and 180")

    start = time.time()
    logger.info(f"📍 Computing hazards for {lat:.4f}, {lon:.4f}")

    try:
        df = run_for_point(lat, lon)

        # Clean dataframe → JSON safe
        df = df.replace([np.inf, -np.inf, np.nan], None)

        cols = ["Hazard"] + [c for c in df.columns if c != "Hazard"]
        records = json.loads(json.dumps(df[cols].to_dict(orient="records")))

        elapsed_ms = (time.time() - start) * 1000

        return {
            "success": True,
            "city": city,
            "lat": float(lat),
            "lon": float(lon),
            "columns": cols,
            "data": records,
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "hazards_count": int(df.shape[0]),
            "computation_time_ms": round(elapsed_ms, 1)
        }

    except Exception as e:
        logger.exception("❌ Hazard computation failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(404)
async def not_found(request: Request, exc):
    return {"error": "Endpoint not found"}

# =================================================
# LOCAL RUN
# =================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)


