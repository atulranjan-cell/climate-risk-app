from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
import logging
import numpy as np
import time
from contextlib import asynccontextmanager

# Import after logging setup to avoid issues
from hazards import run_for_point, HazardError

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(":earth_africa: Climate Hazard API starting...")
    yield
    logger.info(":earth_africa: Climate Hazard API shutting down...")

class HazardResponse(BaseModel):
    success: bool
    city: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    columns: List[str]
    data: List[Dict[str, Any]]
    shape: Tuple[int, int]
    computation_time_ms: Optional[float] = None
    hazards_count: int

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    hazards: List[str] = [
        "Extreme Heat", "Chronic Heat Stress", "Extreme Cold", "Chronic Cold Stress",
        "Temperature Anomaly", "Precipitation Change", "Extreme Precipitation",
        "Water Stress", "Drought Risk", "Seasonal Variability", "Interannual Variability",
        "Riverine Flood Risk", "Coastal Flood Risk", "Wildfire", "Cyclone"
    ]

app = FastAPI(
    title=":earth_africa: Climate Hazard Matrix API",
    description="Compute 15+ climate hazard scores for any lat/lon using Google Earth Engine + WRI Aqueduct",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/ui", StaticFiles(directory="static", html=True), name="static")

@app.get("/")
async def root():
    return {
        "message": ":earth_africa: Climate Hazard Matrix API",
        "endpoints": {
            "health": "/health",
            "run": "/run?lat=25.59&lon=85.14&city=Patna",
            "ui": "/ui",
            "docs": "/docs"
        },
        "example": "GET /run?lat=25.5941&lon=85.1376&city=Patna"
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()

@app.get("/run", response_model=HazardResponse)
async def run_hazards(lat: float, lon: float, city: Optional[str] = None):
    """
    Compute climate hazard matrix for lat/lon
    """
    if not (-90 <= lat <= 90):
        raise HTTPException(status_code=400, detail="Latitude must be between -90 and 90")
    if not (-180 <= lon <= 180):
        raise HTTPException(status_code=400, detail="Longitude must be between -180 and 180")
    
    start_time = time.time()
    logger.info(f":rocket: Computing hazards for {lat:.4f}, {lon:.4f} ({city or 'unknown'})")
    
    try:
        df = run_for_point(lat, lon)
        df = df.replace([float("inf"), float("-inf"), np.nan], None)
        
        cols = ["Hazard"] + [c for c in df.columns if c != "Hazard"]
        computation_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f":white_check_mark: Hazards computed in {computation_time_ms:.1f}ms | Shape: {df.shape}")
        
        return {
            "success": True,
            "city": city,
            "lat": lat,
            "lon": lon,
            "columns": cols,
            "data": df[cols].to_dict(orient="records"),
            "shape": df.shape,
            "hazards_count": df.shape[0],
            "computation_time_ms": round(computation_time_ms, 1)
        }
    
    except HazardError as he:
        logger.error(f"HazardError [{he.stage}]: {he.message}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "HAZARD_ERROR",
                "stage": he.stage,
                "message": he.message
            }
        )
    
    except Exception as e:
        logger.exception("Unhandled backend error")
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": "UNKNOWN",
                "message": "Unexpected server error"
            }
        )

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception):
    return {"error": "Endpoint not found. Try /docs, /ui, /health, or /run?lat=25.59&lon=85.14"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
