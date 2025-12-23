from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
import os
import sys
import json
import glob

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.inference.predictor import Predictor

app = FastAPI(title="Stock Prediction API (Enterprise)", version="1.0.0")

from fastapi.middleware.cors import CORSMiddleware

# Strict CORS for production, permissive for local dev
origins = [
    "http://localhost:5173", # Vite local
    "http://localhost:3000", # React standard
    "http://localhost:8000", # Self
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
predictor = None
MODEL_DIR = "models"
LATEST_POINTER = os.path.join(MODEL_DIR, "latest.json")

# Data Models (Strict Contracts)
class ModelInfo(BaseModel):
    version: str
    trained_on: str
    features_count: int

class PredictionRequest(BaseModel):
    symbol: str = "TSLA"
    model_version: Optional[str] = None # Optional request for specific version
    model_config = ConfigDict(protected_namespaces=())

class PredictionResponse(BaseModel):
    symbol: str
    prediction: float
    confidence_interval: List[float] # Mocked for now, but part of contract
    model_version: str
    generated_at: str
    model_config = ConfigDict(protected_namespaces=())

@app.on_event("startup")
async def startup_event():
    load_latest_model()

def load_latest_model():
    global predictor
    try:
        if not os.path.exists(LATEST_POINTER):
            print(f"⚠️ No 'latest.json' found in {MODEL_DIR}. Model not loaded.")
            return

        with open(LATEST_POINTER, "r") as f:
            pointer = json.load(f)
            version = pointer.get("version")
            
        load_model_version(version)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

def load_model_version(version: str):
    global predictor
    version_dir = os.path.join(MODEL_DIR, version)
    if not os.path.exists(version_dir):
        raise FileNotFoundError(f"Version {version} not found.")
    
    print(f"🔄 Loading model version: {version}...")
    predictor = Predictor.load(version_dir)
    print(f"✅ Model {version} loaded successfully.")

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "model_loaded": predictor is not None,
        "model_version": predictor.config.data.symbol if predictor else None # Using symbol/metadata as proxy for ID if needed, or store ID in predictor
    }

@app.get("/models", response_model=List[ModelInfo])
def list_models():
    """List all available model versions."""
    versions = []
    # Scan models directory
    # Structure: models/vYYYYMMDD_HHMMSS/metadata.json
    version_dirs = glob.glob(os.path.join(MODEL_DIR, "v*"))
    
    for v_dir in version_dirs:
        meta_path = os.path.join(v_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                versions.append(ModelInfo(
                    version=meta["model_version"],
                    trained_on=meta["trained_on"],
                    features_count=len(meta["config"]["features"])
                ))
    
    # Sort by version desc
    versions.sort(key=lambda x: x.version, reverse=True)
    return versions

@app.get("/metadata")
def get_metadata():
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "features": predictor.config.features,
        "sequence_length": predictor.config.sequence_length,
        "scaler": predictor.config.scaler
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # In strict enterprise setting, if request.model_version is specified 
    # and != current, we might reload or reject. For now, we note if mismatch.
    
    try:
        from src.data.loader import DataLoader
        
        # Use our robust loader
        loader = DataLoader(request.symbol, years=2)
        raw_df = loader.fetch()
        
        result_dict = predictor.predict(raw_df)
        
        # Predictor returns {prediction, last_date, model_version}
        # We wrap it in strict response
        
        # Mock confidence interval (e.g., +/- 5%)
        pred_val = result_dict["prediction"]
        ci = [pred_val * 0.95, pred_val * 1.05]
        
        return PredictionResponse(
            symbol=request.symbol,
            prediction=pred_val,
            confidence_interval=ci,
            model_version=result_dict["model_version"],
            generated_at=result_dict["last_date"]
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
