# src/serve.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import pandas as pd
from joblib import load
from pathlib import Path
import argparse


class WifiSample(BaseModel):
    """Input model for Wi-Fi RSSI data."""
    rssi: Dict[str, float]  # { "WAP001": -78, "WAP002": -110, ... }
    phone_id: Optional[int] = None


class LocationPrediction(BaseModel):
    """Output model for location predictions."""
    building: int
    floor: int
    x_m: float
    y_m: float
    confidence_building: Optional[float] = None
    confidence_floor: Optional[float] = None


app = FastAPI(
    title="Indoor Localization API",
    description="Wi-Fi fingerprinting based indoor positioning system",
    version="1.0.0"
)

# Global model variables
building_model = None
floor_model = None
xy_model = None
xy_per_building_models = {}
model_type = "knn"  # Default model type

# AP columns from training (will be loaded from model)
WAP_COLS = None


def load_models(model_type_arg="knn"):
    """Load trained models from disk."""
    global building_model, floor_model, xy_model, xy_per_building_models, WAP_COLS, model_type

    model_type = model_type_arg
    models_dir = Path(__file__).parent.parent / "models"

    try:
        building_model = load(models_dir / f"building_{model_type}.joblib")
        floor_model = load(models_dir / f"floor_{model_type}.joblib")
        xy_model = load(models_dir / f"xy_{model_type}_global.joblib")

        # Load per-building models if they exist
        for building_id in [0, 1, 2]:
            model_path = models_dir / f"xy_{model_type}_building_{building_id}.joblib"
            if model_path.exists():
                xy_per_building_models[building_id] = load(model_path)

        # Extract AP columns from the building model
        # This assumes the model has an 'ap' step that stores keep_cols
        has_ap_step = (hasattr(building_model, 'named_steps') and
                      'ap' in building_model.named_steps)
        if has_ap_step:
            WAP_COLS = building_model.named_steps['ap'].keep_cols
        else:
            # Fallback: generate WAP column names
            WAP_COLS = [f"WAP{i:03d}" for i in range(1, 521)]

        print(f"Loaded {model_type.upper()} models from {models_dir}")
        print(f"Using {len(WAP_COLS)} AP features")

    except Exception as e:
        print(f"Error loading models: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load models when the server starts."""
    load_models(model_type)


@app.post("/predict", response_model=LocationPrediction)
async def predict_location(sample: WifiSample):
    """Predict building, floor, and position from Wi-Fi RSSI data.

    Args:
        sample: Wi-Fi RSSI measurements

    Returns:
        Location prediction with building, floor, and coordinates
    """
    if building_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded")

    try:
        # Prepare input data
        rssi_data = sample.rssi

        # Create feature vector with all WAP columns
        features = {}
        for wap in WAP_COLS:
            # Default to -110 for missing APs
            features[wap] = rssi_data.get(wap, -110)

        # Add phone_id if provided (though our baseline models don't use it)
        if sample.phone_id is not None:
            features['PHONEID'] = sample.phone_id

        # Convert to DataFrame
        X = pd.DataFrame([features])

        # Make predictions
        building_pred = int(building_model.predict(X)[0])
        floor_pred = int(floor_model.predict(X)[0])

        # Use building-specific position model if available
        if building_pred in xy_per_building_models:
            xy_pred = xy_per_building_models[building_pred].predict(X)[0]
        else:
            xy_pred = xy_model.predict(X)[0]

        return LocationPrediction(
            building=building_pred,
            floor=floor_pred,
            x_m=float(xy_pred[0]),
            y_m=float(xy_pred[1])
        )

    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"Prediction error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": building_model is not None,
        "model_type": model_type
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Indoor Localization API",
        "version": "1.0.0",
        "model_type": model_type,
        "endpoints": {
            "POST /predict": "Predict location from Wi-Fi RSSI data",
            "GET /health": "Health check"
        }
    }


def main():
    """Main function for running the server."""
    parser = argparse.ArgumentParser(description="Run indoor localization API server")
    parser.add_argument("--model", choices=["knn", "xgb", "mlp"], default="knn",
                       help="Model type to serve (default: knn)")
    parser.add_argument("--host", default="0.0.0.0",
                       help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080,
                       help="Port to bind to (default: 8080)")
    args = parser.parse_args()

    # Set global model type
    global model_type
    model_type = args.model

    print(f"Starting Indoor Localization API server")
    print(f"Model type: {model_type.upper()}")
    print(f"Host: {args.host}:{args.port}")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
