from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    UploadFile,
    File,
    Form,
)
from fastapi.responses import JSONResponse
from pathlib import Path
import os
import uuid
import base64
import io
import secrets

from PIL import Image
import numpy as np
import rasterio

from api.inference import (
    predict,
    calculate_metrics,
    clear_model_cache,
    MODEL_INFO,
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="SpaceNet 6 Building Extraction API",
    description="Inference API for SAR and Optical building-footprint models.",
    version="1.0.0",
)


# ============================================================
# DIRECTORIES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UPLOAD_DIR = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "api_uploads"
)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LIMITS AND AUTH
# ============================================================

MAX_UPLOAD_BYTES = int(
    os.environ.get("SN6_MAX_UPLOAD_MB", "64")
) * 1024 * 1024

# When SN6_API_KEY is set, /predict requires a matching
# X-API-Key header. When it is unset the API is open and local
# only, which is the development default.
API_KEY = os.environ.get("SN6_API_KEY", "").strip()


# ============================================================
# HELPERS
# ============================================================

def require_api_key(
    x_api_key: str | None = Header(default=None),
):

    if not API_KEY:
        return

    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing X-API-Key header.",
        )

    if not secrets.compare_digest(
        x_api_key,
        API_KEY
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )


def save_upload(
    upload: UploadFile,
    suffix: str = ".tif"
) -> Path:
    """
    Save an uploaded file temporarily, enforcing a size limit.
    """
    filename = f"{uuid.uuid4().hex}{suffix}"
    path = UPLOAD_DIR / filename

    size = 0

    try:
        with path.open("wb") as buffer:
            while chunk := upload.file.read(
                1024 * 1024
            ):
                size += len(chunk)

                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Upload exceeds the "
                            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
                        ),
                    )

                buffer.write(chunk)

    except Exception:
        path.unlink(missing_ok=True)
        raise

    if size == 0:
        path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    return path


def array_to_png_base64(array: np.ndarray) -> str:
    """
    Convert a binary mask or grayscale array to base64 PNG.
    """

    array = np.asarray(array)

    if array.dtype != np.uint8:
        array = (array > 0).astype(np.uint8) * 255

    image = Image.fromarray(array)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def create_overlay(
    input_array: np.ndarray,
    prediction: np.ndarray,
) -> str:
    """
    Create a simple RGB overlay.

    Prediction buildings are shown in red.
    """

    input_array = np.asarray(input_array)

    # Use first 3 channels if available
    if input_array.ndim == 3:
        if input_array.shape[0] >= 3:
            image = input_array[:3]
        else:
            image = np.repeat(input_array[:1], 3, axis=0)

        image = np.transpose(image, (1, 2, 0))

    else:
        image = input_array

        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)

    # Normalize for visualization
    image = image.astype(np.float32)

    for channel in range(3):
        c = image[..., channel]

        valid = np.isfinite(c)

        if valid.any():
            low = np.percentile(c[valid], 2)
            high = np.percentile(c[valid], 98)

            if high > low:
                c = (c - low) / (high - low)

            c = np.clip(c, 0, 1)

        image[..., channel] = c

    image = (image * 255).astype(np.uint8)

    # Red prediction overlay
    mask = prediction.astype(bool)

    image[mask, 0] = 255
    image[mask, 1] = image[mask, 1] // 3
    image[mask, 2] = image[mask, 2] // 3

    result = Image.fromarray(image)

    buffer = io.BytesIO()
    result.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "SpaceNet 6 Building Extraction API",
        "status": "running",
        "docs": "/docs",
    }


# ============================================================
# AVAILABLE MODELS
# ============================================================

@app.get("/models")
def available_models():
    return MODEL_INFO


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
async def run_prediction(
    model: str = Form(...),

    sar_image: UploadFile | None = File(None),

    optical_image: UploadFile | None = File(None),

    ground_truth: UploadFile | None = File(None),

    _: None = Depends(require_api_key),
):
    """
    Run building-footprint prediction.

    Required input depends on selected model:

    E1 → SAR
    E2 → SAR
    E3 → Optical
    E4 → SAR

    Ground truth is optional and enables metric calculation.
    """

    # --------------------------------------------------------
    # Validate model
    # --------------------------------------------------------

    model = model.upper()

    if model not in MODEL_INFO:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Choose from E1, E2, E3, E4.",
        )

    modality = MODEL_INFO[model]["modality"]

    # --------------------------------------------------------
    # Validate required input
    # --------------------------------------------------------

    if modality == "SAR" and sar_image is None:
        raise HTTPException(
            status_code=400,
            detail=f"{model} requires a SAR GeoTIFF.",
        )

    if modality == "Optical" and optical_image is None:
        raise HTTPException(
            status_code=400,
            detail=f"{model} requires an Optical GeoTIFF.",
        )

    # --------------------------------------------------------
    # Save uploaded files
    # --------------------------------------------------------

    sar_path = None
    optical_path = None
    gt_path = None

    try:

        if sar_image is not None:
            sar_path = save_upload(sar_image)

        if optical_image is not None:
            optical_path = save_upload(optical_image)

        if ground_truth is not None:
            gt_path = save_upload(ground_truth)

        # ----------------------------------------------------
        # Run model
        # ----------------------------------------------------

        result = predict(
            model_name=model,
            sar_path=str(sar_path) if sar_path else None,
            optical_path=str(optical_path) if optical_path else None,
        )

        prediction = result["prediction"]
        probability = result["probability"]
        input_array = result["input"]

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = None

        if gt_path is not None:

            with rasterio.open(gt_path) as src:
                gt = src.read(1)

            gt = (gt > 0).astype(np.uint8)

            # Resize/check dimensions if necessary
            if gt.shape != prediction.shape:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Ground truth shape {gt.shape} does not match "
                        f"prediction shape {prediction.shape}."
                    ),
                )

            metrics = calculate_metrics(
                prediction,
                gt,
            )

        # ----------------------------------------------------
        # Convert outputs for frontend
        # ----------------------------------------------------

        prediction_png = array_to_png_base64(
            prediction.astype(np.uint8) * 255
        )

        probability_png = array_to_png_base64(
            (probability * 255).astype(np.uint8)
        )

        overlay_png = create_overlay(
            input_array,
            prediction,
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return JSONResponse(
            content={
                "success": True,
                "model": model,
                "model_name": result["model_name"],
                "modality": result["modality"],

                "prediction": {
                    "height": int(prediction.shape[0]),
                    "width": int(prediction.shape[1]),
                    "positive_pixels": int(prediction.sum()),
                    "building_percentage": float(
                        prediction.mean() * 100
                    ),
                },

                "metrics": metrics,

                "images": {
                    "prediction_png": prediction_png,
                    "probability_png": probability_png,
                    "overlay_png": overlay_png,
                },
            }
        )

    finally:

        # ----------------------------------------------------
        # Cleanup temporary uploads
        # ----------------------------------------------------

        for path in [sar_path, optical_path, gt_path]:

            if path is not None:

                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "auth_required": bool(API_KEY),
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
    }


# ============================================================
# MODEL CACHE
# ============================================================

@app.post("/models/{model_name}/unload")
def unload_model(
    model_name: str,
    _: None = Depends(require_api_key),
):
    """
    Release cached checkpoints to free device and host memory.
    """
    released = clear_model_cache()

    return {
        "success": True,
        "released": released,
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )