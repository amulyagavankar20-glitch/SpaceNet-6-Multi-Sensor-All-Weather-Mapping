from collections import OrderedDict
from pathlib import Path
import os
import threading

import numpy as np
import rasterio
import torch

from api.models import (
    load_e1_model,
    load_e2_model,
    load_e3_model,
    load_e4_model,
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CROP_SIZE = 512


# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_INFO = {
    "E1": {
        "name": "E1 - SAR U-Net",
        "modality": "SAR",
    },
    "E2": {
        "name": "E2 - SAR ResNet34-U-Net",
        "modality": "SAR",
    },
    "E3": {
        "name": "E3 - Optical Teacher",
        "modality": "Optical",
    },
    "E4": {
        "name": "E4 - SAR + Knowledge Distillation",
        "modality": "SAR",
    },
}


# ============================================================
# MODEL CACHE
#
# Checkpoints are ~100 MB each, so only a bounded number are kept
# resident. Access is guarded by a lock because the API runs in a
# multi-threaded event loop and several requests can ask for the
# same model at once.
# ============================================================

MODEL_CACHE_SIZE = int(
    os.environ.get("SN6_MODEL_CACHE_SIZE", "1")
)

_LOADERS = {
    "E1": load_e1_model,
    "E2": load_e2_model,
    "E3": load_e3_model,
    "E4": load_e4_model,
}

_MODELS = OrderedDict()

_MODEL_LOCK = threading.Lock()


def get_model(model_name: str):

    model_name = model_name.upper()

    if model_name not in MODEL_INFO:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(MODEL_INFO.keys())}"
        )

    with _MODEL_LOCK:

        if model_name in _MODELS:

            _MODELS.move_to_end(model_name)

            return _MODELS[model_name]

        print(f"Loading {model_name}...")

        model = _LOADERS[model_name]().to(DEVICE)
        model.eval()

        _MODELS[model_name] = model

        while len(_MODELS) > max(MODEL_CACHE_SIZE, 1):

            evicted, _ = _MODELS.popitem(last=False)

            print(f"Evicted {evicted} from model cache")

            del evicted

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        print(f"{model_name} loaded on {DEVICE}")

        return model


def clear_model_cache():

    with _MODEL_LOCK:

        count = len(_MODELS)

        _MODELS.clear()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return count


# ============================================================
# CENTER CROP
# ============================================================

def center_crop(
    array: np.ndarray,
    crop_size: int = CROP_SIZE,
):
    """
    Center crop an image to crop_size x crop_size.

    Supports:
        [C, H, W]
        [H, W]
    """

    if array.ndim == 3:

        _, height, width = array.shape

    elif array.ndim == 2:

        height, width = array.shape

    else:

        raise ValueError(
            f"Unsupported array shape: {array.shape}"
        )

    if height < crop_size or width < crop_size:

        raise ValueError(
            f"Input size {height}x{width} is smaller "
            f"than required crop size {crop_size}x{crop_size}."
        )

    top = (height - crop_size) // 2
    left = (width - crop_size) // 2

    if array.ndim == 3:

        cropped = array[
            :,
            top:top + crop_size,
            left:left + crop_size,
        ]

    else:

        cropped = array[
            top:top + crop_size,
            left:left + crop_size,
        ]

    return cropped, top, left


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input(array: np.ndarray):

    tensor = torch.from_numpy(
        array.astype(np.float32)
    )

    # [C,H,W] -> [1,C,H,W]
    tensor = tensor.unsqueeze(0)

    return tensor.to(DEVICE)


# ============================================================
# PREDICTION
# ============================================================

@torch.inference_mode()
def predict(
    model_name: str,
    sar_path: str | None = None,
    optical_path: str | None = None,
):

    model_name = model_name.upper()

    if model_name not in MODEL_INFO:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    model = get_model(model_name)

    modality = MODEL_INFO[model_name]["modality"]

    # --------------------------------------------------------
    # LOAD INPUT
    # --------------------------------------------------------

    if modality == "SAR":

        if sar_path is None:
            raise ValueError(
                f"{model_name} requires a SAR image."
            )

        from api.preprocessing import load_sar

        array, profile = load_sar(sar_path)

    else:

        if optical_path is None:
            raise ValueError(
                f"{model_name} requires an optical image."
            )

        from api.preprocessing import load_optical

        array, profile = load_optical(optical_path)

    # --------------------------------------------------------
    # CENTER CROP TO 512x512
    # --------------------------------------------------------

    array, top, left = center_crop(
        array,
        CROP_SIZE,
    )

    # --------------------------------------------------------
    # UPDATE GEO PROFILE
    # --------------------------------------------------------

    profile = profile.copy()

    profile["height"] = CROP_SIZE
    profile["width"] = CROP_SIZE

    # Update transform for cropped image
    from rasterio.transform import Affine

    original_transform = profile["transform"]

    profile["transform"] = (
        original_transform
        * Affine.translation(left, top)
    )

    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    tensor = prepare_input(array)

    # --------------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------------

    logits = model(tensor)

    # Some models may return tuple
    if isinstance(logits, tuple):
        logits = logits[0]

    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    probability = torch.sigmoid(logits)

    # --------------------------------------------------------
    # BINARY MASK
    # --------------------------------------------------------

    prediction = (
        probability >= 0.5
    ).float()

    # --------------------------------------------------------
    # REMOVE BATCH/CHANNEL
    # --------------------------------------------------------

    probability = (
        probability[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    prediction = (
        prediction[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    return {
        "prediction": prediction,
        "probability": probability,
        "input": array,
        "profile": profile,
        "model": model,
        "model_name": MODEL_INFO[model_name]["name"],
        "modality": modality,
    }


# ============================================================
# SAVE PREDICTION GEOTIFF
# ============================================================

def save_prediction_geotiff(
    prediction: np.ndarray,
    profile: dict,
    output_path: str,
):

    output_profile = profile.copy()

    output_profile.update(
        {
            "count": 1,
            "dtype": "uint8",
            "nodata": 0,
        }
    )

    with rasterio.open(
        output_path,
        "w",
        **output_profile,
    ) as dst:

        dst.write(
            prediction.astype(np.uint8),
            1,
        )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
):

    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    intersection = np.logical_and(
        prediction,
        ground_truth,
    ).sum()

    prediction_area = prediction.sum()
    ground_truth_area = ground_truth.sum()

    union = np.logical_or(
        prediction,
        ground_truth,
    ).sum()

    dice = (
        2 * intersection
        / (prediction_area + ground_truth_area + 1e-8)
    )

    iou = (
        intersection
        / (union + 1e-8)
    )

    true_positive = intersection

    precision = (
        true_positive
        / (prediction_area + 1e-8)
    )

    recall = (
        true_positive
        / (ground_truth_area + 1e-8)
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SpaceNet 6 inference module")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Crop size: {CROP_SIZE}x{CROP_SIZE}")

    print("\nAvailable models:")

    for key, info in MODEL_INFO.items():

        print(
            f" {key}: "
            f"{info['name']} "
            f"({info['modality']})"
        )

    print("\n✓ Inference module loaded successfully")