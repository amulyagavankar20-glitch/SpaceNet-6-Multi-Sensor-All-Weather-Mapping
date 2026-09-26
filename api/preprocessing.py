from pathlib import Path
import json
import numpy as np
import rasterio


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "preprocessing_config.json"


# ============================================================
# LOAD TRAINING PREPROCESSING STATISTICS
# ============================================================

with open(CONFIG_PATH, "r") as f:
    preprocessing_config = json.load(f)


# Support the verified statistics file if present.
STATS_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "verified_preprocessing_stats.json"
)

if STATS_PATH.exists():
    with open(STATS_PATH, "r") as f:
        preprocessing_stats = json.load(f)
else:
    preprocessing_stats = preprocessing_config


# ============================================================
# SAR STATISTICS
# ============================================================

SAR_MEAN = np.array(
    preprocessing_stats["sar"]["mean"],
    dtype=np.float32
)

SAR_STD = np.array(
    preprocessing_stats["sar"]["std"],
    dtype=np.float32
)

SAR_P2 = np.array(
    preprocessing_stats["sar"]["clip_p2"],
    dtype=np.float32
)

SAR_P98 = np.array(
    preprocessing_stats["sar"]["clip_p98"],
    dtype=np.float32
)

SAR_EPS = float(
    preprocessing_stats["sar"].get("epsilon", 1e-6)
)


# ============================================================
# OPTICAL STATISTICS
# ============================================================

OPT_MEAN = np.array(
    preprocessing_stats["optical"]["mean"],
    dtype=np.float32
)

OPT_STD = np.array(
    preprocessing_stats["optical"]["std"],
    dtype=np.float32
)

OPT_P2 = np.array(
    preprocessing_stats["optical"]["clip_p2"],
    dtype=np.float32
)

OPT_P98 = np.array(
    preprocessing_stats["optical"]["clip_p98"],
    dtype=np.float32
)


# ============================================================
# SAR PREPROCESSING
# ============================================================

def preprocess_sar(sar):
    """
    Apply the exact SAR preprocessing used during training.

    Raw SAR
        ↓
    nodata handling
        ↓
    dB conversion
        ↓
    P2-P98 clipping
        ↓
    training-set z-score normalization
        ↓
    invalid pixels = 0

    Input:
        SAR array [4, H, W]

    Output:
        float32 array [4, H, W]
    """

    sar = np.asarray(sar, dtype=np.float32)

    if sar.ndim != 3:
        raise ValueError(
            f"SAR must have shape [C,H,W], got {sar.shape}"
        )

    if sar.shape[0] != 4:
        raise ValueError(
            f"SAR must have 4 channels, got {sar.shape[0]}"
        )

    output = np.zeros_like(sar, dtype=np.float32)

    for c in range(4):

        channel = sar[c]

        valid = (
            np.isfinite(channel)
            & (channel != 0)
        )

        if not np.any(valid):
            continue

        # Linear intensity -> dB
        db = np.zeros_like(channel, dtype=np.float32)

        db[valid] = (
            10.0
            * np.log10(channel[valid] + SAR_EPS)
        )

        # Percentile clipping
        db_valid = db[valid]

        db_valid = np.clip(
            db_valid,
            SAR_P2[c],
            SAR_P98[c]
        )

        # Z-score normalization
        normalized = (
            db_valid - SAR_MEAN[c]
        ) / max(SAR_STD[c], 1e-8)

        output[c][valid] = normalized.astype(
            np.float32
        )

    return output


# ============================================================
# OPTICAL PREPROCESSING
# ============================================================

def preprocess_optical(optical):
    """
    Apply the exact optical preprocessing used during training.

    uint8
        ↓
    [0,1]
        ↓
    P2-P98 clipping
        ↓
    training-set z-score normalization

    Input:
        Optical array [4,H,W]

    Output:
        float32 array [4,H,W]
    """

    optical = np.asarray(
        optical,
        dtype=np.float32
    )

    if optical.ndim != 3:
        raise ValueError(
            f"Optical must have shape [C,H,W], got {optical.shape}"
        )

    if optical.shape[0] != 4:
        raise ValueError(
            f"Optical must have 4 channels, got {optical.shape[0]}"
        )

    # uint8 -> [0,1]
    optical = optical / 255.0

    output = np.zeros_like(
        optical,
        dtype=np.float32
    )

    for c in range(4):

        channel = optical[c]

        # Clip using training statistics
        clipped = np.clip(
            channel,
            OPT_P2[c],
            OPT_P98[c]
        )

        # Z-score normalization
        output[c] = (
            clipped - OPT_MEAN[c]
        ) / max(OPT_STD[c], 1e-8)

    return output.astype(np.float32)


# ============================================================
# READ GEOTIFF
# ============================================================

def read_geotiff(path):
    """
    Read a GeoTIFF into a NumPy array.

    Returns:
        array: [C,H,W]
        profile: rasterio metadata
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with rasterio.open(path) as src:

        array = src.read()

        profile = src.profile.copy()

    return array, profile


# ============================================================
# LOAD + PREPROCESS SAR
# ============================================================

def load_sar(path):
    """
    Read and preprocess a SAR GeoTIFF.
    """

    sar, profile = read_geotiff(path)

    sar = preprocess_sar(sar)

    return sar, profile


# ============================================================
# LOAD + PREPROCESS OPTICAL
# ============================================================

def load_optical(path):
    """
    Read and preprocess an optical GeoTIFF.
    """

    optical, profile = read_geotiff(path)

    optical = preprocess_optical(optical)

    return optical, profile


# ============================================================
# OPTIONAL MASK PREPROCESSING
# ============================================================

def preprocess_mask(mask):
    """
    Convert SpaceNet mask:

        0   -> background
        255 -> building

    Output:
        binary float32 mask [H,W]
    """

    mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = mask[0]

    return (
        mask > 0
    ).astype(np.float32)


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SpaceNet 6 preprocessing module")
    print("=" * 60)

    print("\nSAR statistics:")
    print("Mean:", SAR_MEAN)
    print("Std :", SAR_STD)
    print("P2  :", SAR_P2)
    print("P98 :", SAR_P98)

    print("\nOptical statistics:")
    print("Mean:", OPT_MEAN)
    print("Std :", OPT_STD)
    print("P2  :", OPT_P2)
    print("P98 :", OPT_P98)

    print("\n✓ Preprocessing module loaded successfully")