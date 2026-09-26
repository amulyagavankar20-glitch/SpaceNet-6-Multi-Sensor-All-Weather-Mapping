import base64
from io import BytesIO

import numpy as np
import rasterio
import requests
import streamlit as st
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

API_URL = "http://127.0.0.1:8000"

MODEL_INFO = {
    "E1": {
        "name": "E1 - Plain U-Net",
        "modality": "SAR",
        "description": "Baseline SAR-only U-Net",
    },
    "E2": {
        "name": "E2 - ResNet34 U-Net",
        "modality": "SAR",
        "description": "ResNet34 encoder + U-Net decoder",
    },
    "E3": {
        "name": "E3 - Optical Teacher",
        "modality": "Optical",
        "description": "Optical-only teacher model",
    },
    "E4": {
        "name": "E4 - SAR + Knowledge Distillation",
        "modality": "SAR",
        "description": "SAR-only student trained using optical knowledge",
    },
}


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SpaceNet 6 Building Extraction",
    page_icon="🏢",
    layout="wide",
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def decode_image(base64_string):
    """Decode Base64 PNG returned by FastAPI."""
    image_bytes = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_bytes))


def normalize_channel(channel):
    """Normalize a channel to uint8 for visualization."""
    channel = channel.astype(np.float32)

    valid = np.isfinite(channel)

    if not np.any(valid):
        return np.zeros(channel.shape, dtype=np.uint8)

    values = channel[valid]

    p2 = np.percentile(values, 2)
    p98 = np.percentile(values, 98)

    if p98 <= p2:
        return np.zeros(channel.shape, dtype=np.uint8)

    channel = np.clip(channel, p2, p98)
    channel = (channel - p2) / (p98 - p2)

    return (channel * 255).astype(np.uint8)


def create_input_preview(file_bytes, modality):
    """
    Create a displayable 512x512 preview from the uploaded GeoTIFF.
    This is only for visualization; actual preprocessing is done by API.
    """

    with rasterio.MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:

            data = src.read()

    # Center crop to match model inference
    _, height, width = data.shape

    crop_size = 512

    top = max((height - crop_size) // 2, 0)
    left = max((width - crop_size) // 2, 0)

    data = data[
        :,
        top:top + crop_size,
        left:left + crop_size
    ]

    if modality == "Optical":

        # Optical has 4 channels.
        # Display first 3 channels as RGB.
        if data.shape[0] >= 3:
            rgb = np.stack(
                [
                    normalize_channel(data[0]),
                    normalize_channel(data[1]),
                    normalize_channel(data[2]),
                ],
                axis=-1,
            )
        else:
            rgb = normalize_channel(data[0])

        return Image.fromarray(rgb)

    else:

        # SAR is not RGB.
        # Display first SAR channel as grayscale.
        sar = normalize_channel(data[0])

        return Image.fromarray(sar)


def create_mask_preview(file_bytes):

    with rasterio.MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            mask = src.read(1)

    height, width = mask.shape

    crop_size = 512

    top = max((height - crop_size) // 2, 0)
    left = max((width - crop_size) // 2, 0)

    mask = mask[
        top:top + crop_size,
        left:left + crop_size
    ]

    mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    return Image.fromarray(mask)


def run_prediction(
    model,
    sar_bytes=None,
    sar_name=None,
    optical_bytes=None,
    optical_name=None,
    gt_bytes=None,
    gt_name=None,
):

    files = {}

    if sar_bytes is not None:
        files["sar_image"] = (
            sar_name,
            sar_bytes,
            "image/tiff",
        )

    if optical_bytes is not None:
        files["optical_image"] = (
            optical_name,
            optical_bytes,
            "image/tiff",
        )

    if gt_bytes is not None:
        files["ground_truth"] = (
            gt_name,
            gt_bytes,
            "image/tiff",
        )

    response = requests.post(
        f"{API_URL}/predict",
        data={"model": model},
        files=files,
        timeout=300,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# HEADER
# ============================================================

st.title("🏢 SpaceNet 6 Building Extraction")

st.markdown(
    """
    **Multi-sensor building footprint extraction**

    Compare SAR-only models, the optical teacher, and the
    SAR student trained with optical knowledge distillation.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Model Selection")

    model = st.selectbox(
        "Select Model",
        options=["E1", "E2", "E3", "E4"],
        format_func=lambda x: MODEL_INFO[x]["name"],
    )

    selected_model = MODEL_INFO[model]

    st.markdown("---")

    st.write("**Modality:**", selected_model["modality"])

    st.write("**Description:**")
    st.caption(selected_model["description"])

    st.markdown("---")

    st.info(
        "E4 uses SAR only during inference. "
        "Optical imagery is used only during training "
        "through knowledge distillation."
    )


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("Input Data")

if selected_model["modality"] == "SAR":

    sar_file = st.file_uploader(
        "Upload SAR GeoTIFF",
        type=["tif", "tiff"],
        key="sar_upload",
    )

    optical_file = None

else:

    optical_file = st.file_uploader(
        "Upload Optical GeoTIFF",
        type=["tif", "tiff"],
        key="optical_upload",
    )

    sar_file = None


ground_truth_file = st.file_uploader(
    "Ground Truth Mask (Optional)",
    type=["tif", "tiff"],
    key="gt_upload",
)


# ============================================================
# PREVIEW
# ============================================================

if sar_file is not None or optical_file is not None:

    st.subheader("Input Preview")

    input_file = sar_file if sar_file is not None else optical_file

    input_bytes = input_file.getvalue()

    preview = create_input_preview(
        input_bytes,
        selected_model["modality"],
    )

    st.image(
        preview,
        caption=f"{selected_model['modality']} Input — Center 512×512",
        width=512,
    )


if ground_truth_file is not None:

    st.subheader("Ground Truth Preview")

    gt_bytes_preview = ground_truth_file.getvalue()

    gt_preview = create_mask_preview(gt_bytes_preview)

    st.image(
        gt_preview,
        caption="Ground Truth Mask — Center 512×512",
        width=512,
    )


# ============================================================
# RUN BUTTON
# ============================================================

st.divider()

run_button = st.button(
    "🚀 RUN PREDICTION",
    type="primary",
    use_container_width=True,
)


# ============================================================
# PREDICTION
# ============================================================

if run_button:

    if selected_model["modality"] == "SAR" and sar_file is None:

        st.error("Please upload a SAR GeoTIFF.")

    elif selected_model["modality"] == "Optical" and optical_file is None:

        st.error("Please upload an Optical GeoTIFF.")

    else:

        with st.spinner("Running model inference..."):

            try:

                sar_bytes = (
                    sar_file.getvalue()
                    if sar_file is not None
                    else None
                )

                optical_bytes = (
                    optical_file.getvalue()
                    if optical_file is not None
                    else None
                )

                gt_bytes = (
                    ground_truth_file.getvalue()
                    if ground_truth_file is not None
                    else None
                )

                result = run_prediction(
                    model=model,

                    sar_bytes=sar_bytes,
                    sar_name=(
                        sar_file.name
                        if sar_file is not None
                        else None
                    ),

                    optical_bytes=optical_bytes,
                    optical_name=(
                        optical_file.name
                        if optical_file is not None
                        else None
                    ),

                    gt_bytes=gt_bytes,
                    gt_name=(
                        ground_truth_file.name
                        if ground_truth_file is not None
                        else None
                    ),
                )

                st.session_state["prediction_result"] = result

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to FastAPI. "
                    "Make sure the API is running on "
                    "http://127.0.0.1:8000"
                )

            except Exception as e:

                st.error(f"Prediction failed: {e}")


# ============================================================
# DISPLAY RESULT
# ============================================================

if "prediction_result" in st.session_state:

    result = st.session_state["prediction_result"]

    if result.get("success"):

        st.divider()

        st.subheader(
            f"Prediction — {result['model_name']}"
        )

        # ----------------------------------------------------
        # Images
        # ----------------------------------------------------

        prediction_image = decode_image(
            result["images"]["prediction_png"]
        )

        probability_image = decode_image(
            result["images"]["probability_png"]
        )

        overlay_image = decode_image(
            result["images"]["overlay_png"]
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.image(
                prediction_image,
                caption="Predicted Building Mask",
                width="stretch",
            )

        with col2:

            st.image(
                overlay_image,
                caption="Prediction Overlay",
                width="stretch",
            )

        with col3:

            st.image(
                probability_image,
                caption="Prediction Probability",
                width="stretch",
            )

        # ----------------------------------------------------
        # Prediction Statistics
        # ----------------------------------------------------

        st.divider()

        st.subheader("Prediction Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Output Size",
                f"{result['prediction']['height']} × "
                f"{result['prediction']['width']}",
            )

        with col2:

            st.metric(
                "Building Pixels",
                f"{result['prediction']['positive_pixels']:,}",
            )

        with col3:

            st.metric(
                "Building Area",
                f"{result['prediction']['building_percentage']:.2f}%",
            )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = result.get("metrics")

        if metrics is not None:

            st.divider()

            st.subheader("Ground Truth Metrics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Dice",
                    f"{metrics['dice']:.4f}",
                )

            with col2:
                st.metric(
                    "IoU",
                    f"{metrics['iou']:.4f}",
                )

            with col3:
                st.metric(
                    "Precision",
                    f"{metrics['precision']:.4f}",
                )

            with col4:
                st.metric(
                    "Recall",
                    f"{metrics['recall']:.4f}",
                )

        else:

            st.info(
                "Upload a ground-truth mask to calculate "
                "Dice, IoU, Precision and Recall."
            )

    else:

        st.error("Prediction failed.")