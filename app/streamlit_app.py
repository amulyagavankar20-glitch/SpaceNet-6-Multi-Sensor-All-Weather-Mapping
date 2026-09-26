import base64
import io
import requests
import streamlit as st
from PIL import Image


API_URL = "http://127.0.0.1:8000"

MODEL_INFO = {
    "E1": {
        "name": "E1 - SAR U-Net",
        "input": "SAR",
    },
    "E2": {
        "name": "E2 - SAR ResNet34-U-Net",
        "input": "SAR",
    },
    "E3": {
        "name": "E3 - Optical Teacher",
        "input": "Optical",
    },
    "E4": {
        "name": "E4 - SAR + Knowledge Distillation",
        "input": "SAR",
    },
}


st.set_page_config(
    page_title="SpaceNet 6 Building Extraction",
    page_icon="🏢",
    layout="wide",
)


# ============================================================
# HELPERS
# ============================================================

def decode_base64_image(encoded):
    return Image.open(
        io.BytesIO(base64.b64decode(encoded))
    )


def call_prediction(model, sar_file=None, optical_file=None, gt_file=None):
    files = {}

    if sar_file is not None:
        sar_file.seek(0)
        files["sar_image"] = (
            sar_file.name,
            sar_file,
            "image/tiff",
        )

    if optical_file is not None:
        optical_file.seek(0)
        files["optical_image"] = (
            optical_file.name,
            optical_file,
            "image/tiff",
        )

    if gt_file is not None:
        gt_file.seek(0)
        files["ground_truth"] = (
            gt_file.name,
            gt_file,
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


def display_result(result, model):
    st.markdown(f"### {MODEL_INFO[model]['name']}")

    if not result.get("success", False):
        st.error("Prediction failed.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            decode_base64_image(result["images"]["prediction_png"]),
            caption="Predicted Building Mask",
            use_container_width=True,
        )

    with col2:
        st.image(
            decode_base64_image(result["images"]["overlay_png"]),
            caption="Prediction Overlay",
            use_container_width=True,
        )

    with col3:
        st.image(
            decode_base64_image(result["images"]["probability_png"]),
            caption="Prediction Probability",
            use_container_width=True,
        )

    stats = result.get("metrics")

    if stats:
        cols = st.columns(4)

        cols[0].metric(
            "Dice",
            f"{stats.get('dice', 0):.4f}",
        )

        cols[1].metric(
            "IoU",
            f"{stats.get('iou', 0):.4f}",
        )

        cols[2].metric(
            "Precision",
            f"{stats.get('precision', 0):.4f}",
        )

        cols[3].metric(
            "Recall",
            f"{stats.get('recall', 0):.4f}",
        )


# ============================================================
# HEADER
# ============================================================

st.title("🏢 SpaceNet 6 Building Extraction")

st.caption(
    "SAR-only building extraction with optical-guided knowledge distillation"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Mode")

mode = st.sidebar.radio(
    "Select mode",
    [
        "Single Model Prediction",
        "Research Comparison",
    ],
)


# ============================================================
# SINGLE MODEL MODE
# ============================================================

if mode == "Single Model Prediction":

    st.sidebar.header("Model")

    model = st.sidebar.selectbox(
        "Select model",
        list(MODEL_INFO.keys()),
        format_func=lambda x: MODEL_INFO[x]["name"],
    )

    required_input = MODEL_INFO[model]["input"]

    st.sidebar.markdown(
        f"**Required input:** `{required_input}`"
    )

    st.header(MODEL_INFO[model]["name"])

    if required_input == "SAR":

        sar_file = st.file_uploader(
            "Upload SAR GeoTIFF",
            type=["tif", "tiff"],
            key="single_sar",
        )

        optical_file = None

    else:

        optical_file = st.file_uploader(
            "Upload Optical GeoTIFF",
            type=["tif", "tiff"],
            key="single_optical",
        )

        sar_file = None

    gt_file = st.file_uploader(
        "Ground Truth Mask (optional)",
        type=["tif", "tiff"],
        key="single_gt",
    )

    if sar_file:

        st.subheader("Input SAR")

        st.info(
            "SAR input detected. The API uses the same "
            "preprocessing and 512×512 center-crop protocol "
            "used for the official test evaluation."
        )

    if optical_file:

        st.subheader("Input Optical")

        st.info(
            "Optical input detected. The API uses the same "
            "preprocessing and 512×512 center-crop protocol "
            "used for the official test evaluation."
        )

    run_prediction = st.button(
        "🚀 Run Prediction",
        type="primary",
        use_container_width=True,
    )

    if run_prediction:

        if required_input == "SAR" and sar_file is None:
            st.error("Please upload a SAR GeoTIFF.")

        elif required_input == "Optical" and optical_file is None:
            st.error("Please upload an Optical GeoTIFF.")

        else:

            with st.spinner("Running model..."):

                try:

                    result = call_prediction(
                        model=model,
                        sar_file=sar_file,
                        optical_file=optical_file,
                        gt_file=gt_file,
                    )

                    st.success("Prediction completed.")

                    display_result(result, model)

                except Exception as e:

                    st.error(
                        f"Prediction failed: {str(e)}"
                    )


# ============================================================
# RESEARCH COMPARISON MODE
# ============================================================

else:

    st.header("🔬 Research Comparison")

    st.markdown(
        """
        Compare **E1, E2, E3 and E4 on the same test tile**.

        - **E1:** SAR U-Net
        - **E2:** SAR ResNet34-U-Net
        - **E3:** Optical Teacher
        - **E4:** SAR + Optical Knowledge Distillation
        """
    )

    st.divider()

    st.subheader("1. Upload the Same Test Tile")

    st.info(
        "Upload the SAR, Optical and Ground Truth files "
        "belonging to the same tile."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        sar_file = st.file_uploader(
            "SAR GeoTIFF",
            type=["tif", "tiff"],
            key="comparison_sar",
        )

    with col2:

        optical_file = st.file_uploader(
            "Optical GeoTIFF",
            type=["tif", "tiff"],
            key="comparison_optical",
        )

    with col3:

        gt_file = st.file_uploader(
            "Ground Truth Mask",
            type=["tif", "tiff"],
            key="comparison_gt",
        )

    if sar_file and optical_file and gt_file:

        st.success("All three inputs uploaded.")

        st.divider()

        st.subheader("2. Run All Models")

        run_comparison = st.button(
            "🚀 Run E1 + E2 + E3 + E4",
            type="primary",
            use_container_width=True,
        )

        if run_comparison:

            results = {}

            progress = st.progress(0)

            models = ["E1", "E2", "E3", "E4"]

            for i, model in enumerate(models):

                with st.spinner(
                    f"Running {MODEL_INFO[model]['name']}..."
                ):

                    try:

                        result = call_prediction(
                            model=model,
                            sar_file=sar_file,
                            optical_file=optical_file,
                            gt_file=gt_file,
                        )

                        results[model] = result

                    except Exception as e:

                        st.error(
                            f"{model} failed: {str(e)}"
                        )

                progress.progress(
                    (i + 1) / len(models)
                )

            if len(results) == 4:

                st.success(
                    "All four models completed successfully."
                )

                st.session_state["comparison_results"] = results


    # ========================================================
    # DISPLAY COMPARISON
    # ========================================================

    if "comparison_results" in st.session_state:

        results = st.session_state["comparison_results"]

        st.divider()

        st.subheader("3. Same-Tile Prediction Comparison")

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        if "E2" in results:

            st.markdown("### Ground Truth")

            gt_prediction = results["E2"]

            st.image(
                decode_base64_image(
                    gt_prediction["images"]["overlay_png"]
                ),
                caption="E2 Prediction Overlay",
                width=512,
            )

        # ----------------------------------------------------
        # E1 / E2 / E4
        # ----------------------------------------------------

        st.markdown("### SAR Models")

        sar_models = ["E1", "E2", "E4"]

        cols = st.columns(3)

        for col, model in zip(cols, sar_models):

            with col:

                result = results[model]

                st.markdown(
                    f"#### {MODEL_INFO[model]['name']}"
                )

                st.image(
                    decode_base64_image(
                        result["images"]["prediction_png"]
                    ),
                    caption="Building Mask",
                    use_container_width=True,
                )

                metrics = result.get("metrics")

                if metrics:

                    st.metric(
                        "Dice",
                        f"{metrics.get('dice', 0):.4f}",
                    )

                    st.metric(
                        "IoU",
                        f"{metrics.get('iou', 0):.4f}",
                    )

        # ----------------------------------------------------
        # E3
        # ----------------------------------------------------

        st.markdown("### Optical Teacher")

        e3 = results["E3"]

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                decode_base64_image(
                    e3["images"]["prediction_png"]
                ),
                caption="E3 Building Mask",
                use_container_width=True,
            )

        with col2:

            metrics = e3.get("metrics")

            if metrics:

                st.metric(
                    "Dice",
                    f"{metrics.get('dice', 0):.4f}",
                )

                st.metric(
                    "IoU",
                    f"{metrics.get('iou', 0):.4f}",
                )

                st.metric(
                    "Precision",
                    f"{metrics.get('precision', 0):.4f}",
                )

                st.metric(
                    "Recall",
                    f"{metrics.get('recall', 0):.4f}",
                )

        # ----------------------------------------------------
        # Summary table
        # ----------------------------------------------------

        st.divider()

        st.subheader("4. Quantitative Comparison")

        table_data = []

        for model in ["E1", "E2", "E3", "E4"]:

            metrics = results[model].get("metrics")

            if metrics:

                table_data.append(
                    {
                        "Model": model,
                        "Dice": metrics.get("dice", 0),
                        "IoU": metrics.get("iou", 0),
                        "Precision": metrics.get(
                            "precision", 0
                        ),
                        "Recall": metrics.get(
                            "recall", 0
                        ),
                    }
                )

        if table_data:

            st.dataframe(
                table_data,
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # Research interpretation
        # ----------------------------------------------------

        st.divider()

        st.subheader("5. Experiment Progression")

        st.markdown(
            """
            **E1 → E2**

            Architectural improvement from a plain U-Net
            to a ResNet34 encoder-decoder.

            **E2 → E3**

            Demonstrates the upper-bound benefit of having
            optical imagery available directly during inference.

            **E2 → E4**

            Tests whether optical information transferred
            during training through knowledge distillation
            can improve SAR-only inference.
            """
        )