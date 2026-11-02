import os
import tempfile
import streamlit as st

from src.pipeline import process_scene, process_cropped_sign
from src.sign_config import ID_TO_SIGN


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Traffic Sign Audio Guidance",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #666;
    margin-bottom: 25px;
}

.result-card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    margin-top: 15px;
}

.instruction {
    font-size: 22px;
    font-weight: 600;
    padding: 15px;
    border-radius: 10px;
}

.metric-box {
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #ddd;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚦 Traffic Sign Audio Guidance</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Traffic Sign Detection → Classification → NLP Instruction → Audio Guidance'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System Configuration")

    mode = st.radio(
        "Input Type",
        [
            "Full Driving Scene",
            "Cropped Traffic Sign"
        ]
    )

    st.divider()

    st.subheader("🤖 Model")

    st.write("**Classifier:** Hybrid Random Forest")
    st.write("**Features:** HOG + HSV")
    st.write("**Training:** Synthetic + GTSRB")

    st.divider()

    st.subheader("🔊 Audio")

    st.write("**TTS:** eSpeak NG")
    st.write("**Output:** WAV")

    st.divider()

    st.subheader("📊 Validation")

    st.metric(
        "Real Validation Accuracy",
        "98.42%"
    )

    st.divider()

    with st.expander("Supported Signs"):

        for class_id, sign in ID_TO_SIGN.items():
            st.write(
                f"{class_id}. {sign.display_name}"
            )


# ============================================================
# MODEL CHECK
# ============================================================

MODEL_PATH = "models/sign_classifier_hybrid.joblib"

if not os.path.exists(MODEL_PATH):

    st.error(
        "❌ Trained model not found.\n\n"
        "Please run:\n"
        "`python -m src.train`"
    )

    st.stop()


# ============================================================
# FILE UPLOAD
# ============================================================

st.header("📷 Upload Image")

uploaded_file = st.file_uploader(
    "Upload a traffic sign image or driving-scene image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    # --------------------------------------------------------
    # IMAGE PREVIEW
    # --------------------------------------------------------

    col1, col2 = st.columns([1, 1])

    with col1:

        st.subheader("Input Image")

        st.image(
            image_bytes,
            use_container_width=True
        )

    with col2:

        st.subheader("Analysis")

        if mode == "Full Driving Scene":

            st.info(
                "The system will first detect candidate sign regions "
                "and then classify the highest-confidence candidate."
            )

        else:

            st.info(
                "The uploaded image is treated as an already-cropped "
                "traffic sign."
            )


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    if st.button(
        "🔍 Analyze Traffic Sign",
        type="primary",
        use_container_width=True
    ):

        os.makedirs("outputs", exist_ok=True)

        temp_path = None

        try:

            # ------------------------------------------------
            # SAVE UPLOADED IMAGE TEMPORARILY
            # ------------------------------------------------

            extension = os.path.splitext(
                uploaded_file.name
            )[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=extension
            ) as temp_file:

                temp_file.write(image_bytes)
                temp_path = temp_file.name


            # ------------------------------------------------
            # RUN PIPELINE
            # ------------------------------------------------

            with st.spinner(
                "Detecting sign → Classifying → Generating instruction → Creating audio..."
            ):

                if mode == "Full Driving Scene":

                    results = process_scene(temp_path)

                else:

                    results = process_cropped_sign(temp_path)


            # =================================================
            # RESULTS
            # =================================================

            st.divider()

            st.header("🎯 Detection Result")


            if not results:

                st.warning(
                    "⚠️ No traffic sign was recognized in this image."
                )

            else:

                result = results[0]

                sign_name = result["sign_name"]
                confidence = result["confidence"]
                instruction = result["instruction"]
                audio_path = result["audio_path"]
                class_id = result["class_id"]

                sign = ID_TO_SIGN[class_id]

                # ---------------------------------------------
                # RESULT METRICS
                # ---------------------------------------------

                m1, m2, m3 = st.columns(3)

                with m1:

                    st.metric(
                        "Detected Sign",
                        sign_name
                    )

                with m2:

                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.1f}%"
                    )

                with m3:

                    st.metric(
                        "Severity",
                        str(
                            getattr(
                                sign,
                                "severity",
                                "N/A"
                            )
                        ).title()
                    )


                # ---------------------------------------------
                # CONFIDENCE
                # ---------------------------------------------

                st.subheader("Classification Confidence")

                st.progress(
                    min(max(confidence, 0.0), 1.0)
                )


                # ---------------------------------------------
                # NLP INSTRUCTION
                # ---------------------------------------------

                st.subheader("🧠 Generated Safety Instruction")

                st.markdown(
                    f"""
                    <div class="instruction">
                    {instruction}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # ---------------------------------------------
                # AUDIO
                # ---------------------------------------------

                st.subheader("🔊 Audio Guidance")

                if os.path.exists(audio_path):

                    st.audio(
                        audio_path,
                        format="audio/wav"
                    )

                    st.success(
                        "Audio guidance generated successfully."
                    )

                else:

                    st.warning(
                        "Audio file could not be found."
                    )


                # ---------------------------------------------
                # TECHNICAL DETAILS
                # ---------------------------------------------

                with st.expander("🔬 Technical Details"):

                    st.write(
                        f"**Class ID:** {class_id}"
                    )

                    st.write(
                        f"**Sign:** {sign_name}"
                    )

                    st.write(
                        f"**Confidence:** {confidence * 100:.2f}%"
                    )

                    st.write(
                        f"**Instruction:** {instruction}"
                    )

                    st.write(
                        f"**Audio:** {audio_path}"
                    )


        except Exception as e:

            st.error(
                f"❌ An error occurred:\n\n{e}"
            )

        finally:

            # ------------------------------------------------
            # DELETE TEMPORARY INPUT
            # ------------------------------------------------

            if temp_path and os.path.exists(temp_path):

                try:
                    os.remove(temp_path)
                except:
                    pass


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Traffic Sign Audio Guidance System | "
    "HOG + HSV + Random Forest + NLP + eSpeak NG"
)