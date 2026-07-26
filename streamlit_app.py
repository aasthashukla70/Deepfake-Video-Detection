import streamlit as st
import tempfile
import os

from src.video_predict_final import predict_video


# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="DeepTrace AI",
    page_icon="🎥",
    layout="wide"
)


# ==========================
# CUSTOM CSS
# ==========================

st.markdown(
    """
    <style>

    .block-container {
        max-width:1400px;
        padding-left:4rem;
        padding-right:4rem;
    }


    h1 {
        text-align:center;
        font-size:55px;
    }


    h2 {
        text-align:center;
    }


    .stButton button {

        width:100%;
        border-radius:10px;
        height:45px;

    }


    </style>

    """,

    unsafe_allow_html=True
)



# ==========================
# HEADER
# ==========================

st.title("🎥 DeepTrace AI")


st.subheader(
    "AI-powered Deepfake Detection System"
)


st.write(
    "Analyze suspicious videos using face detection and deep learning."
)


st.divider()



# ==========================
# MAIN COLUMNS
# ==========================

upload_col, demo_col = st.columns(
    2,
    gap="large"
)



# ==========================
# UPLOAD SECTION
# ==========================

with upload_col:


    st.header(
        "📁 Upload Video Evidence"
    )


    st.write(
        "Upload a suspicious video and let our AI pipeline analyze facial manipulation patterns."
    )


    st.info(
        """
Supported formats:

MP4 | AVI | MOV


AI Pipeline:

✓ Frame Extraction

✓ Face Detection using OpenCV

✓ ResNet18 Classification
        """
    )


    uploaded_video = st.file_uploader(
        "Choose video",
        type=[
            "mp4",
            "avi",
            "mov"
        ]
    )


    if uploaded_video:


        st.video(
            uploaded_video
        )


        if st.button(
            "Analyze Video"
        ):


            with st.spinner(
                "Analyzing video..."
            ):


                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as temp:


                    temp.write(
                        uploaded_video.read()
                    )


                    video_path = temp.name



                prediction, confidence = predict_video(
                    video_path
                )


                os.remove(
                    video_path
                )



            st.session_state.result = {

                "prediction": prediction,
                "confidence": confidence,
                "video": uploaded_video

            }


            st.switch_page(
                "pages/1_Result.py"
            )





# ==========================
# DEMO VIDEOS
# ==========================

with demo_col:


    st.header(
        "🎬 Try Demo Videos"
    )


    st.write(
        "Test the system using sample original and deepfake videos."
    )


    original_col, deepfake_col = st.columns(
        2
    )


    demo_path = "static/demo_videos"



    # -------------------------
    # ORIGINAL VIDEOS
    # -------------------------

    with original_col:


        st.subheader(
            "Original Videos"
        )


        for i in range(1,6):


            if st.button(
                f"Original_{i}",
                key=f"original_{i}"
            ):


                path = os.path.join(
                    demo_path,
                    f"original_{i}.mp4"
                )


                if os.path.exists(path):


                    prediction, confidence = predict_video(
                        path
                    )


                    st.session_state.result = {

                        "prediction": prediction,
                        "confidence": confidence,
                        "video": path

                    }


                    st.switch_page(
                        "pages/1_Result.py"
                    )


                else:

                    st.error(
                        f"Missing file: {path}"
                    )





    # -------------------------
    # DEEPFAKE VIDEOS
    # -------------------------

    with deepfake_col:


        st.subheader(
            "Deepfake Videos"
        )


        for i in range(1,6):


            if st.button(
                f"Deepfake_{i}",
                key=f"deepfake_{i}"
            ):


                path = os.path.join(
                    demo_path,
                    f"deepfake_{i}.mp4"
                )


                if os.path.exists(path):


                    prediction, confidence = predict_video(
                        path
                    )


                    st.session_state.result = {

                        "prediction": prediction,
                        "confidence": confidence,
                        "video": path

                    }


                    st.switch_page(
                        "pages/1_Result.py"
                    )


                else:

                    st.error(
                        f"Missing file: {path}"
                    )