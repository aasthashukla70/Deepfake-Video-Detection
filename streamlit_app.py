import streamlit as st
import tempfile
import os

from src.video_predict_final import predict_video


st.set_page_config(
    page_title="DeepTrace AI",
    page_icon="🎥",
    layout="centered"
)


st.title("🎥 DeepTrace AI")
st.subheader("AI-powered Deepfake Detection System")

st.write(
    "Upload a video and let the deep learning model analyze "
    "facial manipulation patterns."
)


uploaded_video = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov"]
)


if uploaded_video:


    st.video(uploaded_video)


    if st.button("Analyze Video"):


        with st.spinner("Analyzing video..."):


            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as temp_file:

                temp_file.write(
                    uploaded_video.read()
                )

                video_path = temp_file.name



            prediction, confidence = predict_video(
                video_path
            )


            os.remove(video_path)



        st.success("Analysis Completed")


        if prediction == "Deepfake":

            st.error(
                f"Prediction: {prediction}"
            )

        else:

            st.success(
                f"Prediction: {prediction}"
            )


        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


st.markdown("---")

st.subheader("⚙ AI Pipeline")

st.write(
    """
    1. Frame Extraction  
    2. Face Detection using OpenCV  
    3. ResNet18 Transfer Learning Classification  
    4. Video-level Prediction
    """
)