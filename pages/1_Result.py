import streamlit as st
import os


st.set_page_config(
    page_title="DeepTrace AI Result",
    page_icon="🎥",
    layout="wide"
)


st.title("🎥 DeepTrace AI")

st.header(
    "🔍 Detection Result"
)


if "result" not in st.session_state:

    st.warning(
        "No analysis available. Please analyze a video first."
    )

    st.stop()



result = st.session_state.result



left,right = st.columns(
    2,
    gap="large"
)



with left:


    st.subheader(
        " Classification"
    )


    if result["prediction"]=="Deepfake":

        st.error(
            f"Prediction: {result['prediction']}"
        )

    else:

        st.success(
            f"Prediction: {result['prediction']}"
        )


    st.subheader(
        "📊 Confidence"
    )


    st.progress(
        result["confidence"]/100
    )


    st.write(
        f"{result['confidence']:.2f}%"
    )



with right:


    st.subheader(
        "🎥 Video Evidence"
    )


    if result["video"]:

        st.video(
            result["video"]
        )


    st.write(
        "Model: ResNet18 Transfer Learning"
    )


    st.write(
        "Frames analyzed: 60"
    )



st.divider()


st.header(
    "⚙ AI Processing Pipeline"
)


c1,c2,c3 = st.columns(3)



with c1:

    st.info(
        """
        🎞 Frame Extraction
        
        Video frames sampled
        """
    )


with c2:

    st.info(
        """
        🙂 Face Detection
        
        OpenCV face detector
        """
    )


with c3:

    st.info(
        """
        🧠 Classification
        
        ResNet18 Transfer Learning
        """
    )


if st.button(
    "⬅ Analyze Another Video"
):

    st.switch_page(
        "streamlit_app.py"
    )