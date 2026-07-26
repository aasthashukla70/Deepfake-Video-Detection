# 🎥 DeepTrace AI - Deepfake Video Detection System

## 🚀 Live Demo

Streamlit Application:

https://deepfake-video-detection-ml.streamlit.app/

---

## 📌 Overview

DeepTrace AI is a deep learning-based deepfake video detection system that analyzes facial manipulation patterns in videos and classifies them as **Original** or **Deepfake**.

The system uses computer vision techniques for video processing, face extraction, and a ResNet18-based deep learning classifier to detect manipulated facial regions.

The project demonstrates a complete ML pipeline:

**Video Input → Frame Extraction → Face Detection → Deep Learning Classification → Video-Level Prediction**

---

# ✨ Features

- Upload video files for deepfake analysis
- Detect faces from video frames using OpenCV
- Extract facial regions for classification
- Deepfake classification using ResNet18 Transfer Learning
- Confidence score generation
- Demo videos for testing
- Interactive Streamlit deployment

---

# 🏗️ System Architecture

```
Input Video
      |
      ↓
Frame Extraction
      |
      ↓
Face Detection
(OpenCV DNN Detector)
      |
      ↓
Face Preprocessing
      |
      ↓
ResNet18 Deep Learning Model
      |
      ↓
Binary Classification
(Original / Deepfake)
      |
      ↓
Confidence Score
```

---

# 🧠 Model Details

## Deep Learning Model

**Architecture:**

- ResNet18 Transfer Learning
- Binary Image Classification

The pretrained ResNet18 model was fine-tuned to classify extracted facial images into:

- Original
- Deepfake


## Video-Level Prediction

Since deepfake detection is performed on frames:

1. Multiple frames are sampled from the video
2. Faces are detected from each frame
3. Each face is classified independently
4. Predictions are aggregated to generate the final video-level result

---

# 📂 Dataset

The project uses the:

## FaceForensics++ Dataset

Dataset characteristics:

- Real and manipulated face videos
- C23 compression quality
- Deepfake generated samples

For development and experimentation, a controlled subset was used:

- 25 Original videos
- 25 Deepfake videos

Frames were extracted from videos and processed into facial image samples for model training.

---

# ⚙️ Tech Stack

## Programming Language

- Python

## Deep Learning

- PyTorch
- Torchvision

## Computer Vision

- OpenCV
- OpenCV DNN Face Detector

## Machine Learning

- Scikit-learn
- NumPy

## Image Processing

- Pillow

## Deployment

- Streamlit Cloud

---

# 📁 Project Structure

```
Deepfake-Video-Detection/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── resnet18_final.pth
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
│
├── src/
│   ├── frame_extractor.py
│   ├── face_extractor.py
│   └── video_predict_final.py
│
├── pages/
│   └── 1_Result.py
│
├── static/
│   └── demo_videos/
│
├── streamlit_app.py
├── requirements.txt
└── README.md
```


---

# 📊 Model Evaluation

Evaluation was performed on the validation dataset.

| Metric | Score |
|---|---:|
| Accuracy | 55.83% |
| Precision | 59.56% |
| Recall | 36.33% |
| F1 Score | 45.13% |

The model demonstrates the complete deepfake detection workflow and provides a foundation for further improvement using larger datasets and temporal learning approaches.

---

# 💻 Running Locally

## Clone Repository

```bash
git clone https://github.com/aasthashukla70/Deepfake-Video-Detection.git

Navigate into the project:

cd Deepfake-Video-Detection
Create Virtual Environment
python -m venv venv

Activate environment:

Windows:

venv\Scripts\activate
Install Dependencies
pip install -r requirements.txt
Run Application
streamlit run streamlit_app.py
🔮 Future Improvements
Train on a larger FaceForensics++ subset
Improve temporal modeling using CNN-LSTM/Transformer architectures
Add real-time webcam deepfake detection
Improve model generalization across different manipulation techniques
Add explainable AI visualization for detected regions
👩‍💻 Author

Aastha Shukla

B.Tech Computer Science Engineering

⭐ If you find this project useful, consider giving it a star!


After replacing:

```powershell
git add README.md
git commit -m "Updated project documentation"
git push