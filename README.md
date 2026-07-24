\# Deepfake Video Detection



\## Overview



A deep learning based system that detects whether a video is Original or Deepfake.



The pipeline extracts video frames, detects faces, and classifies facial regions using a ResNet18 based classifier.



\## Dataset



Dataset used:

FaceForensics++



Category:

Deepfakes



Compression:

C23



\## Pipeline



Video Input

→ Frame Extraction

→ Face Detection using OpenCV DNN

→ Face Cropping with Padding

→ ResNet18 Classification

→ Video Level Prediction



\## Model



Architecture:

ResNet18 Transfer Learning



Input:

224 x 224 face images



Classes:

\- Original

\- Deepfake



\## Results



Validation Accuracy:

66.67%



Precision:

69.14%



Recall:

60.21%



F1 Score:

64.37%



\## Running the Project



Install dependencies:



