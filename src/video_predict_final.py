import cv2
import torch

from PIL import Image
from torchvision import transforms

from src.model import DeepFakeCNN


# ==========================
# Configuration
# ==========================

import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "resnet18_final.pth"
)

NUM_FRAMES = 60

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ==========================
# Load Face Detector
# ==========================

face_detector = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)



# ==========================
# Load Classifier
# ==========================

model = DeepFakeCNN().to(device)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)


model.eval()


print("Model loaded successfully.")



# ==========================
# Transform
# ==========================

transform = transforms.Compose([

    transforms.Resize(
        (224,224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])



# ==========================
# Face Detection
# SAME AS TRAINING
# ==========================

def detect_face(image):

    h, w = image.shape[:2]


    blob = cv2.dnn.blobFromImage(
        image,
        1.0,
        (300,300),
        (104.0,177.0,123.0)
    )


    face_detector.setInput(blob)

    detections = face_detector.forward()


    for i in range(detections.shape[2]):

        confidence = detections[0,0,i,2]


        if confidence > 0.5:


            box = (
                detections[0,0,i,3:7]
                *
                [w,h,w,h]
            )


            x1,y1,x2,y2 = box.astype(int)


            face_width = x2-x1
            face_height = y2-y1


            padding = 0.25


            x1 = int(
                x1 - padding*face_width
            )

            y1 = int(
                y1 - padding*face_height
            )

            x2 = int(
                x2 + padding*face_width
            )

            y2 = int(
                y2 + padding*face_height
            )


            x1=max(0,x1)
            y1=max(0,y1)

            x2=min(w,x2)
            y2=min(h,y2)


            return image[y1:y2,x1:x2]


    return None



# ==========================
# Extract Frames
# ==========================

def extract_frames(video_path):

    cap = cv2.VideoCapture(
        video_path
    )


    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    indices = set(
        torch.linspace(
            0,
            total_frames-1,
            NUM_FRAMES
        ).long().tolist()
    )


    frames=[]

    current=0


    while True:

        ret,frame = cap.read()

        if not ret:
            break


        if current in indices:

            frames.append(frame)


        current += 1


    cap.release()


    return frames



# ==========================
# Video Prediction
# ==========================

def predict_video(video_path):


    frames = extract_frames(
        video_path
    )


    deepfake_scores=[]


    faces_found=0



    for frame in frames:


        face = detect_face(
            frame
        )


        if face is None:
            continue


        faces_found += 1


        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )


        image = Image.fromarray(
            face
        )


        image = transform(
            image
        )


        image = image.unsqueeze(0)

        image = image.to(device)



        with torch.no_grad():

            output = model(
                image
            )


            probability = torch.softmax(
                output,
                dim=1
            )


            deepfake_probability = probability[0][1].item()


            deepfake_scores.append(
                deepfake_probability
            )



    if len(deepfake_scores)==0:

        return (
            "No face detected",
            0
        )


    average_score = sum(
        deepfake_scores
    ) / len(deepfake_scores)



    if average_score >= 0.5:

        label="Deepfake"

        confidence=average_score

    else:

        label="Original"

        confidence=1-average_score



    print(
        f"Faces analysed: {faces_found}/{len(frames)}"
    )


    return (
        label,
        confidence*100
    )



# ==========================
# Main
# ==========================

if __name__=="__main__":

    import sys


    if len(sys.argv)!=2:

        print(
            "python src/video_predict_final.py <video_path>"
        )


    else:

        label,confidence=predict_video(
            sys.argv[1]
        )


        print(
            f"Prediction : {label}"
        )

        print(
            f"Confidence : {confidence:.2f}%"
        )