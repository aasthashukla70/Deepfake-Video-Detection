from pathlib import Path
import os
import shutil

from src.frame_extractor import extract_frames
from src.face_extractor import extract_faces
from src.predict import predict

def process_video(video_path):

    video_path = Path(video_path)

    print("Processing video:")
    print(video_path)

    frames_folder = Path("data/temp_frames")

    faces_folder = Path("data/temp_faces")

    if frames_folder.exists():
        shutil.rmtree(frames_folder)

    if faces_folder.exists():
        shutil.rmtree(faces_folder)

    frames_folder.mkdir(parents=True)
    faces_folder.mkdir(parents=True)


    extract_frames(
        str(video_path),
        str(frames_folder)
    )


    print("Frames extracted successfully.")

    extract_faces(
        str(frames_folder),
        str(faces_folder)
    )

    print("Faces extracted successfully.")

def predict_video(faces_folder):

    predictions = []

    faces_folder = Path(faces_folder)

    for face_image in sorted(faces_folder.glob("*.jpg")):

        prediction, confidence = predict(str(face_image))

        predictions.append(prediction)

        print(
            face_image.name,
            "→",
            prediction,
            f"({confidence:.2f}%)"
        )


    deepfake_count = predictions.count("Deepfake")
    original_count = predictions.count("Original")


    if deepfake_count > original_count:
        final_prediction = "Deepfake"
    else:
        final_prediction = "Original"


    confidence = (
        max(deepfake_count, original_count)
        / len(predictions)
    ) * 100


    print("\nFinal Video Result")
    print("-------------------------")
    print("Prediction:", final_prediction)
    print(f"Confidence: {confidence:.2f}%")

    return final_prediction, confidence

def detect_video(video_path):

    process_video(video_path)

    prediction, confidence = predict_video(
        "data/temp_faces"
    )

    return prediction, confidence

if __name__ == "__main__":

    prediction, confidence = detect_video(
        "data/test/test_video.mp4"
    )

    print("\nReturned Result")
    print(prediction)
    print(confidence)