import cv2
import os


# =========================
# Load Face Detector Model
# =========================

prototxt_path = "models/deploy.prototxt"
model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"


face_detector = cv2.dnn.readNetFromCaffe(
    prototxt_path,
    model_path
)



# =========================
# Detect Face + Add Padding
# =========================

def detect_face(image):

    h, w = image.shape[:2]


    blob = cv2.dnn.blobFromImage(
        image,
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )


    face_detector.setInput(blob)

    detections = face_detector.forward()


    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]


        if confidence > 0.5:

            box = (
                detections[0, 0, i, 3:7]
                *
                [w, h, w, h]
            )


            x1, y1, x2, y2 = box.astype(int)


            # Original face dimensions

            face_width = x2 - x1
            face_height = y2 - y1


            # Add padding
            padding = 0.25


            x1 = int(x1 - padding * face_width)
            y1 = int(y1 - padding * face_height)

            x2 = int(x2 + padding * face_width)
            y2 = int(y2 + padding * face_height)



            # Keep coordinates inside image

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(w, x2)
            y2 = min(h, y2)



            face = image[y1:y2, x1:x2]


            return face


    return None





# =========================
# Extract Faces From Frames
# =========================

def extract_faces(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)


    count = 1


    for filename in sorted(os.listdir(input_folder)):


        if filename.endswith(".jpg"):


            path = os.path.join(
                input_folder,
                filename
            )


            image = cv2.imread(path)


            if image is None:
                print("Could not read:", filename)
                continue



            face = detect_face(image)



            if face is not None:


                output_path = os.path.join(
                    output_folder,
                    f"face_{count:04d}.jpg"
                )


                cv2.imwrite(
                    output_path,
                    face
                )


                count += 1



            else:

                print(
                    "No face detected:",
                    filename
                )



    print(
        "Total faces saved:",
        count - 1
    )





# =========================
# Process Dataset
# =========================

def process_dataset(input_root, output_root):


    for video_folder in sorted(os.listdir(input_root)):


        input_folder = os.path.join(
            input_root,
            video_folder
        )


        output_folder = os.path.join(
            output_root,
            video_folder
        )



        if os.path.isdir(input_folder):


            print(
                "\nProcessing:",
                video_folder
            )


            extract_faces(
                input_folder,
                output_folder
            )





# =========================
# Main
# =========================

def main():


    original_input = (
        "data/processed/original"
    )

    deepfake_input = (
        "data/processed/deepfakes"
    )


    # New dataset
    # Keeping old data untouched

    original_output = (
        "data/faces_padding/original"
    )

    deepfake_output = (
        "data/faces_padding/deepfakes"
    )



    print(
        "Processing Original Videos"
    )


    process_dataset(
        original_input,
        original_output
    )



    print(
        "\nProcessing Deepfake Videos"
    )


    process_dataset(
        deepfake_input,
        deepfake_output
    )





if __name__ == "__main__":

    main()