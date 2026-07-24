import os


def count_videos(folder):

    videos = os.listdir(folder)

    return len([
        v for v in videos
        if os.path.isdir(os.path.join(folder, v))
    ])



def count_faces(folder):

    total = 0

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.endswith(".jpg"):
                total += 1

    return total



if __name__ == "__main__":

    original_path = "data/faces_padding/original"
    deepfake_path = "data/faces_padding/deepfakes"


    print("Original videos:",
          count_videos(original_path))

    print("Deepfake videos:",
          count_videos(deepfake_path))


    print()

    print("Original faces:",
          count_faces(original_path))

    print("Deepfake faces:",
          count_faces(deepfake_path))