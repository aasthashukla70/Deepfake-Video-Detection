import os


DATASET_PATH = "data/faces"


def count_images(folder):
    count = 0

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".jpg"):
                count += 1

    return count


def count_folders(folder):
    return len([
        name for name in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, name))
    ])


def main():

    original_path = os.path.join(DATASET_PATH, "original")
    deepfake_path = os.path.join(DATASET_PATH, "deepfakes")


    original_videos = count_folders(original_path)
    deepfake_videos = count_folders(deepfake_path)


    original_images = count_images(original_path)
    deepfake_images = count_images(deepfake_path)


    print("="*40)
    print("Dataset Analysis Report")
    print("="*40)

    print(f"Original videos   : {original_videos}")
    print(f"Deepfake videos   : {deepfake_videos}")

    print()

    print(f"Original images   : {original_images}")
    print(f"Deepfake images   : {deepfake_images}")

    print()

    print(f"Total images      : {original_images + deepfake_images}")

    print("="*40)


if __name__ == "__main__":
    main()