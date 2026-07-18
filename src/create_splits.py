import os
import random
from pathlib import Path
DATASET_PATH = Path("data/faces")
SPLIT_PATH = Path("data/splits")

ORIGINAL_PATH = DATASET_PATH / "original"
DEEPFAKE_PATH = DATASET_PATH / "deepfakes"

SPLIT_PATH.mkdir(parents=True, exist_ok=True)

def get_video_folders(path):
    folders = []

    for folder in os.listdir(path):
        folder_path = path / folder

        if folder_path.is_dir():
            folders.append(folder)

    folders.sort()
    return folders

original_videos = get_video_folders(ORIGINAL_PATH)
deepfake_videos = get_video_folders(DEEPFAKE_PATH)

print(f"Original videos: {len(original_videos)}")
print(f"Deepfake videos: {len(deepfake_videos)}")

random.seed(42)
random.shuffle(original_videos)
random.shuffle(deepfake_videos)

split_index = int(0.8 * len(original_videos))

train_original = original_videos[:split_index]
val_original = original_videos[split_index:]

train_deepfake = deepfake_videos[:split_index]
val_deepfake = deepfake_videos[split_index:]

print(f"Train Original: {len(train_original)}")
print(f"Validation Original: {len(val_original)}")

print(f"Train Deepfake: {len(train_deepfake)}")
print(f"Validation Deepfake: {len(val_deepfake)}")

def save_split(filename, original_list, deepfake_list):
    with open(SPLIT_PATH / filename, "w") as f:
        for video in original_list:
            f.write(f"original/{video}\n")

        for video in deepfake_list:
            f.write(f"deepfakes/{video}\n")

save_split("train.txt", train_original, train_deepfake)
save_split("val.txt", val_original, val_deepfake)

print("Train and validation splits saved successfully.")