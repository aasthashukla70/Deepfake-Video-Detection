from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    def __init__(self, split_file, transform=None):
        self.transform = transform
        self.samples = []

        dataset_path = Path("data/faces")
        split_path = Path("data/splits") / split_file

        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        with open(split_path, "r") as f:
            video_list = [line.strip() for line in f if line.strip()]

        for video in video_list:
            image_folder = dataset_path / video

            if "original" in video:
                label = 0
            else:
                label = 1

            for image_path in sorted(image_folder.glob("*.jpg")):
                self.samples.append((image_path, label))

        print(f"Loaded {len(self.samples)} images from {split_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    dataset = FaceDataset("train.txt")

    print(f"Dataset size: {len(dataset)}")

    image, label = dataset[0]

    print(f"Image size: {image.size}")
    print(f"Label: {label}")