from pathlib import Path
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import torch


class VideoDataset(Dataset):

    def __init__(self, split_file, transform=None):

        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor()
            ])
        self.samples = []


        dataset_path = Path("data/faces")

        split_path = Path("data/splits") / split_file


        if not split_path.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_path}"
            )


        with open(split_path, "r") as f:

            video_list = [
                line.strip()
                for line in f
                if line.strip()
            ]


        for video in video_list:

            video_folder = dataset_path / video


            if "original" in video:
                label = 0
            else:
                label = 1


            frames = sorted(
                video_folder.glob("*.jpg")
            )


            if len(frames) == 60:

                self.samples.append(
                    (
                        frames,
                        label
                    )
                )


        print(
            f"Loaded {len(self.samples)} videos from {split_file}"
        )



    def __len__(self):

        return len(self.samples)



    def __getitem__(self, index):

        frame_paths, label = self.samples[index]


        frames = []


        for frame_path in frame_paths:

            image = Image.open(
                frame_path
            ).convert("RGB")


            if self.transform:
                image = self.transform(image)
            frames.append(image)


        frames = torch.stack(frames)


        return frames, label



if __name__ == "__main__":


    dataset = VideoDataset(
        "train.txt"
    )


    print(
        "Dataset size:",
        len(dataset)
    )


    frames, label = dataset[0]


    print(
        "Frames shape:",
        frames.shape
    )

    print(
        "Label:",
        label
    )