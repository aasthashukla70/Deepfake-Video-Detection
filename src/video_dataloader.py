from torch.utils.data import DataLoader
from torchvision import transforms

from video_dataset import VideoDataset


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


train_dataset = VideoDataset(
    "train.txt",
    transform
)

val_dataset = VideoDataset(
    "val.txt",
    transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)


val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)


if __name__ == "__main__":

    print("Training videos:", len(train_dataset))
    print("Validation videos:", len(val_dataset))

    videos, labels = next(iter(train_loader))

    print("Video batch shape:", videos.shape)
    print("Labels shape:", labels.shape)