from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import FaceDataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = FaceDataset(
    split_file="train.txt",
    transform=transform
)

val_dataset = FaceDataset(
    split_file="val.txt",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

if __name__ == "__main__":
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    images, labels = next(iter(train_loader))

    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")