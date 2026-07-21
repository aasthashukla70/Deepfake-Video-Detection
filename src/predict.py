import sys
import torch
from PIL import Image
from torchvision import transforms

from src.model import DeepFakeCNN

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = DeepFakeCNN().to(device)

model.load_state_dict(
    torch.load(
        "models/resnet18_best.pth",
        map_location=device
    )
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(probabilities, 1)

    classes = ["Original", "Deepfake"]

    return (
        classes[prediction.item()],
        confidence.item() * 100
    )

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage:")
        print("python src/predict.py <image_path>")
    else:

        prediction, confidence = predict(sys.argv[1])

        print(f"Prediction : {prediction}")
        print(f"Confidence : {confidence:.2f}%")