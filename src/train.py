import os
import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import train_loader, val_loader
from model import DeepFakeCNN


# =========================
# Configuration
# =========================

NUM_EPOCHS = 10
LEARNING_RATE = 0.001

MODEL_PATH = "models/resnet18_best.pth"

os.makedirs("models", exist_ok=True)


# =========================
# Device
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# =========================
# Model Setup
# =========================

model = DeepFakeCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("Model, loss function and optimizer initialized successfully.")



# =========================
# Training
# =========================

def train_one_epoch():

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0


    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(outputs, labels)


        loss.backward()


        optimizer.step()


        running_loss += loss.item()


        _, predicted = torch.max(outputs, 1)


        total += labels.size(0)
        correct += (predicted == labels).sum().item()



    loss = running_loss / len(train_loader)

    accuracy = 100 * correct / total


    return loss, accuracy




# =========================
# Validation
# =========================

def validate():

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)


            outputs = model(images)


            loss = criterion(outputs, labels)


            running_loss += loss.item()


            _, predicted = torch.max(outputs, 1)


            total += labels.size(0)
            correct += (predicted == labels).sum().item()



    loss = running_loss / len(val_loader)

    accuracy = 100 * correct / total


    return loss, accuracy




# =========================
# Main Training Loop
# =========================

if __name__ == "__main__":


    best_accuracy = 0


    for epoch in range(NUM_EPOCHS):

        print("\n" + "="*50)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print("="*50)


        train_loss, train_acc = train_one_epoch()

        val_loss, val_acc = validate()



        print(f"Train Loss      : {train_loss:.4f}")
        print(f"Train Accuracy  : {train_acc:.2f}%")

        print(f"Val Loss        : {val_loss:.4f}")
        print(f"Val Accuracy    : {val_acc:.2f}%")



        if val_acc > best_accuracy:

            best_accuracy = val_acc

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print("Best model saved!")



    print("\nTraining Finished")

    print(
        f"Best Validation Accuracy: {best_accuracy:.2f}%"
    )