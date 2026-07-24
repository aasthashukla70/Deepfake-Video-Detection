import torch

from video_dataloader import val_loader
from video_model import CNNLSTM

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


model = CNNLSTM().to(device)


model.load_state_dict(
    torch.load(
        "models/cnn_lstm_frozen_best.pth",
        map_location=device
    )
)


model.eval()


print("Model loaded successfully.")

def evaluate():

    TP = TN = FP = FN = 0


    with torch.no_grad():

        for videos, labels in val_loader:

            videos = videos.to(device)
            labels = labels.to(device)


            outputs = model(videos)


            _, predictions = torch.max(outputs, 1)


            for actual, predicted in zip(labels, predictions):

                actual = actual.item()
                predicted = predicted.item()


                # 0 = Original
                # 1 = Deepfake

                if actual == 0 and predicted == 0:
                    TN += 1

                elif actual == 1 and predicted == 1:
                    TP += 1

                elif actual == 0 and predicted == 1:
                    FP += 1

                elif actual == 1 and predicted == 0:
                    FN += 1



    total = TP + TN + FP + FN


    accuracy = (TP + TN) / total


    precision = (
        TP / (TP + FP)
        if (TP + FP) != 0
        else 0
    )


    recall = (
        TP / (TP + FN)
        if (TP + FN) != 0
        else 0
    )


    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) != 0
        else 0
    )


    print(f"\nAccuracy : {accuracy*100:.2f}%")

    print("\nConfusion Matrix")
    print(f"[[{TN} {FP}]")
    print(f" [{FN} {TP}]]")


    print("\nMetrics")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")



if __name__ == "__main__":
    evaluate()