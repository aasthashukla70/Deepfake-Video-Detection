import torch
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from dataloader import val_loader
from model import DeepFakeCNN



# =========================
# Configuration
# =========================

MODEL_PATH = "models/experiments/resnet18_frozen_baseline.pth"


# =========================
# Device
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)



# =========================
# Load Model
# =========================

model = DeepFakeCNN(use_dropout=False).to(device)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)


model.eval()


print("Model loaded successfully.")



# =========================
# Evaluation
# =========================

def evaluate():

    y_true = []
    y_pred = []

    wrong_predictions = []

    video_results = {}


    with torch.no_grad():

        for images, labels, paths in val_loader:


            images = images.to(device)
            labels = labels.to(device)


            outputs = model(images)


            probabilities = torch.softmax(
                outputs,
                dim=1
            )


            confidence, predicted = torch.max(
                probabilities,
                1
            )


            for i in range(len(labels)):


                true_label = labels[i].item()

                pred_label = predicted[i].item()

                conf = confidence[i].item()


                path = paths[i]


                y_true.append(true_label)

                y_pred.append(pred_label)



                # Store wrong predictions

                if true_label != pred_label:

                    wrong_predictions.append(
                        (
                            path,
                            true_label,
                            pred_label,
                            conf
                        )
                    )



                # Video level analysis

                parts = path.split("\\")

                video_name = parts[-2]


                if video_name not in video_results:

                    video_results[video_name] = {
                        "correct":0,
                        "total":0
                    }


                video_results[video_name]["total"] += 1


                if true_label == pred_label:

                    video_results[video_name]["correct"] += 1



    # =========================
    # Metrics
    # =========================

    accuracy = (
        np.mean(
            np.array(y_true)
            ==
            np.array(y_pred)
        )
        * 100
    )


    cm = confusion_matrix(
        y_true,
        y_pred
    )


    precision = precision_score(
        y_true,
        y_pred
    )


    recall = recall_score(
        y_true,
        y_pred
    )


    f1 = f1_score(
        y_true,
        y_pred
    )



    print("\nAccuracy : {:.2f}%".format(
        accuracy
    ))


    print("\nConfusion Matrix")

    print(cm)



    print("\nMetrics")

    print(
        "Precision : {:.4f}".format(
            precision
        )
    )

    print(
        "Recall    : {:.4f}".format(
            recall
        )
    )

    print(
        "F1-Score  : {:.4f}".format(
            f1
        )
    )



    # =========================
    # Video Performance
    # =========================

    print("\nVideo Performance")
    print("-------------------------")


    for video, result in video_results.items():

        score = (
            result["correct"]
            /
            result["total"]
        ) * 100


        print(
            f"{video:<15} "
            f"{result['correct']}/{result['total']} "
            f"({score:.2f}%)"
        )



    # =========================
    # Wrong Predictions
    # =========================

    print("\nWrong Predictions:")


    for item in wrong_predictions[:20]:

        path, true, pred, conf = item


        print(
            f"\nPath: {path}"
        )

        print(
            f"True: {true} | "
            f"Predicted: {pred} | "
            f"Confidence: {conf*100:.2f}%"
        )




if __name__ == "__main__":

    evaluate()