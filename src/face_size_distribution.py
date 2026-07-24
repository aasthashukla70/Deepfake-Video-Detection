import os
from PIL import Image


def analyze(folder):

    small = 0
    medium = 0
    large = 0

    total = 0


    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.endswith(".jpg"):

                path = os.path.join(root, file)

                image = Image.open(path)

                w, h = image.size

                size = min(w, h)

                total += 1


                if size < 64:
                    small += 1

                elif size < 128:
                    medium += 1

                else:
                    large += 1


    print("\nFolder:", folder)

    print("Total:", total)

    print(
        "Small (<64):",
        small,
        f"({small/total*100:.2f}%)"
    )

    print(
        "Medium (64-128):",
        medium,
        f"({medium/total*100:.2f}%)"
    )

    print(
        "Large (>128):",
        large,
        f"({large/total*100:.2f}%)"
    )


if __name__ == "__main__":

    analyze("data/faces/original")

    analyze("data/faces/deepfakes")