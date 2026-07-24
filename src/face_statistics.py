import os
from PIL import Image


def get_statistics(folder):

    widths = []
    heights = []

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.endswith(".jpg"):

                path = os.path.join(root, file)

                image = Image.open(path)

                w, h = image.size

                widths.append(w)
                heights.append(h)


    return widths, heights



if __name__ == "__main__":

    folders = [
        "data/faces/original",
        "data/faces/deepfakes"
    ]


    for folder in folders:

        widths, heights = get_statistics(folder)


        print("\nFolder:", folder)

        print("Images:", len(widths))

        print(
            "Width  -> min:",
            min(widths),
            "max:",
            max(widths),
            "avg:",
            sum(widths)/len(widths)
        )

        print(
            "Height -> min:",
            min(heights),
            "max:",
            max(heights),
            "avg:",
            sum(heights)/len(heights)
        )