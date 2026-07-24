from PIL import Image
import matplotlib.pyplot as plt
import os


def show_samples(folder, title):

    images = []

    files = sorted(os.listdir(folder))[:6]

    for file in files:
        path = os.path.join(folder, file)
        img = Image.open(path)
        images.append(img)


    plt.figure(figsize=(12,4))

    for i, img in enumerate(images):

        plt.subplot(2,3,i+1)
        plt.imshow(img)
        plt.axis("off")


    plt.suptitle(title)
    plt.show()



show_samples(
    "data/faces/deepfakes/469_481",
    "Failed Deepfake Example 469_481"
)


show_samples(
    "data/faces/deepfakes/192_134",
    "Detected Deepfake Example 192_134"
)