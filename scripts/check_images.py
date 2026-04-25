import matplotlib.pyplot as plt
import os
from tensorflow.keras.utils import load_img

path = r"C:\Users\Speed\OneDrive - MUST University\Documents\Projects sem6\paper\dataset_clean\train"

classes = ["cancer", "normal"]

for c in classes:
    folder = os.path.join(path, c)

    img_name = os.listdir(folder)[0]  # first image
    img_path = os.path.join(folder, img_name)

    img = load_img(img_path)

    plt.imshow(img)
    plt.title(c)
    plt.axis("off")
    plt.show()