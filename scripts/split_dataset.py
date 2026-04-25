import os
import shutil
import random

BASE = r"C:\Users\Speed\OneDrive - MUST University\Documents\Projects sem6\paper"

source = os.path.join(BASE, "data", "raw")
dest = os.path.join(BASE, "dataset_clean")

os.makedirs(dest, exist_ok=True)

split = {"train": 0.7, "val": 0.15, "test": 0.15}

label_map = {
    "malignant": "cancer",
    "benign": "cancer",
    "normal": "normal"
}

classes = ["cancer", "normal"]

# create folders
for s in split:
    for c in classes:
        os.makedirs(os.path.join(dest, s, c), exist_ok=True)

# process dataset
for folder in os.listdir(source):
    folder_path = os.path.join(source, folder)

    if not os.path.isdir(folder_path):
        continue

    label = label_map.get(folder.lower())
    if label is None:
        continue

    files = os.listdir(folder_path)
    random.shuffle(files)

    n = len(files)
    train_end = int(n * split["train"])
    val_end = train_end + int(n * split["val"])

    splits = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }

    for split_name, split_files in splits.items():
        out_dir = os.path.join(dest, split_name, label)

        for f in split_files:
            shutil.copy(
                os.path.join(folder_path, f),
                os.path.join(out_dir, f)
            )

print("DONE: dataset created successfully.")