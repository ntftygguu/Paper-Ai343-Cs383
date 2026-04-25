import os

BASE = r"C:\Users\Speed\OneDrive - MUST University\Documents\Projects sem6\paper\dataset_clean"

splits = ["train", "val", "test"]
classes = ["cancer", "normal"]

for s in splits:
    print("\n", s.upper())
    for c in classes:
        path = os.path.join(BASE, s, c)
        count = len(os.listdir(path))
        print(f"{c}: {count}")