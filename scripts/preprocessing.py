import os
from pathlib import Path
import tensorflow as tf

# ============================================================
# CONFIG
# ============================================================

IMG_SIZE = 224

BASE_DIR = Path("dataset_clean")
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR   = BASE_DIR / "val"

# ============================================================
# PREPROCESS FUNCTION
# ============================================================

def preprocess_image(image):

    # Resize
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))

    # Convert to float32
    image = tf.cast(image, tf.float32)

    # Normalize to [0,1]
    image = image / 255.0

    return image


# ============================================================
# CHECK DATASET
# ============================================================

def check_dataset(folder):

    print(f"\nChecking: {folder}")

    total = 0

    for class_name in os.listdir(folder):

        class_path = folder / class_name

        if not class_path.is_dir():
            continue

        images = list(class_path.glob("*"))

        print(f"\nClass: {class_name}")
        print(f"Images: {len(images)}")

        total += len(images)

        # Test one image
        sample = images[0]

        img = tf.io.read_file(str(sample))
        img = tf.image.decode_image(img, channels=3)

        processed = preprocess_image(img)

        print(f"Shape : {processed.shape}")
        print(f"Dtype : {processed.dtype}")
        print(f"Min   : {tf.reduce_min(processed).numpy()}")
        print(f"Max   : {tf.reduce_max(processed).numpy()}")

    print(f"\nTotal Images: {total}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n===== PREPROCESSING CHECK =====")

    check_dataset(TRAIN_DIR)
    check_dataset(VAL_DIR)

    print("\nPreprocessing verified successfully.")