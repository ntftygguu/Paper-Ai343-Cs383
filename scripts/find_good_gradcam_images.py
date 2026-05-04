"""
find_good_gradcam_images.py — Find correctly predicted images for Grad-CAM.

Prints indices and filenames of images the model gets right,
so you can plug those indices into gradcam.py → IMAGE_INDEX.

Usage:
    python find_good_gradcam_images.py
"""

from pathlib import Path

import numpy as np
import tensorflow as tf

try:
    from config import CLASS_NAMES, CONFIG
except ImportError:
    from .config import CLASS_NAMES, CONFIG


# ============================================================
# SETTINGS
# ============================================================
IMG_SIZE    = CONFIG["img_size"]
THRESHOLD   = 0.5              # FIX: was 0.7 — use 0.5 as default
DATASET_DIR = Path(CONFIG["dataset_dir"])
MODEL_PATH  = Path(CONFIG["model_path"])
MAX_SHOW    = 10               # how many correct images to show per class

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


# ============================================================
# LOAD MODEL
# ============================================================
print("\n[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"[INFO] Model loaded from: {MODEL_PATH}")
print(f"[INFO] Output shape: {model.output_shape}")


# ============================================================
# PREDICT SINGLE IMAGE
# ============================================================
def predict_image(image_path: Path) -> tuple[float, str]:
    image       = tf.keras.utils.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    image_array = tf.keras.utils.img_to_array(image)
    image_batch = np.expand_dims(image_array, axis=0)
    probability = float(model.predict(image_batch, verbose=0)[0][0])
    label       = "cancer" if probability >= THRESHOLD else "normal"
    return probability, label


# ============================================================
# SEARCH EACH CLASS
# ============================================================
print(f"\n[INFO] Threshold: {THRESHOLD}")
print(f"[INFO] Searching test set: {DATASET_DIR / 'test'}\n")

for class_name in CLASS_NAMES:
    class_dir = DATASET_DIR / "test" / class_name
    if not class_dir.exists():
        print(f"[WARNING] Folder not found: {class_dir} — skipped")
        continue

    image_files = sorted(
        p for p in class_dir.iterdir()
        if p.suffix.lower() in VALID_EXTENSIONS
    )

    print(f"===== CORRECTLY PREDICTED: {class_name.upper()} =====")
    print(f"  Total images in folder: {len(image_files)}")

    found        = 0
    wrong        = 0
    wrong_sample = []

    for index, image_path in enumerate(image_files):
        probability, predicted_label = predict_image(image_path)

        if predicted_label == class_name:
            if found < MAX_SHOW:
                print(
                    f"  ✅ INDEX: {index:>4} | "
                    f"PROB: {probability:.4f} | "
                    f"FILE: {image_path.name}"
                )
            found += 1
        else:
            wrong += 1
            if len(wrong_sample) < 3:
                wrong_sample.append((index, probability, image_path.name))

    if found == 0:
        print(f"  ❌ No correctly predicted images found for '{class_name}'.")
        print(f"     This suggests the model is collapsing — check training.")
    else:
        print(f"\n  Summary: {found} correct / {wrong} wrong out of {len(image_files)}")
        accuracy = found / len(image_files) * 100
        print(f"  Class accuracy: {accuracy:.1f}%")

    if wrong_sample:
        print(f"\n  Wrong predictions (sample):")
        for idx, prob, fname in wrong_sample:
            print(f"    INDEX: {idx} | PROB: {prob:.4f} | FILE: {fname}")

    print()

print("[INFO] Done. Use an INDEX from above in gradcam.py → IMAGE_INDEX")