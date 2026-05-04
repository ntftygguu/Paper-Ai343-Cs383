"""
gradcam.py — Grad-CAM visualisation for lung cancer detector.

FIX: Corrected import — RESULTS_DIR is now exported from config.py.
     No label inversion needed — CLASS_NAMES enforces correct label order.

Usage:
    python gradcam.py

    Edit IMAGE_CLASS and IMAGE_INDEX below to select a different image.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

try:
    from config import CLASS_NAMES, CONFIG, RESULTS_DIR
except ImportError:
    from .config import CLASS_NAMES, CONFIG, RESULTS_DIR


# ============================================================
# SETTINGS — edit these to select a different image
# ============================================================
IMG_SIZE    = CONFIG["img_size"]
DATASET_DIR = Path(CONFIG["dataset_dir"])
MODEL_PATH  = Path(CONFIG["model_path"])
GRADCAM_DIR = RESULTS_DIR / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_CLASS = "cancer"   # "cancer" or "normal"
IMAGE_INDEX = 14       # index of image in the sorted test folder
THRESHOLD   = 0.5        # FIX: was 0.7 — use 0.5 as default, tune after eval


# ============================================================
# LOAD MODEL
# ============================================================
print("\n[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("\n===== MODEL SUMMARY =====")
model.summary()


# ============================================================
# SELECT IMAGE
# ============================================================
image_folder = DATASET_DIR / "test" / IMAGE_CLASS
image_files  = sorted(
    p for p in image_folder.iterdir()
    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
)

if not image_files:
    raise ValueError(f"No images found in: {image_folder}")

if IMAGE_INDEX >= len(image_files):
    raise IndexError(
        f"IMAGE_INDEX={IMAGE_INDEX} out of range — "
        f"only {len(image_files)} images in {image_folder}"
    )

image_path = image_files[IMAGE_INDEX]
print(f"\n[INFO] Selected image: {image_path}")


# ============================================================
# PREDICT
# ============================================================
image       = tf.keras.utils.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
image_array = tf.keras.utils.img_to_array(image)
image_batch = np.expand_dims(image_array, axis=0)

probability     = model.predict(image_batch, verbose=0)[0][0]
predicted_label = "cancer" if probability >= THRESHOLD else "normal"

print("\n===== PREDICTION =====")
print(f"  Cancer probability : {probability:.4f}")
print(f"  Threshold          : {THRESHOLD}")
print(f"  Predicted label    : {predicted_label}")
print(f"  Actual folder      : {IMAGE_CLASS}")
print(f"  Correct            : {'✅' if predicted_label == IMAGE_CLASS else '❌'}")


# ============================================================
# GRAD-CAM
# ============================================================
# Find EfficientNet base model
base_model = next(
    (layer for layer in model.layers if "efficientnet" in layer.name.lower()),
    None,
)
if base_model is None:
    raise ValueError("EfficientNet base model not found inside the loaded model.")

print(f"\n[INFO] Found base model: {base_model.name}")

last_conv_layer = base_model.get_layer("top_conv")
grad_model      = tf.keras.models.Model(
    inputs  = base_model.input,
    outputs = [last_conv_layer.output, base_model.output],
)

# Collect layers after the base model (classification head)
found_base        = False
after_base_layers = []
for layer in model.layers:
    if layer.name == base_model.name:
        found_base = True
        continue
    if found_base:
        after_base_layers.append(layer)

# Compute gradients
input_tensor = tf.convert_to_tensor(image_batch)
with tf.GradientTape() as tape:
    conv_outputs, base_output = grad_model(input_tensor)
    x = base_output
    for layer in after_base_layers:
        x = layer(x)
    prediction = x[:, 0]

grads        = tape.gradient(prediction, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
conv_outputs = conv_outputs[0]
heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap      = tf.squeeze(heatmap)
heatmap      = np.maximum(heatmap, 0)
heatmap      = heatmap / (np.max(heatmap) + 1e-8)


# ============================================================
# OVERLAY
# ============================================================
original_image  = cv2.imread(str(image_path))
original_image  = cv2.resize(original_image, (IMG_SIZE, IMG_SIZE))

heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
heatmap_uint8   = np.uint8(255 * heatmap_resized)
heatmap_color   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
overlay         = cv2.addWeighted(original_image, 0.6, heatmap_color, 0.4, 0)

original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
heatmap_rgb  = cv2.cvtColor(heatmap_color,  cv2.COLOR_BGR2RGB)
overlay_rgb  = cv2.cvtColor(overlay,         cv2.COLOR_BGR2RGB)


# ============================================================
# SAVE
# ============================================================
base_filename = image_path.stem
heatmap_path  = GRADCAM_DIR / f"gradcam_heatmap_{IMAGE_CLASS}_{base_filename}.png"
overlay_path  = GRADCAM_DIR / f"gradcam_overlay_{IMAGE_CLASS}_{base_filename}.png"
summary_path  = GRADCAM_DIR / f"gradcam_summary_{IMAGE_CLASS}_{base_filename}.png"

cv2.imwrite(str(heatmap_path), heatmap_color)
cv2.imwrite(str(overlay_path), overlay)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(original_rgb)
plt.title(f"Original\nActual: {IMAGE_CLASS}")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(heatmap_rgb)
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay_rgb)
plt.title(f"Overlay\nPredicted: {predicted_label} ({probability:.2f})")
plt.axis("off")

plt.suptitle(
    f"Grad-CAM — {IMAGE_CLASS.upper()} image | "
    f"{'CORRECT ✅' if predicted_label == IMAGE_CLASS else 'WRONG ❌'}",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(summary_path, dpi=300)
plt.close()

print("\n===== SAVED GRAD-CAM OUTPUTS =====")
print(f"  Heatmap : {heatmap_path}")
print(f"  Overlay : {overlay_path}")
print(f"  Summary : {summary_path}")