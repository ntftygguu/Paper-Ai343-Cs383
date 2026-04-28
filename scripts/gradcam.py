import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# =============================
# CONFIG
# =============================
IMG_SIZE = 224

BASE_PATH = r"C:\Users\Speed\OneDrive - MUST University\Documents\Projects sem6\paper\dataset_clean"
MODEL_PATH = "cancer_model_fixed.h5"

RESULTS_DIR = "results\gradcam"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Choose image class to visualize: "cancer" or "normal"
IMAGE_CLASS = "cancer"

# Choose image number from that folder
IMAGE_INDEX = 18

# Model threshold selected from your final test result
THRESHOLD = 0.7

# =============================
# LOAD MODEL
# =============================
print("\n[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("\n===== MODEL SUMMARY =====")
model.summary()

# =============================
# GET TEST IMAGE
# =============================
image_folder = os.path.join(BASE_PATH, "test", IMAGE_CLASS)

image_files = [
    f for f in os.listdir(image_folder)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
]

if len(image_files) == 0:
    raise ValueError(f"No images found in: {image_folder}")

image_files.sort()

image_path = os.path.join(image_folder, image_files[IMAGE_INDEX])

print("\n[INFO] Selected image:")
print(image_path)

# =============================
# LOAD AND PREPROCESS IMAGE
# =============================
img = tf.keras.utils.load_img(
    image_path,
    target_size=(IMG_SIZE, IMG_SIZE)
)

img_array = tf.keras.utils.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)

# =============================
# PREDICT
# =============================
prob = model.predict(img_batch, verbose=0)[0][0]

predicted_label = "cancer" if prob > THRESHOLD else "normal"

print("\n===== PREDICTION =====")
print(f"Cancer probability: {prob:.4f}")
print(f"Threshold: {THRESHOLD}")
print(f"Predicted label: {predicted_label}")
print(f"Actual folder label: {IMAGE_CLASS}")

# =============================
# FIND EFFICIENTNET BASE MODEL
# =============================
base_model = None

for layer in model.layers:
    if "efficientnet" in layer.name.lower():
        base_model = layer
        break

if base_model is None:
    raise ValueError("EfficientNet base model not found inside the loaded model.")

print("\n[INFO] Found base model:", base_model.name)

# Last convolutional layer in EfficientNetB0
last_conv_layer_name = "top_conv"

last_conv_layer = base_model.get_layer(last_conv_layer_name)

# =============================
# BUILD GRAD-CAM MODEL
# =============================
# First, get output of EfficientNet last conv layer
grad_model = tf.keras.models.Model(
    inputs=base_model.input,
    outputs=[
        last_conv_layer.output,
        base_model.output
    ]
)

# Get layers after EfficientNet in your Sequential model
after_base_layers = []
found_base = False

for layer in model.layers:
    if layer.name == base_model.name:
        found_base = True
        continue
    if found_base:
        after_base_layers.append(layer)

# =============================
# COMPUTE GRAD-CAM
# =============================
input_tensor = tf.convert_to_tensor(img_batch)

with tf.GradientTape() as tape:
    # If your model has augmentation layer, it is skipped for Grad-CAM.
    # We pass the image directly through EfficientNet.
    conv_outputs, base_output = grad_model(input_tensor)

    x = base_output

    for layer in after_base_layers:
        x = layer(x)

    prediction = x[:, 0]

# Gradients of prediction with respect to conv feature map
grads = tape.gradient(prediction, conv_outputs)

# Global average pooling of gradients
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0]

# Weight feature maps by gradients
heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

# Normalize heatmap
heatmap = np.maximum(heatmap, 0)
heatmap = heatmap / (np.max(heatmap) + 1e-8)

# =============================
# CREATE HEATMAP OVERLAY
# =============================
original_img = cv2.imread(image_path)
original_img = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))

heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
heatmap_uint8 = np.uint8(255 * heatmap_resized)

heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

overlay = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)

# Convert BGR to RGB for matplotlib
original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# =============================
# SAVE OUTPUTS
# =============================
base_filename = os.path.splitext(os.path.basename(image_path))[0]

heatmap_path = os.path.join(
    RESULTS_DIR,
    f"gradcam_heatmap_{IMAGE_CLASS}_{base_filename}.png"
)

overlay_path = os.path.join(
    RESULTS_DIR,
    f"gradcam_overlay_{IMAGE_CLASS}_{base_filename}.png"
)

summary_path = os.path.join(
    RESULTS_DIR,
    f"gradcam_summary_{IMAGE_CLASS}_{base_filename}.png"
)

cv2.imwrite(heatmap_path, heatmap_color)
cv2.imwrite(overlay_path, overlay)

# =============================
# DISPLAY SUMMARY FIGURE
# =============================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(original_rgb)
plt.title(f"Original Image\nActual: {IMAGE_CLASS}")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(heatmap_rgb)
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay_rgb)
plt.title(
    f"Overlay\nPredicted: {predicted_label}\nCancer prob: {prob:.4f}"
)
plt.axis("off")

plt.tight_layout()
plt.savefig(summary_path, dpi=300)
plt.show()

print("\n===== SAVED GRAD-CAM OUTPUTS =====")
print("Heatmap saved to:", heatmap_path)
print("Overlay saved to:", overlay_path)
print("Summary saved to:", summary_path)