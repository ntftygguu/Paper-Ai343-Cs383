import os
import numpy as np
import tensorflow as tf

# =============================
# CONFIG
# =============================
IMG_SIZE = 224
THRESHOLD = 0.7

BASE_PATH = r"C:\Users\Speed\OneDrive - MUST University\Documents\Projects sem6\paper\dataset_clean"
MODEL_PATH = "cancer_model_fixed.h5"

# =============================
# LOAD MODEL
# =============================
print("\n[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# =============================
# PREDICT SINGLE IMAGE
# =============================
def predict_image(image_path):
    img = tf.keras.utils.load_img(
        image_path,
        target_size=(IMG_SIZE, IMG_SIZE)
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    prob = model.predict(img_batch, verbose=0)[0][0]
    pred = "cancer" if prob > THRESHOLD else "normal"

    return prob, pred

# =============================
# FIND GOOD GRAD-CAM IMAGES
# =============================
for image_class in ["cancer", "normal"]:
    folder = os.path.join(BASE_PATH, "test", image_class)

    image_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]

    image_files.sort()

    print(f"\n===== CORRECTLY PREDICTED {image_class.upper()} IMAGES =====")

    found = 0

    for i, filename in enumerate(image_files):
        image_path = os.path.join(folder, filename)

        prob, pred = predict_image(image_path)

        if pred == image_class:
            print(
                f"INDEX: {i} | "
                f"FILE: {filename} | "
                f"CANCER_PROB: {prob:.4f} | "
                f"PRED: {pred}"
            )

            found += 1

        if found >= 10:
            break

    if found == 0:
        print("No correctly predicted images found for this class.")
        