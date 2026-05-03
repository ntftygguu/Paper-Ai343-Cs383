import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = r"C:\Users\Speed\Paper-Ai343-C
s383\can
IMG_PATH = r"C:\Users\Speed\Desktop\ct_scan.jpg"
IMG_SIZE = 224

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# LOAD IMAGE
# -----------------------------
img = image.load_img(IMG_PATH, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)

# normalize
img_array = img_array / 255.0

# add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# -----------------------------
# PREDICTION
# -----------------------------
prediction = model.predict(img_array)

print("Raw prediction:", prediction)

# -----------------------------
# MULTI-CLASS
# -----------------------------
classes = ["benign", "malignant", "normal"]

predicted_class = classes[np.argmax(prediction)]

print("Predicted Class:", predicted_class)

# -----------------------------
# SHOW IMAGE
# -----------------------------
plt.imshow(img)
plt.title(f"Prediction: {predicted_class}")
plt.axis("off")
plt.show()