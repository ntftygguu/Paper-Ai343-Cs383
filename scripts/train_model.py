import os
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 16

BASE_PATH = r"C:\Users\Speed\OneDrive - MUST University\Documents\Projects sem6\paper\dataset_clean"

# -----------------------------
# LOAD DATA
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(BASE_PATH, "train"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(BASE_PATH, "val"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(BASE_PATH, "test"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

print("Classes:", train_ds.class_names)

# -----------------------------
# PERFORMANCE
# -----------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# -----------------------------
# AUGMENTATION (IMPORTANT FIX)
# -----------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
])

# -----------------------------
# CLASS WEIGHTS (FIXED)
# -----------------------------
y_true = np.concatenate([y.numpy().flatten() for _, y in train_ds])

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_true),
    y=y_true
)

class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# -----------------------------
# MODEL (FIXED ARCHITECTURE)
# -----------------------------
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = data_augmentation(inputs)
x = tf.keras.applications.resnet.preprocess_input(x)

base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# IMPORTANT FIX:
# DO NOT freeze too much blindly
base_model.trainable = False

x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)

# -----------------------------
# STAGE 1 TRAINING
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\n===== STAGE 1 =====\n")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    class_weight=class_weights
)

# -----------------------------
# STAGE 2 (FIXED FINE-TUNING)
# -----------------------------
print("\n===== STAGE 2 (FIXED FINE-TUNING) =====\n")

base_model.trainable = True

# FIX: only freeze early layers (NOT 200 blindly)
for layer in base_model.layers[:50]:
    layer.trainable = False

# IMPORTANT FIX: higher LR for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    class_weight=class_weights
)

# -----------------------------
# EVALUATION
# -----------------------------
test_loss, test_acc = model.evaluate(test_ds)
print("\nTEST ACCURACY:", test_acc)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("cancer_model.keras")
print("Model saved.")