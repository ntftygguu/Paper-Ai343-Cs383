import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

# =============================
# CONFIG
# =============================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30

BASE_PATH = r"C:\Users\Speed\OneDrive - MUST University\Documents\Projects sem6\paper\dataset_clean"
MODEL_SAVE_PATH = "cancer_model_fixed.h5"

AUTOTUNE = tf.data.AUTOTUNE

# =============================
# LOAD DATA
# =============================
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(BASE_PATH, "train"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(BASE_PATH, "val"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = train_ds.class_names

print("\n===== ORIGINAL CLASS ORDER =====")
print(class_names)
for index, name in enumerate(class_names):
    print(f"{name} = {index}")

# Original TensorFlow order:
# cancer = 0
# normal = 1
#
# We invert labels so the model learns:
# normal = 0
# cancer = 1

def invert_labels(images, labels):
    labels = 1 - labels
    return images, labels

train_ds = train_ds.map(invert_labels)
val_ds = val_ds.map(invert_labels)

print("\n===== NEW CLASS ORDER =====")
print("normal = 0")
print("cancer = 1")

# =============================
# CHECK TRAIN LABEL COUNTS
# =============================
train_labels = []

for images, labels in train_ds:
    train_labels.extend(labels.numpy().astype(int).flatten())

train_labels = np.array(train_labels)

print("\n===== TRAINING LABEL DISTRIBUTION =====")
print(np.unique(train_labels, return_counts=True))
print("0 = normal")
print("1 = cancer")

class_weights_values = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = dict(enumerate(class_weights_values))

print("\n===== CLASS WEIGHTS =====")
print(class_weights)

# =============================
# PERFORMANCE
# =============================
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# =============================
# MODEL
# =============================
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

model.summary()

# =============================
# CALLBACKS
# =============================
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_recall",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_recall",
        mode="max",
        patience=6,
        restore_best_weights=True,
        verbose=1
    )
]

# =============================
# TRAIN
# =============================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

model.save(MODEL_SAVE_PATH)

print("\nSaved fixed model to:", MODEL_SAVE_PATH)
