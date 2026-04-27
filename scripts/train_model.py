import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# =============================
# CONFIG
# =============================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 5

BASE_PATH = r"C:\Users\Speed\OneDrive - MUST University\Documents\Projects sem6\paper\dataset_clean"


# =============================
# LOAD DATA
# =============================
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


# =============================
# CLASS WEIGHTS (compute BEFORE cache)
# =============================
y_true = np.concatenate([y.numpy().flatten() for _, y in train_ds])

class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_true),
    y=y_true
)

class_weights = {i: w for i, w in enumerate(class_weights_arr)}
print("Class weights:", class_weights)


# =============================
# PERFORMANCE OPTIMIZATION
# =============================
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)


# =============================
# DATA AUGMENTATION
# =============================
data_augmentation = tf.keras.Sequential([
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
])


# =============================
# MODEL ARCHITECTURE (CORRECTED)
# =============================
def build_model():
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Augmentation + preprocessing
    x = data_augmentation(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)

    # Base model
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=x
    )

    base_model.trainable = False

    # Custom classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )

    return model, base_model


# =============================
# TRAINING PIPELINE
# =============================
def train():
    model, base_model = build_model()

    print("\n===== MODEL SUMMARY =====\n")
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # =============================
    # STAGE 1 - TRAIN HEAD
    # =============================
    print("\n===== STAGE 1: TRAIN TOP LAYERS =====\n")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE1,
        class_weight=class_weights,
        callbacks=[early_stop]
    )

    # =============================
    # STAGE 2 - FINE TUNING
    # =============================
    print("\n===== STAGE 2: FINE-TUNING =====\n")

    base_model.trainable = True

    # Freeze BatchNorm layers only
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE2,
        class_weight=class_weights,
        callbacks=[early_stop]
    )

    # =============================
    # TEST EVALUATION
    # =============================
    print("\n===== TEST EVALUATION =====\n")
    test_loss, test_acc = model.evaluate(test_ds)
    print("Test Accuracy:", test_acc)

    # =============================
    # SAVE MODEL
    # =============================
    model.save("cancer_model.keras")
    print("Model saved successfully.")


# =============================
# RUN
# =============================
if __name__ == "__main__":
    train()