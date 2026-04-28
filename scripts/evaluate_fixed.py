import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# =============================
# CONFIG
# =============================
IMG_SIZE = 224
BATCH_SIZE = 32

BASE_PATH = r"C:\Users\Speed\OneDrive - MUST University\Documents\Projects sem6\paper\dataset_clean"
MODEL_PATH = "cancer_model_fixed.h5"

THRESHOLD = 0.5

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

AUTOTUNE = tf.data.AUTOTUNE

# =============================
# LOAD TEST DATA
# =============================
test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(BASE_PATH, "test"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

original_class_names = test_ds.class_names

print("\n===== ORIGINAL CLASS ORDER =====")
print(original_class_names)
for index, name in enumerate(original_class_names):
    print(f"{name} = {index}")

# Original:
# cancer = 0
# normal = 1
#
# Fixed model uses:
# normal = 0
# cancer = 1

def invert_labels(images, labels):
    labels = 1 - labels
    return images, labels

test_ds = test_ds.map(invert_labels)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

class_names = ["normal", "cancer"]

print("\n[INFO] Evaluating on TEST set...\n")
print("\n===== FIXED CLASS ORDER =====")
print("normal = 0")
print("cancer = 1")

# =============================
# LOAD MODEL
# =============================
print("\n[INFO] Loading fixed model...\n")
model = tf.keras.models.load_model(MODEL_PATH)

print("\n===== MODEL OUTPUT CHECK =====")
print(model.output_shape)
model.summary()

# =============================
# PREDICTIONS
# =============================
y_true = []
y_pred = []
y_prob = []

print("\n[INFO] Generating predictions...\n")

for images, labels in test_ds:
    probs = model.predict(images, verbose=0).flatten()

    # cancer = 1
    batch_pred = (probs > THRESHOLD).astype(int)

    labels = labels.numpy().astype(int).flatten()

    y_true.extend(labels)
    y_pred.extend(batch_pred)
    y_prob.extend(probs)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

print("\n===== LABEL CHECK =====")
print("True labels:", np.unique(y_true, return_counts=True))
print("Predicted labels:", np.unique(y_pred, return_counts=True))

print("\n===== RAW PREDICTION CHECK =====")
print("Min probability:", y_prob.min())
print("Max probability:", y_prob.max())
print("Mean probability:", y_prob.mean())

# =============================
# METRICS
# =============================
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
))

cm = confusion_matrix(y_true, y_pred)

print("\n===== CONFUSION MATRIX =====")
print(cm)

# Class order:
# normal = 0
# cancer = 1

tn = cm[0, 0]  # actual normal, predicted normal
fp = cm[0, 1]  # actual normal, predicted cancer
fn = cm[1, 0]  # actual cancer, predicted normal
tp = cm[1, 1]  # actual cancer, predicted cancer

accuracy = (tp + tn) / np.sum(cm)
precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)

print("\n===== CANCER DETECTION METRICS =====")
print(f"Threshold : {THRESHOLD}")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# =============================
# CONFUSION MATRIX PLOT
# =============================
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Test Confusion Matrix - Threshold {THRESHOLD}")
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, f"test_confusion_matrix_threshold_{THRESHOLD}.png"),
    dpi=300
)
plt.show()

# =============================
# METRICS BAR CHART
# =============================
metrics = ["Accuracy", "Cancer Precision", "Cancer Recall", "Cancer F1-score"]
values = [accuracy, precision, recall, f1]

plt.figure()
plt.bar(metrics, values)
plt.ylim(0, 1)
plt.title(f"Test Cancer Detection Metrics - Threshold {THRESHOLD}")
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, f"test_metrics_threshold_{THRESHOLD}.png"),
    dpi=300
)
plt.show()