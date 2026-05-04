"""
evaluate_test.py — Final test set evaluation with confusion matrix + metrics plots.

FIX: No label inversion needed here — CLASS_NAMES in config.py correctly
     enforces normal=0, cancer=1 via class_names= argument.
     (test_fixed.py with invert_labels() is now obsolete and should be deleted.)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

try:
    from config import CLASS_NAMES, CONFIG, RESULTS_DIR
except ImportError:
    from .config import CLASS_NAMES, CONFIG, RESULTS_DIR


AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# ARGS
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set.")
    parser.add_argument("--data-dir",    type=Path,  default=Path(CONFIG["dataset_dir"]))
    parser.add_argument("--model-path",  type=Path,  default=Path(CONFIG["model_path"]))
    parser.add_argument("--results-dir", type=Path,  default=RESULTS_DIR)
    parser.add_argument("--img-size",    type=int,   default=CONFIG["img_size"])
    parser.add_argument("--batch-size",  type=int,   default=CONFIG["batch_size"])
    parser.add_argument("--threshold",   type=float, default=0.5)
    return parser.parse_args()


# ============================================================
# LOAD TEST DATA
# ============================================================
def load_test_dataset(data_dir: Path, img_size: int, batch_size: int):
    test_dir = data_dir / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size  = (img_size, img_size),
        batch_size  = batch_size,
        label_mode  = "binary",
        class_names = list(CLASS_NAMES),   # enforces normal=0, cancer=1
        shuffle     = False,
    )

    print("\n===== CLASS ORDER =====")
    for i, name in enumerate(test_ds.class_names):
        print(f"  {i} = {name}")

    return test_ds.cache().prefetch(AUTOTUNE)


# ============================================================
# PREDICTIONS
# ============================================================
def collect_predictions(model, dataset, threshold: float):
    y_true, y_pred, y_prob = [], [], []

    for images, labels in dataset:
        probs = model.predict(images, verbose=0).flatten()
        preds = (probs >= threshold).astype(int)

        y_true.extend(labels.numpy().astype(int).ravel())
        y_pred.extend(preds)
        y_prob.extend(probs)

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


# ============================================================
# PLOTS
# ============================================================
def save_confusion_matrix(cm, class_names, threshold: float, results_dir: Path):
    path = results_dir / f"confusion_matrix_threshold_{threshold}.png"
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot       = True,
        fmt         = "d",
        cmap        = "Blues",
        xticklabels = class_names,
        yticklabels = class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Test Confusion Matrix — Threshold {threshold}")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Confusion matrix saved → {path}")
    return path


def save_metrics_chart(metrics: dict, threshold: float, results_dir: Path):
    path = results_dir / f"metrics_threshold_{threshold}.png"
    plt.figure(figsize=(8, 5))
    plt.bar(list(metrics.keys()), list(metrics.values()))
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(f"Test Cancer Detection Metrics — Threshold {threshold}")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Metrics chart saved  → {path}")
    return path


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()

    data_dir    = args.data_dir.expanduser().resolve()
    model_path  = args.model_path.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== EVALUATION CONFIG =====")
    print(f"  Dataset    : {data_dir}")
    print(f"  Model      : {model_path}")
    print(f"  Threshold  : {args.threshold}")

    test_ds = load_test_dataset(data_dir, args.img_size, args.batch_size)

    print("\n[INFO] Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("Output shape:", model.output_shape)
    model.summary()

    print("\n[INFO] Generating predictions...")
    y_true, y_pred, y_prob = collect_predictions(model, test_ds, args.threshold)

    print("\n===== LABEL CHECK =====")
    print("True labels     :", np.unique(y_true, return_counts=True))
    print("Predicted labels:", np.unique(y_pred, return_counts=True))

    print("\n===== RAW PREDICTION CHECK =====")
    print(f"  Min prob  : {y_prob.min():.4f}")
    print(f"  Max prob  : {y_prob.max():.4f}")
    print(f"  Mean prob : {y_prob.mean():.4f}")

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_true, y_pred, target_names=list(CLASS_NAMES), zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("\n===== CONFUSION MATRIX =====")
    print(cm)
    print("  Rows = Actual | Columns = Predicted")
    print("  [0]=normal  [1]=cancer")

    tn, fp, fn, tp = cm.ravel()
    accuracy  = (tp + tn) / np.sum(cm)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    print("\n===== CANCER DETECTION METRICS =====")
    print(f"  Threshold   : {args.threshold}")
    print(f"  Accuracy    : {accuracy:.4f}")
    print(f"  Precision   : {precision:.4f}  (of predicted cancers, how many are real)")
    print(f"  Recall      : {recall:.4f}  (of real cancers, how many we catch)")
    print(f"  Specificity : {specificity:.4f}  (of real normals, how many we catch)")
    print(f"  F1-score    : {f1:.4f}")
    print(f"\n  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    metrics = {
        "Accuracy"        : accuracy,
        "Cancer Precision": precision,
        "Cancer Recall"   : recall,
        "Cancer F1"       : f1,
        "Specificity"     : specificity,
    }

    save_confusion_matrix(cm, list(CLASS_NAMES), args.threshold, results_dir)
    save_metrics_chart(metrics, args.threshold, results_dir)

    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()