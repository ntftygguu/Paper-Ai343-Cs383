# ============================================================
# train.py — Lung Cancer Detector
# Tuned for  : 1,097 images (877 train / 220 val)
# Optimizer  : Ranger (RectifiedAdam + Lookahead)
# Model      : EfficientNetB0 + Two-Phase Training
# ============================================================
#
# INSTALL DEPENDENCIES FIRST:
#   pip install tensorflow tensorflow-addons scikit-learn
#
# RUN:
#   python train.py                    (uses config.py defaults)
#   python train.py --batch-size 16    (override any setting)
#
# OUTPUT:
#   cancer_model_ranger_1097.keras     (best model saved here)
#   logs/phase1/                       (TensorBoard Phase 1 logs)
#   logs/phase2/                       (TensorBoard Phase 2 logs)
#
# VIEW TENSORBOARD:
#   tensorboard --logdir logs/
# ============================================================

import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.utils.class_weight import compute_class_weight

try:
    from config import CLASS_NAMES, CONFIG
except ImportError:
    from .config import CLASS_NAMES, CONFIG

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# 1. ARGUMENT PARSER
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Lung Cancer Classifier — Ranger | 1,097 images"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(CONFIG["dataset_dir"]),
        help="Path to dataset_clean directory",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(CONFIG["model_path"]),
        help="Where the trained model will be saved",
    )
    parser.add_argument("--img-size",          type=int,   default=CONFIG["img_size"])
    parser.add_argument("--batch-size",        type=int,   default=CONFIG["batch_size"])
    parser.add_argument("--epochs",            type=int,   default=CONFIG["epochs"])
    parser.add_argument("--fine-tune-epochs",  type=int,   default=CONFIG["fine_tune_epochs"])
    parser.add_argument("--learning-rate",     type=float, default=CONFIG["learning_rate"])
    parser.add_argument("--fine-tune-lr",      type=float, default=CONFIG["fine_tune_lr"])
    parser.add_argument("--weight-decay",      type=float, default=CONFIG["weight_decay"])
    parser.add_argument("--dropout-rate",      type=float, default=CONFIG["dropout_rate"])
    parser.add_argument("--patience",          type=int,   default=CONFIG["patience"])
    parser.add_argument("--monitor",           type=str,   default=CONFIG["monitor_metric"])
    parser.add_argument("--seed",              type=int,   default=CONFIG["seed"])
    return parser.parse_args()


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================
def set_seed(seed: int):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    print(f"  Random seed: {seed}")


# ============================================================
# 3. DATASET BUILDER
# ============================================================
def collect_labels(dataset) -> np.ndarray:
    """Collect all labels from dataset into a flat numpy array."""
    labels = []
    for _, batch_labels in dataset:
        labels.append(batch_labels.numpy().astype("int32").ravel())
    return np.concatenate(labels)


def compute_weights(train_labels: np.ndarray) -> dict:
    """
    Compute balanced class weights.
    Critical for cancer detection — penalises missing cancer more.
    """
    classes = np.unique(train_labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_labels,
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}


def build_datasets(data_dir: Path, img_size: int, batch_size: int, seed: int):
    """
    Load train/val datasets.

    CLASS_NAMES = ["normal", "cancer"] enforces:
        normal = 0
        cancer = 1
    No label inversion needed — CLASS_NAMES handles it.
    """
    train_dir = data_dir / "train"
    val_dir   = data_dir / "val"

    # Safety check before loading
    for d in (train_dir, val_dir):
        if not d.exists():
            raise FileNotFoundError(
                f"\n❌ Directory not found: {d}"
                f"\n   Check your dataset_dir in config.py"
            )

    shared_kwargs = dict(
        image_size  = (img_size, img_size),
        batch_size  = batch_size,
        label_mode  = "binary",
        class_names = list(CLASS_NAMES),   # forces normal=0, cancer=1
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        shuffle = True,
        seed    = seed,
        **shared_kwargs,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        shuffle = False,
        **shared_kwargs,
    )

    # ── Report class mapping ────────────────────────────────
    print("\n===== CLASS ORDER =====")
    for i, name in enumerate(train_ds.class_names):
        print(f"  {i} = {name}")

    # ── Label distribution ──────────────────────────────────
    train_labels = collect_labels(train_ds)
    val_labels   = collect_labels(val_ds)
    unique, counts = np.unique(train_labels, return_counts=True)

    print("\n===== DATASET SUMMARY =====")
    print(f"  Total images       : {len(train_labels) + len(val_labels)}")
    print(f"  Train images       : {len(train_labels)}")
    print(f"  Val images         : {len(val_labels)}")
    print(f"  Batch size         : {batch_size}")
    print(f"  Steps per epoch    : {len(train_labels) // batch_size}")

    print("\n===== TRAINING LABEL DISTRIBUTION =====")
    for u, c in zip(unique, counts):
        label = "normal" if u == 0 else "cancer"
        pct   = c / len(train_labels) * 100
        print(f"  {int(u)} ({label}): {c} samples ({pct:.1f}%)")

    # ── Class weights ────────────────────────────────────────
    class_weights = compute_weights(train_labels)
    print("\n===== CLASS WEIGHTS =====")
    for k, v in class_weights.items():
        label = "normal" if k == 0 else "cancer"
        print(f"  {k} ({label}): {v:.4f}")

    # ── Performance pipeline ─────────────────────────────────
    n_train        = len(train_labels)
    shuffle_buffer = min(n_train, max(batch_size, CONFIG["shuffle_buffer"]))

    train_ds = (
        train_ds
        .cache()
        .shuffle(shuffle_buffer, seed=seed)
        .prefetch(AUTOTUNE)
    )
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, class_weights, n_train


# ============================================================
# 4. AUGMENTATION
# ============================================================
def build_augmentation() -> tf.keras.Sequential:
    """
    Strong augmentation for 1,097 image dataset.
    Only active when model called with training=True.

    GaussianNoise — simulates scanner noise in CT/X-ray images.
    All other transforms increase effective dataset size.
    """
    aug = CONFIG["augmentation"]
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip(aug["flip"]),
        tf.keras.layers.RandomRotation(aug["rotation"]),
        tf.keras.layers.RandomZoom(aug["zoom"]),
        tf.keras.layers.RandomContrast(aug["contrast"]),
        tf.keras.layers.RandomBrightness(aug["brightness"]),
        tf.keras.layers.GaussianNoise(aug["noise_std"]),
    ], name="augmentation")


# ============================================================
# 5. MODEL BUILDER
# ============================================================
def build_model(img_size: int, dropout_rate: float) -> tf.keras.Model:
    """
    EfficientNetB0 backbone + strong classification head.
    Tuned for 1,097 images.

    Architecture:
        Input (224x224x3)
          → Augmentation
          → EfficientNetB0 (ImageNet weights, frozen Phase 1)
          → GlobalAveragePooling2D
          → BatchNorm → Dropout(0.5)
          → Dense(256, relu) + L2
          → BatchNorm → Dropout(0.5)
          → Dense(128, relu) + L2
          → Dropout(0.25)
          → Dense(1, sigmoid)

    Why EfficientNetB0?
        Best accuracy/speed tradeoff for small datasets.
        Pretrained on ImageNet → needs very few medical images to adapt.
    """
    l2_reg = tf.keras.regularizers.l2(CONFIG["l2_reg"])

    base_model = tf.keras.applications.EfficientNetB0(
        include_top = False,
        weights     = "imagenet",
        input_shape = (img_size, img_size, 3),
    )
    base_model.trainable = False   # Frozen in Phase 1

    # ── Build functional model ───────────────────────────────
    inputs = tf.keras.Input(
        shape = (img_size, img_size, 3),
        name  = "input_image",
    )

    # Strong augmentation (training only)
    x = build_augmentation()(inputs)

    # EfficientNetB0 handles its own internal normalisation
    x = base_model(x, training=False)

    # ── Classification head ──────────────────────────────────
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)

    x = tf.keras.layers.BatchNormalization(name="bn_1")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="drop_1")(x)

    x = tf.keras.layers.Dense(
        CONFIG["dense_units_1"],
        activation         = "relu",
        kernel_regularizer = l2_reg,
        name               = "dense_1",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn_2")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="drop_2")(x)

    x = tf.keras.layers.Dense(
        CONFIG["dense_units_2"],
        activation         = "relu",
        kernel_regularizer = l2_reg,
        name               = "dense_2",
    )(x)
    x = tf.keras.layers.Dropout(dropout_rate / 2, name="drop_3")(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation = "sigmoid",
        name       = "output",
    )(x)

    return tf.keras.Model(
        inputs,
        outputs,
        name = "LungCancerDetector_Ranger_1097",
    )


# ============================================================
# 6. OPTIMIZER — RANGER
# ============================================================
def build_ranger(learning_rate: float, weight_decay: float) -> tfa.optimizers.Lookahead:
    """
    Ranger = RectifiedAdam (RAdam) + Lookahead

    Why Ranger beats AdamW for 1,097 images:
      RAdam      → fixes Adam's unstable early variance
                   no warmup needed
      Lookahead  → slow cautious steps every 6 updates
                   prevents overfitting on small datasets
      Together   → fastest convergence + best generalisation
                   for small-medium medical imaging datasets

    Args:
        learning_rate : base LR for RectifiedAdam
        weight_decay  : L2 regularisation strength
    """
    radam = tfa.optimizers.RectifiedAdam(
        learning_rate     = learning_rate,
        weight_decay      = weight_decay,
        min_lr            = CONFIG["min_learning_rate"],
        warmup_proportion = 0.1,   # 10% warmup for stable start
        epsilon           = 1e-8,
    )
    return tfa.optimizers.Lookahead(
        radam,
        sync_period    = CONFIG["lookahead_sync"],    # sync every 6 steps
        slow_step_size = CONFIG["lookahead_alpha"],   # 0.5 slow step
    )


# ============================================================
# 7. COMPILE
# ============================================================
def compile_model(
    model: tf.keras.Model,
    learning_rate: float,
    weight_decay: float,
) -> None:
    """
    Compile with Ranger + Focal loss + full medical metrics.

    BinaryFocalCrossentropy(gamma=2):
        Standard for medical imaging class imbalance.
        Focuses training on hard cases (missed cancers).
        Down-weights easy correct predictions.

    Metrics:
        accuracy  → overall correctness
        precision → of predicted cancers, how many are real
        recall    → of real cancers, how many we catch  ← most important
        auc       → overall discrimination ability
    """
    model.compile(
        optimizer = build_ranger(learning_rate, weight_decay),
        loss      = tf.keras.losses.BinaryFocalCrossentropy(
            gamma       = 2.0,
            from_logits = False,
        ),
        metrics = [
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


# ============================================================
# 8. CALLBACKS
# ============================================================
def build_callbacks(
    model_path : Path,
    monitor    : str,
    patience   : int,
    phase      : int,
) -> list:
    """
    Phase 1 → monitor val_recall  (catch every cancer)
    Phase 2 → monitor val_auc     (overall discrimination)

    ReduceLROnPlateau always watches val_loss (more stable signal).
    TensorBoard logs saved to logs/phase1/ and logs/phase2/.
    """
    metric = monitor if phase == 1 else "val_auc"

    return [
        # Save best model weights automatically
        tf.keras.callbacks.ModelCheckpoint(
            filepath       = str(model_path),
            monitor        = metric,
            mode           = "max",
            save_best_only = True,
            verbose        = 1,
        ),
        # Stop training when metric stops improving
        tf.keras.callbacks.EarlyStopping(
            monitor              = metric,
            mode                 = "max",
            patience             = patience,
            restore_best_weights = True,
            verbose              = 1,
        ),
        # Reduce LR when val_loss stalls
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            mode     = "min",
            factor   = 0.5,
            patience = max(2, patience // 3),
            min_lr   = CONFIG["min_learning_rate"],
            verbose  = 1,
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir        = f"logs/phase{phase}",
            histogram_freq = 1,
        ),
    ]


# ============================================================
# 9. INFERENCE — PREDICT + STAGE + SUGGEST
# ============================================================
def predict_and_suggest(model: tf.keras.Model, img_array: np.ndarray) -> dict:
    """
    Run inference on a single image and return clinical suggestion.

    Args:
        model     : trained Keras model
        img_array : numpy array shape (224, 224, 3) in range [0, 255]

    Returns:
        dict with probability, result, suspected_stage, action

    Example:
        import cv2
        img = cv2.imread("scan.jpg")
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = predict_and_suggest(model, img)
    """
    img_batch = np.expand_dims(img_array.astype(np.float32), axis=0)
    prob      = float(model.predict(img_batch, verbose=0)[0][0])

    if prob < 0.30:
        result = "NORMAL"
        stage  = "N/A"
        action = (
            "No malignancy detected. "
            "Routine annual screening recommended."
        )
    elif prob < 0.50:
        result = "SUSPICIOUS"
        stage  = "Indeterminate"
        action = (
            "Borderline result. "
            "Follow-up CT scan and pulmonologist consultation advised."
        )
    elif prob < 0.70:
        result = "LIKELY MALIGNANT"
        stage  = "Early Stage (Suspected I–II)"
        action = (
            "Biopsy strongly recommended. "
            "Refer to oncologist immediately."
        )
    elif prob < 0.90:
        result = "MALIGNANT"
        stage  = "Intermediate Stage (Suspected II–III)"
        action = (
            "Urgent oncology referral. "
            "PET scan and full staging workup required."
        )
    else:
        result = "HIGHLY MALIGNANT"
        stage  = "Advanced Stage (Suspected III–IV)"
        action = (
            "Immediate oncology intervention. "
            "Multidisciplinary tumor board review advised."
        )

    bar = "=" * 55
    print(f"\n{bar}")
    print(f"  Cancer Probability : {prob:.4f}  ({prob*100:.1f}%)")
    print(f"  Result             : {result}")
    print(f"  Suspected Stage    : {stage}")
    print(f"  Recommended Action : {action}")
    print(f"{bar}")

    return {
        "probability"    : prob,
        "result"         : result,
        "suspected_stage": stage,
        "action"         : action,
    }


# ============================================================
# 10. TRAINING SUMMARY PRINTER
# ============================================================
def print_training_summary(
    history1,
    history2,
    p1_time: float,
    p2_time: float,
    model   : tf.keras.Model,
    val_ds,
) -> None:
    """Print full summary after training completes."""
    p1_epochs = len(history1.history["loss"])
    p2_epochs = len(history2.history["loss"])
    total     = p1_time + p2_time

    # Best metrics from Phase 1
    best_recall_p1 = max(history1.history.get("val_recall", [0]))
    best_auc_p1    = max(history1.history.get("val_auc",    [0]))

    # Best metrics from Phase 2
    best_recall_p2 = max(history2.history.get("val_recall", [0]))
    best_auc_p2    = max(history2.history.get("val_auc",    [0]))

    print("\n" + "="*55)
    print("  TRAINING COMPLETE")
    print("="*55)
    print(f"  Phase 1 epochs     : {p1_epochs}")
    print(f"  Phase 2 epochs     : {p2_epochs}")
    print(f"  Total epochs       : {p1_epochs + p2_epochs}")
    print(f"  Phase 1 time       : {p1_time/60:.1f} min")
    print(f"  Phase 2 time       : {p2_time/60:.1f} min")
    print(f"  Total time         : {total/60:.1f} min")
    print(f"\n  Best Phase 1 Recall: {best_recall_p1:.4f}")
    print(f"  Best Phase 1 AUC   : {best_auc_p1:.4f}")
    print(f"  Best Phase 2 Recall: {best_recall_p2:.4f}")
    print(f"  Best Phase 2 AUC   : {best_auc_p2:.4f}")

    print("\n===== FINAL VALIDATION METRICS =====")
    results      = model.evaluate(val_ds, verbose=1)
    metric_names = model.metrics_names
    for name, value in zip(metric_names, results):
        print(f"  {name:<12}: {value:.4f}")

    print("\n  To view TensorBoard:")
    print("  tensorboard --logdir logs/")
    print("="*55)


# ============================================================
# 11. MAIN
# ============================================================
def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir   = args.data_dir.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Print full config ────────────────────────────────────
    print("\n" + "="*55)
    print("  LUNG CANCER DETECTOR")
    print("  Optimizer : Ranger (RAdam + Lookahead)")
    print("  Dataset   : 1,097 images")
    print("="*55)
    print(f"  Dataset dir     : {data_dir}")
    print(f"  Model output    : {model_path}")
    print(f"  Image size      : {args.img_size}x{args.img_size}")
    print(f"  Batch size      : {args.batch_size}  (54 steps/epoch)")
    print(f"  Phase 1 epochs  : {args.epochs}")
    print(f"  Phase 2 epochs  : {args.fine_tune_epochs}")
    print(f"  Phase 1 LR      : {args.learning_rate}")
    print(f"  Phase 2 LR      : {args.fine_tune_lr}")
    print(f"  Weight decay    : {args.weight_decay}")
    print(f"  Dropout         : {args.dropout_rate}")
    print(f"  Patience        : {args.patience}")
    print(f"  Fine-tune layers: {CONFIG['fine_tune_layers']}")
    print(f"  Lookahead sync  : every {CONFIG['lookahead_sync']} steps")
    print("="*55)

    # ── Load datasets ────────────────────────────────────────
    train_ds, val_ds, class_weights, n_train = build_datasets(
        data_dir   = data_dir,
        img_size   = args.img_size,
        batch_size = args.batch_size,
        seed       = args.seed,
    )

    # ── Build model ──────────────────────────────────────────
    model = build_model(
        img_size     = args.img_size,
        dropout_rate = args.dropout_rate,
    )

    # ────────────────────────────────────────────────────────
    # PHASE 1 — Train classification head (base frozen)
    # ────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  PHASE 1: Training Head  |  Base: FROZEN")
    print("  Optimizer: Ranger  |  LR: 1e-4  |  WD: 2e-4")
    print("="*55)

    compile_model(
        model,
        learning_rate = args.learning_rate,
        weight_decay  = args.weight_decay,
    )
    model.summary()

    start_p1 = time.time()

    history1 = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = args.epochs,
        class_weight    = class_weights,
        callbacks       = build_callbacks(
            model_path = model_path,
            monitor    = args.monitor,
            patience   = args.patience,
            phase      = 1,
        ),
    )

    p1_time   = time.time() - start_p1
    p1_epochs = len(history1.history["loss"])
    print(f"\n  Phase 1 done: {p1_epochs} epochs in {p1_time/60:.1f} min")
    print(f"  Per epoch   : {p1_time/p1_epochs:.1f} sec")

    # ────────────────────────────────────────────────────────
    # PHASE 2 — Fine-tune top 15 layers of EfficientNetB0
    # ────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  PHASE 2: Fine-Tuning  |  Base: PARTIAL UNFREEZE")
    print(f"  Unfreezing top {CONFIG['fine_tune_layers']} layers of EfficientNetB0")
    print("  Optimizer: Ranger  |  LR: 1e-5  |  WD: 1e-5")
    print("="*55)

    base_model            = model.get_layer("efficientnetb0")
    base_model.trainable  = True
    fine_tune_at          = len(base_model.layers) - CONFIG["fine_tune_layers"]

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    trainable = sum(1 for l in base_model.layers if l.trainable)
    total_b   = len(base_model.layers)
    print(f"  Unfrozen: {trainable} / {total_b} base layers")

    # Recompile with smaller LR + smaller weight decay
    compile_model(
        model,
        learning_rate = args.fine_tune_lr,
        weight_decay  = CONFIG["fine_tune_wd"],
    )

    start_p2 = time.time()

    history2 = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = p1_epochs + args.fine_tune_epochs,
        initial_epoch   = p1_epochs,
        class_weight    = class_weights,
        callbacks       = build_callbacks(
            model_path = model_path,
            monitor    = args.monitor,
            patience   = max(3, args.patience - 3),
            phase      = 2,
        ),
    )

    p2_time = time.time() - start_p2

    # ── Save final model ─────────────────────────────────────
    model.save(model_path)
    print(f"\n  ✅ Model saved → {model_path}")

    # ── Full summary ─────────────────────────────────────────
    print_training_summary(
        history1, history2,
        p1_time, p2_time,
        model, val_ds,
    )

    return model, history1, history2


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    model, history1, history2 = main()