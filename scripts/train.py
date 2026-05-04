# ============================================================
# train.py — Lung Cancer Detector
# Tuned for  : 1,097 images (877 train / 220 val)
# Optimizer  : Ranger (RectifiedAdam + Lookahead)
# Model      : EfficientNetB0 + Two-Phase Training
# ============================================================
#
# FIXES APPLIED:
#   1. monitor_metric changed to val_auc (was val_recall → caused collapse)
#   2. class_weight REMOVED from model.fit() (focal loss handles imbalance)
#   3. focal gamma lowered to 0.5 (was 2.0 — too aggressive)
#   4. batch_size reduced to 16 (was 32 — too large for 877 images)
#
# INSTALL DEPENDENCIES FIRST:
#   pip install tensorflow tensorflow-addons scikit-learn
#
# RUN:
#   python train.py
#   python train.py --batch-size 16
#
# OUTPUT:
#   cancer_model_ranger_1097.keras
#   logs/phase1/
#   logs/phase2/
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
    parser.add_argument("--data-dir",       type=Path,  default=Path(CONFIG["dataset_dir"]))
    parser.add_argument("--model-path",     type=Path,  default=Path(CONFIG["model_path"]))
    parser.add_argument("--img-size",       type=int,   default=CONFIG["img_size"])
    parser.add_argument("--batch-size",     type=int,   default=CONFIG["batch_size"])
    parser.add_argument("--epochs",         type=int,   default=CONFIG["epochs"])
    parser.add_argument("--fine-tune-epochs", type=int, default=CONFIG["fine_tune_epochs"])
    parser.add_argument("--learning-rate",  type=float, default=CONFIG["learning_rate"])
    parser.add_argument("--fine-tune-lr",   type=float, default=CONFIG["fine_tune_lr"])
    parser.add_argument("--weight-decay",   type=float, default=CONFIG["weight_decay"])
    parser.add_argument("--dropout-rate",   type=float, default=CONFIG["dropout_rate"])
    parser.add_argument("--patience",       type=int,   default=CONFIG["patience"])
    parser.add_argument("--monitor",        type=str,   default=CONFIG["monitor_metric"])
    parser.add_argument("--seed",           type=int,   default=CONFIG["seed"])
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
    labels = []
    for _, batch_labels in dataset:
        labels.append(batch_labels.numpy().astype("int32").ravel())
    return np.concatenate(labels)


def build_datasets(data_dir: Path, img_size: int, batch_size: int, seed: int):
    train_dir = data_dir / "train"
    val_dir   = data_dir / "val"

    for d in (train_dir, val_dir):
        if not d.exists():
            raise FileNotFoundError(
                f"\n❌ Directory not found: {d}"
                f"\n   Check dataset_dir in config.py"
            )

    shared_kwargs = dict(
        image_size  = (img_size, img_size),
        batch_size  = batch_size,
        label_mode  = "binary",
        class_names = list(CLASS_NAMES),  # forces normal=0, cancer=1
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, shuffle=True, seed=seed, **shared_kwargs
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, shuffle=False, **shared_kwargs
    )

    print("\n===== CLASS ORDER =====")
    for i, name in enumerate(train_ds.class_names):
        print(f"  {i} = {name}")

    train_labels = collect_labels(train_ds)
    val_labels   = collect_labels(val_ds)
    unique, counts = np.unique(train_labels, return_counts=True)

    print("\n===== DATASET SUMMARY =====")
    print(f"  Total images    : {len(train_labels) + len(val_labels)}")
    print(f"  Train images    : {len(train_labels)}")
    print(f"  Val images      : {len(val_labels)}")
    print(f"  Batch size      : {batch_size}")
    print(f"  Steps per epoch : {len(train_labels) // batch_size}")

    print("\n===== TRAINING LABEL DISTRIBUTION =====")
    for u, c in zip(unique, counts):
        label = "normal" if u == 0 else "cancer"
        pct   = c / len(train_labels) * 100
        print(f"  {int(u)} ({label}): {c} samples ({pct:.1f}%)")

    # FIX: class weights kept for logging only — NOT passed to model.fit()
    # focal loss already handles class imbalance — using both causes collapse
    classes = np.unique(train_labels)
    weights = compute_class_weight("balanced", classes=classes, y=train_labels)
    class_weights_info = {int(c): float(w) for c, w in zip(classes, weights)}
    print("\n===== CLASS WEIGHTS (INFO ONLY — not used in training) =====")
    for k, v in class_weights_info.items():
        label = "normal" if k == 0 else "cancer"
        print(f"  {k} ({label}): {v:.4f}")

    n_train        = len(train_labels)
    shuffle_buffer = min(n_train, max(batch_size, CONFIG["shuffle_buffer"]))

    train_ds = train_ds.cache().shuffle(shuffle_buffer, seed=seed).prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, n_train


# ============================================================
# 4. AUGMENTATION
# ============================================================
def build_augmentation() -> tf.keras.Sequential:
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
    l2_reg = tf.keras.regularizers.l2(CONFIG["l2_reg"])

    base_model = tf.keras.applications.EfficientNetB0(
        include_top = False,
        weights     = "imagenet",
        input_shape = (img_size, img_size, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="input_image")
    x = build_augmentation()(inputs)
    x = base_model(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_1")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="drop_1")(x)

    x = tf.keras.layers.Dense(
        CONFIG["dense_units_1"], activation="relu",
        kernel_regularizer=l2_reg, name="dense_1"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn_2")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="drop_2")(x)

    x = tf.keras.layers.Dense(
        CONFIG["dense_units_2"], activation="relu",
        kernel_regularizer=l2_reg, name="dense_2"
    )(x)
    x = tf.keras.layers.Dropout(dropout_rate / 2, name="drop_3")(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    return tf.keras.Model(inputs, outputs, name="LungCancerDetector_Ranger_1097")


# ============================================================
# 6. OPTIMIZER — RANGER
# ============================================================
def build_ranger(learning_rate: float, weight_decay: float) -> tfa.optimizers.Lookahead:
    radam = tfa.optimizers.RectifiedAdam(
        learning_rate     = learning_rate,
        weight_decay      = weight_decay,
        min_lr            = CONFIG["min_learning_rate"],
        warmup_proportion = 0.1,
        epsilon           = 1e-8,
    )
    return tfa.optimizers.Lookahead(
        radam,
        sync_period    = CONFIG["lookahead_sync"],
        slow_step_size = CONFIG["lookahead_alpha"],
    )


# ============================================================
# 7. COMPILE
# ============================================================
def compile_model(model: tf.keras.Model, learning_rate: float, weight_decay: float) -> None:
    model.compile(
        optimizer = build_ranger(learning_rate, weight_decay),
        # FIX: gamma lowered from 2.0 → 0.5
        # gamma=2.0 was too aggressive on a small dataset and
        # combined with the (now removed) class_weight caused full collapse
        loss = tf.keras.losses.BinaryFocalCrossentropy(
            gamma       = 0.5,
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
def build_callbacks(model_path: Path, monitor: str, patience: int, phase: int) -> list:
    # FIX: Phase 1 now monitors val_auc instead of val_recall
    # val_recall=1.0 was trivially achieved by predicting everything as cancer
    metric = monitor  # val_auc for both phases (set in config.py)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath       = str(model_path),
            monitor        = metric,
            mode           = "max",
            save_best_only = True,
            verbose        = 1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor              = metric,
            mode                 = "max",
            patience             = patience,
            restore_best_weights = True,
            verbose              = 1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            mode     = "min",
            factor   = 0.5,
            patience = max(2, patience // 3),
            min_lr   = CONFIG["min_learning_rate"],
            verbose  = 1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir        = f"logs/phase{phase}",
            histogram_freq = 1,
        ),
    ]


# ============================================================
# 9. INFERENCE — PREDICT + STAGE + SUGGEST
# ============================================================
def predict_and_suggest(model: tf.keras.Model, img_array: np.ndarray) -> dict:
    img_batch = np.expand_dims(img_array.astype(np.float32), axis=0)
    prob      = float(model.predict(img_batch, verbose=0)[0][0])

    if prob < 0.30:
        result = "NORMAL"
        stage  = "N/A"
        action = "No malignancy detected. Routine annual screening recommended."
    elif prob < 0.50:
        result = "SUSPICIOUS"
        stage  = "Indeterminate"
        action = "Borderline result. Follow-up CT scan and pulmonologist consultation advised."
    elif prob < 0.70:
        result = "LIKELY MALIGNANT"
        stage  = "Early Stage (Suspected I–II)"
        action = "Biopsy strongly recommended. Refer to oncologist immediately."
    elif prob < 0.90:
        result = "MALIGNANT"
        stage  = "Intermediate Stage (Suspected II–III)"
        action = "Urgent oncology referral. PET scan and full staging workup required."
    else:
        result = "HIGHLY MALIGNANT"
        stage  = "Advanced Stage (Suspected III–IV)"
        action = "Immediate oncology intervention. Multidisciplinary tumor board review advised."

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
# 10. TRAINING SUMMARY
# ============================================================
def print_training_summary(history1, history2, p1_time, p2_time, model, val_ds) -> None:
    p1_epochs = len(history1.history["loss"])
    p2_epochs = len(history2.history["loss"])
    total     = p1_time + p2_time

    best_auc_p1 = max(history1.history.get("val_auc", [0]))
    best_auc_p2 = max(history2.history.get("val_auc", [0]))
    best_recall_p1 = max(history1.history.get("val_recall", [0]))
    best_recall_p2 = max(history2.history.get("val_recall", [0]))

    print("\n" + "="*55)
    print("  TRAINING COMPLETE")
    print("="*55)
    print(f"  Phase 1 epochs     : {p1_epochs}")
    print(f"  Phase 2 epochs     : {p2_epochs}")
    print(f"  Total epochs       : {p1_epochs + p2_epochs}")
    print(f"  Phase 1 time       : {p1_time/60:.1f} min")
    print(f"  Phase 2 time       : {p2_time/60:.1f} min")
    print(f"  Total time         : {total/60:.1f} min")
    print(f"\n  Best Phase 1 AUC   : {best_auc_p1:.4f}")
    print(f"  Best Phase 1 Recall: {best_recall_p1:.4f}")
    print(f"  Best Phase 2 AUC   : {best_auc_p2:.4f}")
    print(f"  Best Phase 2 Recall: {best_recall_p2:.4f}")

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

    print("\n" + "="*55)
    print("  LUNG CANCER DETECTOR")
    print("  Optimizer : Ranger (RAdam + Lookahead)")
    print("="*55)
    print(f"  Dataset dir     : {data_dir}")
    print(f"  Model output    : {model_path}")
    print(f"  Image size      : {args.img_size}x{args.img_size}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Phase 1 epochs  : {args.epochs}")
    print(f"  Phase 2 epochs  : {args.fine_tune_epochs}")
    print(f"  Monitor metric  : {args.monitor}")
    print(f"  Focal gamma     : 0.5  (fixed from 2.0)")
    print(f"  Class weight    : OFF  (focal loss handles imbalance)")
    print("="*55)

    # ── Load datasets ────────────────────────────────────────
    # FIX: build_datasets no longer returns class_weights
    train_ds, val_ds, n_train = build_datasets(
        data_dir   = data_dir,
        img_size   = args.img_size,
        batch_size = args.batch_size,
        seed       = args.seed,
    )

    # ── Build model ──────────────────────────────────────────
    model = build_model(img_size=args.img_size, dropout_rate=args.dropout_rate)

    # ────────────────────────────────────────────────────────
    # PHASE 1 — Train head (base frozen)
    # ────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  PHASE 1: Training Head  |  Base: FROZEN")
    print("="*55)

    compile_model(model, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    model.summary()

    start_p1 = time.time()

    history1 = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = args.epochs,
        class_weight    = None,    # FIX: REMOVED — was causing prediction collapse
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

    # ────────────────────────────────────────────────────────
    # PHASE 2 — Fine-tune top layers
    # ────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  PHASE 2: Fine-Tuning  |  Base: PARTIAL UNFREEZE")
    print(f"  Unfreezing top {CONFIG['fine_tune_layers']} layers of EfficientNetB0")
    print("="*55)

    base_model           = model.get_layer("efficientnetb0")
    base_model.trainable = True
    fine_tune_at         = len(base_model.layers) - CONFIG["fine_tune_layers"]

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    trainable = sum(1 for l in base_model.layers if l.trainable)
    print(f"  Unfrozen: {trainable} / {len(base_model.layers)} base layers")

    compile_model(model, learning_rate=args.fine_tune_lr, weight_decay=CONFIG["fine_tune_wd"])

    start_p2 = time.time()

    history2 = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = p1_epochs + args.fine_tune_epochs,
        initial_epoch   = p1_epochs,
        class_weight    = None,    # FIX: REMOVED
        callbacks       = build_callbacks(
            model_path = model_path,
            monitor    = args.monitor,
            patience   = max(3, args.patience - 3),
            phase      = 2,
        ),
    )

    p2_time = time.time() - start_p2

    model.save(model_path)
    print(f"\n  ✅ Model saved → {model_path}")

    print_training_summary(history1, history2, p1_time, p2_time, model, val_ds)

    return model, history1, history2


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    model, history1, history2 = main()