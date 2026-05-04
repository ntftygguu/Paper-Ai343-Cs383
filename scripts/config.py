from pathlib import Path

# ============================================================
# config.py — Lung Cancer Detector
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATASET_DIR  = PROJECT_ROOT / "dataset_clean"
RESULTS_DIR  = PROJECT_ROOT / "results"

CLASS_NAMES = ["normal", "cancer"]   # normal=0, cancer=1

CONFIG = {
    # ── Paths ────────────────────────────────────────────────
    "dataset_dir" : str(DATASET_DIR),
    "model_path"  : "cancer_model_ranger_1097.keras",
    "results_dir" : str(RESULTS_DIR),

    # ── Image / batch ────────────────────────────────────────
    "img_size"    : 224,
    "batch_size"  : 16,           # FIX: was 32 — too large for 877 images

    # ── Training ─────────────────────────────────────────────
    "epochs"           : 30,
    "fine_tune_epochs" : 10,
    "learning_rate"    : 1e-4,
    "fine_tune_lr"     : 1e-5,
    "weight_decay"     : 1e-4,
    "fine_tune_wd"     : 1e-5,
    "min_learning_rate": 1e-7,

    # ── Regularisation ───────────────────────────────────────
    "dropout_rate" : 0.4,         # FIX: was 0.5 — slightly too aggressive
    "l2_reg"       : 1e-4,

    # ── Callbacks ────────────────────────────────────────────
    "patience"       : 8,         # FIX: was 6 — give model more time
    "monitor_metric" : "val_auc", # FIX: was "val_recall" — ROOT CAUSE of collapse

    # ── Architecture ─────────────────────────────────────────
    "dense_units_1"   : 256,
    "dense_units_2"   : 128,
    "fine_tune_layers": 20,

    # ── Optimizer (Ranger) ───────────────────────────────────
    "lookahead_sync"  : 6,
    "lookahead_alpha" : 0.5,

    # ── Data pipeline ────────────────────────────────────────
    "shuffle_buffer"  : 1000,
    "seed"            : 42,

    # ── Augmentation ─────────────────────────────────────────
    # FIX: noise_std was 0.05 — too noisy for CT/X-ray images
    "augmentation": {
        "flip"      : "horizontal",
        "rotation"  : 0.1,
        "zoom"      : 0.1,
        "contrast"  : 0.1,
        "brightness": 0.1,
        "noise_std" : 0.01,       # FIX: was 0.05
    },
}