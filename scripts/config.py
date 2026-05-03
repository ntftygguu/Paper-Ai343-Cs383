CLASS_NAMES = ["normal", "cancer"]
CONFIG = {
    "dataset_dir": "dataset_clean",
    "model_path": "cancer_model.keras",
    "img_size": 224,
    "batch_size": 32,
    "epochs": 30,
    "fine_tune_epochs": 10,
    "learning_rate": 1e-4,
    "fine_tune_lr": 1e-5,
    "weight_decay": 1e-4,
    "dropout_rate": 0.5,
    "l2_reg": 1e-4,
    "patience": 6,
    "monitor_metric": "val_recall",
    "seed": 42,
    "shuffle_buffer": 1000,
    "dense_units_1": 256,
    "dense_units_2": 128,
    "fine_tune_layers": 20,
    "fine_tune_wd": 1e-5,
    "min_learning_rate": 1e-7,
    "lookahead_sync": 6,
    "lookahead_alpha": 0.5,

    "augmentation": {
        "flip": "horizontal",
        "rotation": 0.1,
        "zoom": 0.1,
        "contrast": 0.1,
        "brightness": 0.1,
        "noise_std": 0.05,
    }
}