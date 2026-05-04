"""
split_dataset.py — Splits raw data into train / val / test sets.

FIX #1: benign is now correctly mapped to "normal" (not "cancer").
         Benign tumors are non-cancerous.
FIX #2: Stratified split ensures balanced class ratios in every split.

Raw folder structure expected:
    data/raw/
        malignant/   → cancer
        benign/      → normal   ← FIXED (was wrongly "cancer")
        normal/      → normal

Output:
    dataset_clean/
        train/
            cancer/
            normal/
        val/
            cancer/
            normal/
        test/
            cancer/
            normal/
"""

import random
import shutil
from collections import defaultdict
from pathlib import Path

try:
    from config import DATASET_DIR, PROJECT_ROOT
except ImportError:
    from .config import DATASET_DIR, PROJECT_ROOT


# ── Settings ─────────────────────────────────────────────────────────────────
SOURCE_DIR = PROJECT_ROOT / "data" / "raw"
DEST_DIR     = DATASET_DIR
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED  = 42

# FIX: benign → "normal" (benign tumors are NOT cancer)
LABEL_MAP = {
    "malignant": "cancer",
    "benign"   : "normal",   # ← FIX: was "cancer" — WRONG
    "normal"   : "normal",
}

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

# ── Setup output directories ──────────────────────────────────────────────────
for split_name in SPLIT_RATIOS:
    for class_name in ("cancer", "normal"):
        (DEST_DIR / split_name / class_name).mkdir(parents=True, exist_ok=True)

# ── Collect all files grouped by label ───────────────────────────────────────
label_to_files: dict[str, list[Path]] = defaultdict(list)

for folder in SOURCE_DIR.iterdir():
    if not folder.is_dir():
        continue

    label = LABEL_MAP.get(folder.name.lower())
    if label is None:
        print(f"[WARNING] Unknown folder '{folder.name}' — skipped.")
        continue

    files = [p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]
    label_to_files[label].extend(files)
    print(f"[INFO] {folder.name!r} → '{label}' — {len(files)} files")

# ── Stratified split per class ────────────────────────────────────────────────
# FIX: split each class independently → guarantees balanced splits
random.seed(RANDOM_SEED)

total_copied = defaultdict(lambda: defaultdict(int))

for label, files in label_to_files.items():
    random.shuffle(files)

    n          = len(files)
    train_end  = int(n * SPLIT_RATIOS["train"])
    val_end    = train_end + int(n * SPLIT_RATIOS["val"])

    split_files = {
        "train": files[:train_end],
        "val"  : files[train_end:val_end],
        "test" : files[val_end:],
    }

    for split_name, split_list in split_files.items():
        dest = DEST_DIR / split_name / label
        for file_path in split_list:
            dest_file = dest / file_path.name
            # Handle filename collisions across source folders
            if dest_file.exists():
                dest_file = dest / f"{file_path.stem}_{file_path.parent.name}{file_path.suffix}"
            shutil.copy(file_path, dest_file)
            total_copied[split_name][label] += 1

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n===== SPLIT SUMMARY =====")
grand_total = 0
for split_name in ("train", "val", "test"):
    cancer_n = total_copied[split_name]["cancer"]
    normal_n = total_copied[split_name]["normal"]
    total    = cancer_n + normal_n
    grand_total += total
    print(f"\n  {split_name.upper()} — {total} images")
    print(f"    cancer : {cancer_n} ({cancer_n/total*100:.1f}%)")
    print(f"    normal : {normal_n} ({normal_n/total*100:.1f}%)")

print(f"\n  TOTAL : {grand_total} images")
print("\n✅ DONE: Dataset split successfully.")