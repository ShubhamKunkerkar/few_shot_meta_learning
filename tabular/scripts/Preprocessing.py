"""
Preprocessing.py
----------------
Preprocesses raw tabular CSV files into episodic PyTorch tensors (.pt)
for few-shot meta-learning (MAML, PLATIPUS, VAMPIRE).

Reads split and support/query configurations directly from tabular/config.json
(or command-line overrides).

Pipeline per CSV:
  1. Feature Engineering:
     - Global One-Hot Encoding for Location (9 categories)
     - Global One-Hot Encoding for EMP_STAT (10 categories)
     - Cyclic time transformation (time_sin, time_cos)
     - Fixed global TSLS outlier capping (1220.0 minutes)
     - Heart Rate alignment (resting delta HR)
     - 10-minute lookahead smoking target
  2. Targeted Simple Normalization:
     - StandardScaler fitted strictly on continuous features (AGEP_A, CIGNOW_A, TSLS_capped, HR)
     - Bounded / binary / one-hot features remain as natural 0/1 or [-1, 1] signals
  3. Temporal Support/Query split:
     - Configurable support_days / k_shot per dataset & test case via tabular/config.json
  4. Exports explicit, self-explanatory .pt files into tabular/data/processed/
"""

import os
import glob
import json
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------
# Global Fixed Vocabularies & Columns
# ----------------------------------------------------------------------
LOCATION_CLASSES = [
    'Home', 'Public Place', 'Office', 'Construction Site',
    'Factory', 'Restaurant', 'Gym', 'Entertainment', 'Market'
]

EMP_STAT_CLASSES = [10, 11, 12, 13, 14, 101, 103, 104, 105, 108]

CONTINUOUS_COLS = ["AGEP_A", "CIGNOW_A", "TSLS_capped", "HR"]
STATIC_COLS = ["time_sin", "time_cos", "is_sleeping", "SEX_A"]
ONE_HOT_LOC_COLS = [f"loc_{loc}" for loc in LOCATION_CLASSES]
ONE_HOT_EMP_COLS = [f"emp_{emp}" for emp in EMP_STAT_CLASSES]

# Full 27 ordered features
FEATURE_COLS = STATIC_COLS + CONTINUOUS_COLS + ONE_HOT_LOC_COLS + ONE_HOT_EMP_COLS
TARGET_COL = "Did the user smoke in next 10 mins"
DEFAULT_TSLS_CAP = 1220.0


def engineer_features(
    df: pd.DataFrame,
    tsls_cap: float = DEFAULT_TSLS_CAP
) -> pd.DataFrame:
    """
    Applies standard feature engineering to a raw tabular DataFrame:
      1. Cyclic time encoding (time_sin, time_cos).
      2. Sex normalization (0/1).
      3. Global TSLS capping.
      4. HR delta alignment.
      5. Global One-Hot Encoding for Location (9 classes).
      6. Global One-Hot Encoding for EMP_STAT (10 classes).
      7. Next-10-min target creation.
    """
    df = df.copy()

    # 1. Cyclic time encoding (24h period)
    df["time_sin"] = np.sin(2 * np.pi * df["time"] / 24.0).astype(np.float32)
    df["time_cos"] = np.cos(2 * np.pi * df["time"] / 24.0).astype(np.float32)

    # 2. Binary / Static signals
    df["is_sleeping"] = df["is_sleeping"].astype(np.float32)
    df["SEX_A"] = (df["SEX_A"] == 2).astype(np.float32)  # 0: Male (1), 1: Female (2)

    # 3. Continuous signals
    df["AGEP_A"] = df["AGEP_A"].astype(np.float32)
    df["CIGNOW_A"] = df["CIGNOW_A"].astype(np.float32)
    df["TSLS_capped"] = df["TSLS"].clip(upper=tsls_cap).astype(np.float32)

    # 4. Heart rate delta alignment (if absolute BPM > 40 is detected, subtract baseline)
    if df["HR"].mean() > 40.0:
        sleeping_mask = df["is_sleeping"] == 1
        if sleeping_mask.sum() > 0:
            baseline_hr = df.loc[sleeping_mask, "HR"].median()
        else:
            baseline_hr = df["HR"].quantile(0.25)
        df["HR"] = (df["HR"] - baseline_hr).astype(np.float32)
    else:
        df["HR"] = df["HR"].astype(np.float32)

    # 5. One-Hot Location (9 binary columns)
    for loc in LOCATION_CLASSES:
        df[f"loc_{loc}"] = (df["Location"] == loc).astype(np.float32)

    # 6. One-Hot EMP_STAT (10 binary columns)
    for emp in EMP_STAT_CLASSES:
        df[f"emp_{emp}"] = (df["EMP_STAT"] == emp).astype(np.float32)

    # 7. Next-10-min target shift
    if "Did The User Smoke?" in df.columns:
        group_col = "task_id" if "task_id" in df.columns else None
        if group_col:
            df[TARGET_COL] = df.groupby(group_col)["Did The User Smoke?"].shift(-1).fillna(0).astype(np.int64)
        else:
            df[TARGET_COL] = df["Did The User Smoke?"].shift(-1).fillna(0).astype(np.int64)
    elif TARGET_COL not in df.columns:
        raise ValueError(f"Neither '{TARGET_COL}' nor 'Did The User Smoke?' found in CSV columns.")

    return df


def build_episodes(
    X: np.ndarray,
    y: np.ndarray,
    task_ids: np.ndarray,
    task_subset: np.ndarray,
    support_steps: int
) -> List[Dict[str, Any]]:
    """Splits each task temporally into Support (first S steps) and Query (remaining steps)."""
    episodes = []
    for tid in task_subset:
        idx = np.where(task_ids == tid)[0]
        s_idx, q_idx = idx[:support_steps], idx[support_steps:]

        if len(s_idx) == 0 or len(q_idx) == 0:
            continue

        episodes.append({
            "x_s": torch.tensor(X[s_idx], dtype=torch.float32),
            "y_s": torch.tensor(y[s_idx], dtype=torch.int64),
            "x_q": torch.tensor(X[q_idx], dtype=torch.float32),
            "y_q": torch.tensor(y[q_idx], dtype=torch.int64),
            "task_id": int(tid),
        })
    return episodes


def process_csv(
    csv_path: str,
    output_dir: Path,
    support_steps: int = 144,
    support_days: int = 1,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    scaler: Optional[StandardScaler] = None,
    tsls_cap: float = DEFAULT_TSLS_CAP,
    is_primary: bool = False
) -> Dict[str, Any]:
    """Processes a single CSV and outputs standardized .pt episodic dataset files."""
    csv_file = Path(csv_path)
    stem = csv_file.stem

    print(f"\n{'='*65}")
    print(f"Processing CSV: {csv_file.name}")
    print(f"Support Steps (k_shot): {support_steps} ({support_days} day(s)) | Query Steps: remaining")
    print(f"{'='*65}")

    df_raw = pd.read_csv(csv_path)
    if "task_id" not in df_raw.columns:
        df_raw["task_id"] = 0

    unique_tasks = np.unique(df_raw["task_id"].values)
    n_tasks = len(unique_tasks)
    print(f"  Rows: {len(df_raw):,} | Tasks: {n_tasks:,}")

    # 1. Feature Engineering
    df = engineer_features(df_raw, tsls_cap=tsls_cap)
    y_raw = df[TARGET_COL].values.astype(np.int64)
    task_ids = df["task_id"].values

    is_multitask = n_tasks >= 5

    # 2. Continuous Scaling
    X_cont = df[CONTINUOUS_COLS].values.astype(np.float32)

    if is_multitask:
        # Multi-task dataset -> Perform train / val / test split
        n_test = int(n_tasks * test_frac)
        n_val = int(n_tasks * val_frac)
        n_train = n_tasks - n_val - n_test

        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(unique_tasks)
        train_ids = shuffled[:n_train]
        val_ids = shuffled[n_train:n_train + n_val]
        test_ids = shuffled[n_train + n_val:]

        if scaler is None:
            scaler = StandardScaler()
            train_mask = np.isin(task_ids, train_ids)
            scaler.fit(X_cont[train_mask])

        # Scale continuous features
        X_cont_scaled = scaler.transform(X_cont).astype(np.float32)
        df[CONTINUOUS_COLS] = X_cont_scaled
        X_all = df[FEATURE_COLS].values.astype(np.float32)

        train_eps = build_episodes(X_all, y_raw, task_ids, train_ids, support_steps)
        val_eps = build_episodes(X_all, y_raw, task_ids, val_ids, support_steps)
        test_eps = build_episodes(X_all, y_raw, task_ids, test_ids, support_steps)

        # Save partitioned files
        train_out_path = output_dir / f"{stem}_train_tasks.pt"
        val_out_path = output_dir / f"{stem}_val_tasks.pt"
        test_out_path = output_dir / f"{stem}_test_tasks.pt"

        torch.save(train_eps, train_out_path)
        torch.save(val_eps, val_out_path)
        torch.save(test_eps, test_out_path)

        print(f"  Saved multi-task datasets -> Train: {len(train_eps):,} | Val: {len(val_eps):,} | Test: {len(test_eps):,}")
    else:
        # Single-task / Test-case CSV
        if scaler is None:
            scaler = StandardScaler().fit(X_cont)

        X_cont_scaled = scaler.transform(X_cont).astype(np.float32)
        df[CONTINUOUS_COLS] = X_cont_scaled
        X_all = df[FEATURE_COLS].values.astype(np.float32)

        episodes = build_episodes(X_all, y_raw, task_ids, unique_tasks, support_steps)
        tasks_out_path = output_dir / f"{stem}_tasks.pt"
        torch.save(episodes, tasks_out_path)
        print(f"  Saved test-case dataset -> {tasks_out_path.name} ({len(episodes)} episodes)")

    # 3. Save Metadata
    meta = {
        "source_csv": csv_file.name,
        "feature_cols": FEATURE_COLS,
        "continuous_cols": CONTINUOUS_COLS,
        "static_cols": STATIC_COLS,
        "one_hot_loc_cols": ONE_HOT_LOC_COLS,
        "one_hot_emp_cols": ONE_HOT_EMP_COLS,
        "target_col": TARGET_COL,
        "support_days": support_days,
        "support_steps": support_steps,
        "k_shot": support_steps,
        "n_tasks": n_tasks,
        "location_classes": LOCATION_CLASSES,
        "emp_stat_classes": EMP_STAT_CLASSES,
        "tsls_cap": float(tsls_cap),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
        "seed": seed,
    }
    torch.save(meta, output_dir / f"{stem}_meta.pt")

    return {"scaler": scaler, "tsls_cap": tsls_cap}


def main():
    parser = argparse.ArgumentParser(description="Preprocess tabular CSV files into episodic meta-learning tensors.")
    parser.add_argument("--config", type=str, default="tabular/config.json",
                        help="Path to centralized JSON configuration file.")
    parser.add_argument("--raw_dir", type=str, default=None, help="Directory containing raw CSV files.")
    parser.add_argument("--csv_path", type=str, default=None, help="Optional specific CSV path (or comma-separated paths).")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for .pt files.")
    parser.add_argument("--support_days", type=int, default=None, help="Number of support days (1 day = steps_per_day).")
    parser.add_argument("--k_shot", type=int, default=None, help="Explicit support steps count (overrides support_days).")
    parser.add_argument("--val_frac", type=float, default=None, help="Validation fraction for multi-task datasets.")
    parser.add_argument("--test_frac", type=float, default=None, help="Test fraction for multi-task datasets.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    args = parser.parse_args()

    # 1. Load JSON Configuration if available
    cfg = {}
    if os.path.exists(args.config):
        print(f"Loading configuration from: {args.config}")
        with open(args.config, "r") as f:
            cfg = json.load(f)

    dataset_cfg = cfg.get("dataset", {})
    meta_split_cfg = cfg.get("meta_training_split", {})
    test_cases_cfg = cfg.get("test_cases", {})

    # 2. Resolve Parameters (CLI args override JSON config, which overrides defaults)
    raw_dir = args.raw_dir or dataset_cfg.get("raw_dir", "tabular/data/raw")
    output_dir = args.output_dir or dataset_cfg.get("output_dir", "tabular/data/processed")
    steps_per_day = dataset_cfg.get("steps_per_day", 144)
    seed = args.seed if args.seed is not None else dataset_cfg.get("seed", 42)

    val_frac = args.val_frac if args.val_frac is not None else meta_split_cfg.get("val_frac", 0.15)
    test_frac = args.test_frac if args.test_frac is not None else meta_split_cfg.get("test_frac", 0.15)
    default_support_days = args.support_days if args.support_days is not None else meta_split_cfg.get("support_days", 1)
    default_k_shot = args.k_shot if args.k_shot is not None else meta_split_cfg.get("k_shot", default_support_days * steps_per_day)

    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover CSV files
    if args.csv_path:
        csv_files = [p.strip() for p in args.csv_path.split(",") if p.strip()]
    else:
        csv_files = sorted(glob.glob(str(Path(raw_dir) / "*.csv")))

    csv_files.sort(key=lambda p: 0 if "dataset_prepro" in Path(p).name.lower() else 1)

    scaler = None
    tsls_cap = DEFAULT_TSLS_CAP

    for idx, f in enumerate(csv_files):
        csv_name = Path(f).name
        is_primary = (idx == 0)

        # Check for dataset-specific support/k_shot configuration
        if is_primary:
            support_days = default_support_days
            support_steps = default_k_shot
        else:
            # Apply uniform global test_cases configuration to all test cases
            support_days = test_cases_cfg.get("support_days", default_support_days)
            support_steps = test_cases_cfg.get("k_shot", support_days * steps_per_day)

        res = process_csv(
            csv_path=f,
            output_dir=out_dir,
            support_steps=support_steps,
            support_days=support_days,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
            scaler=scaler,
            tsls_cap=tsls_cap,
            is_primary=is_primary
        )
        if scaler is None:
            scaler = res["scaler"]
            tsls_cap = res["tsls_cap"]

    print(f"\n{'='*65}")
    print(f"All files successfully preprocessed and saved to: {out_dir.resolve()}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
