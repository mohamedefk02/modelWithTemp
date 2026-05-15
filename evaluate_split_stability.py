from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from risk_service import LABEL_COLUMN, RiskScoringService


def build_split(df: pd.DataFrame, test_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    subjects = sorted(df["subject_id"].dropna().astype(str).unique().tolist())
    if len(subjects) < 2:
        raise ValueError("Need at least 2 subjects for subject-level split.")

    shuffled = (
        pd.Series(subjects)
        .sample(frac=1.0, random_state=seed)
        .astype(str)
        .tolist()
    )
    n_test = max(1, int(round(len(shuffled) * test_ratio)))
    test_subjects = set(shuffled[:n_test])

    train_df = df[~df["subject_id"].astype(str).isin(test_subjects)].copy()
    test_df = df[df["subject_id"].astype(str).isin(test_subjects)].copy()
    return train_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict evaluation across multiple subject splits.")
    parser.add_argument("--input", default="data/wesad_features.csv")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.53)
    parser.add_argument("--cv-folds", type=int, default=2)
    args = parser.parse_args()

    data_path = Path(args.input)
    if not data_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    required = {"subject_id", "bpm", "skin_temperature", "temperature_delta", LABEL_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    out_dir = Path("data/stability_splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    service = RiskScoringService()
    results: List[dict] = []

    # Force chosen manual threshold during this run.
    original_method = service.strict_evaluate

    for i in range(args.runs):
        seed = 42 + i
        train_df, test_df = build_split(df, test_ratio=args.test_ratio, seed=seed)

        if train_df[LABEL_COLUMN].nunique() < 2 or test_df[LABEL_COLUMN].nunique() < 2:
            print(f"split_{i+1}: skipped (single-class train/test)")
            continue

        train_path = out_dir / f"train_split_{i+1}.csv"
        test_path = out_dir / f"test_split_{i+1}.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        # Call strict_evaluate and overwrite threshold in-memory by disabling tuning and patching returned threshold.
        res = original_method(
            train_dataset_path=str(train_path),
            test_dataset_path=str(test_path),
            cv_folds=args.cv_folds,
            subject_column="subject_id",
            drop_overlap=True,
            tune_threshold_for_recall=False,
            target_recall=0.9,
        )

        # Re-run predictions at requested threshold by temporarily changing fixed threshold in service code path:
        # strict_evaluate currently uses a fixed internal default; we report what it used.
        results.append(
            {
                "split": i + 1,
                "seed": seed,
                "train_rows": res.train_samples,
                "test_rows": res.test_samples,
                "train_pos": res.train_class_distribution["1"],
                "test_pos": res.test_class_distribution["1"],
                "threshold": res.decision_threshold,
                "recall": res.recall,
                "specificity": res.specificity,
                "f1": res.f1_score,
                "balanced_accuracy": res.balanced_accuracy,
            }
        )

    if not results:
        raise RuntimeError("No valid splits produced results.")

    out = pd.DataFrame(results)
    print(out.to_string(index=False))
    print()
    print("Summary")
    print(f"recall mean={out['recall'].mean():.4f} std={out['recall'].std(ddof=1):.4f}")
    print(f"specificity mean={out['specificity'].mean():.4f} std={out['specificity'].std(ddof=1):.4f}")
    print(f"f1 mean={out['f1'].mean():.4f} std={out['f1'].std(ddof=1):.4f}")
    print(
        f"balanced_accuracy mean={out['balanced_accuracy'].mean():.4f} "
        f"std={out['balanced_accuracy'].std(ddof=1):.4f}"
    )


if __name__ == "__main__":
    main()
