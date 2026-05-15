from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create subject-level train/test split with minimum positives in test."
    )
    parser.add_argument("--input", default="data/wesad_features.csv", help="Input CSV path")
    parser.add_argument("--train-output", default="data/train.csv", help="Train CSV output path")
    parser.add_argument("--test-output", default="data/test.csv", help="Test CSV output path")
    parser.add_argument("--subject-column", default="subject_id", help="Subject ID column name")
    parser.add_argument("--label-column", default="label", help="Binary label column name")
    parser.add_argument("--test-subject-ratio", type=float, default=0.25, help="Initial test subject ratio")
    parser.add_argument("--min-test-positives", type=int, default=10, help="Minimum positive rows required in test")
    parser.add_argument("--min-train-positives", type=int, default=10, help="Minimum positive rows required in train")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_csv(input_path)
    if args.subject_column not in df.columns:
        raise ValueError(f"Missing subject column: {args.subject_column}")
    if args.label_column not in df.columns:
        raise ValueError(f"Missing label column: {args.label_column}")

    df = df.dropna(subset=[args.subject_column, args.label_column]).copy()
    df[args.subject_column] = df[args.subject_column].astype(str)
    df[args.label_column] = pd.to_numeric(df[args.label_column], errors="coerce")
    df = df.dropna(subset=[args.label_column]).copy()
    df[args.label_column] = df[args.label_column].astype(int)

    subjects = sorted(df[args.subject_column].unique().tolist())
    if len(subjects) < 2:
        raise ValueError("Need at least 2 subjects to split train/test.")

    pos_counts = (
        df[df[args.label_column] == 1].groupby(args.subject_column)[args.label_column].size().to_dict()
    )
    for s in subjects:
        pos_counts.setdefault(s, 0)

    # Pick test subjects by highest positive contribution first, then add more subjects if needed.
    ranked = sorted(subjects, key=lambda s: (pos_counts[s], s), reverse=True)
    min_subjects = max(1, int(round(len(subjects) * args.test_subject_ratio)))
    test_subjects: list[str] = ranked[:min_subjects]

    def test_positive_count(chosen: list[str]) -> int:
        return int(df[(df[args.subject_column].isin(chosen)) & (df[args.label_column] == 1)].shape[0])

    total_positives = int((df[args.label_column] == 1).sum())
    idx = min_subjects
    while (
        test_positive_count(test_subjects) < args.min_test_positives
        and idx < len(ranked)
        and (total_positives - test_positive_count(test_subjects)) > args.min_train_positives
    ):
        test_subjects.append(ranked[idx])
        idx += 1

    chosen_test_positives = test_positive_count(test_subjects)
    chosen_train_positives = total_positives - chosen_test_positives
    if chosen_test_positives < args.min_test_positives:
        raise ValueError(
            f"Could not reach min-test-positives={args.min_test_positives}. "
            f"Maximum available with subject split is {test_positive_count(ranked)}."
        )
    if chosen_train_positives < args.min_train_positives:
        raise ValueError(
            f"Split violates min-train-positives={args.min_train_positives}. "
            f"Train positives would be {chosen_train_positives}."
        )

    test_set = set(test_subjects)
    train_df = df[~df[args.subject_column].isin(test_set)].copy()
    test_df = df[df[args.subject_column].isin(test_set)].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Split produced empty train or test set.")
    if train_df[args.label_column].nunique() < 2:
        raise ValueError("Train set has <2 classes.")
    if test_df[args.label_column].nunique() < 2:
        raise ValueError("Test set has <2 classes.")

    train_path = Path(args.train_output)
    test_path = Path(args.test_output)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    cols = df.columns.tolist()
    train_df[cols].to_csv(train_path, index=False)
    test_df[cols].to_csv(test_path, index=False)

    print(f"Saved train -> {train_path} rows={len(train_df)} subjects={train_df[args.subject_column].nunique()}")
    print(f"Saved test  -> {test_path} rows={len(test_df)} subjects={test_df[args.subject_column].nunique()}")
    print(f"Train labels: {train_df[args.label_column].value_counts().to_dict()}")
    print(f"Test labels:  {test_df[args.label_column].value_counts().to_dict()}")
    print(f"Test positives: {test_positive_count(test_subjects)}")


if __name__ == "__main__":
    main()
