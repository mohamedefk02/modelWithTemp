from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def fill_temperature_delta(path: Path) -> None:
    df = pd.read_csv(path)

    if "skin_temperature" not in df.columns:
        raise ValueError(f"{path}: missing skin_temperature column")

    if "label" not in df.columns:
        raise ValueError(f"{path}: missing label column")

    df["skin_temperature"] = pd.to_numeric(df["skin_temperature"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    normal_mask = df["label"] == 0
    if normal_mask.any():
        baseline = float(df.loc[normal_mask, "skin_temperature"].median())
    else:
        baseline = float(df["skin_temperature"].median())

    df["temperature_delta"] = df["skin_temperature"] - baseline
    df.to_csv(path, index=False)

    missing = int(df["temperature_delta"].isna().sum())
    print(f"{path}: baseline={baseline:.4f}, rows={len(df)}, missing_temperature_delta={missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute temperature_delta from skin_temperature in CSV files.")
    parser.add_argument("files", nargs="+", help="CSV paths to update in-place")
    args = parser.parse_args()

    for file_str in args.files:
        fill_temperature_delta(Path(file_str))


if __name__ == "__main__":
    main()
