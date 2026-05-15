from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np


HEADER = ["bpm", "rmssd", "sdnn", "skin_temperature", "temperature_delta", "label"]


def _clip(v: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, v)))


def generate_row(rng: np.random.Generator, label: int) -> list[float | int]:
    # WESAD-like physiological priors.
    if label == 0:
        # Non-stress: baseline/amusement/meditation-like
        bpm = _clip(rng.normal(73, 8), 50, 105)
        rmssd = _clip(rng.normal(44, 14), 8, 140)
        sdnn = _clip(rng.normal(58, 16), 10, 160)
        baseline_temp = rng.normal(36.35, 0.23)
        temp_delta = _clip(rng.normal(-0.05, 0.22), -1.2, 0.9)
    else:
        # Stress-like
        bpm = _clip(rng.normal(98, 11), 70, 145)
        rmssd = _clip(rng.normal(21, 8), 3, 95)
        sdnn = _clip(rng.normal(30, 10), 4, 120)
        baseline_temp = rng.normal(36.30, 0.23)
        temp_delta = _clip(rng.normal(-0.62, 0.32), -2.3, 0.8)

    # Correlate HRV inversely with elevated BPM.
    bpm_z = (bpm - 80.0) / 20.0
    rmssd = _clip(rmssd - 2.6 * bpm_z + rng.normal(0, 1.1), 1.0, 180.0)
    sdnn = _clip(sdnn - 2.2 * bpm_z + rng.normal(0, 1.3), 1.0, 200.0)

    skin_temp = _clip(baseline_temp + temp_delta + rng.normal(0, 0.05), 33.5, 38.2)

    return [
        round(bpm, 3),
        round(rmssd, 3),
        round(sdnn, 3),
        round(skin_temp, 3),
        round(temp_delta, 3),
        int(label),
    ]


def generate_csv_until_size(
    out_path: Path,
    target_mb: int,
    seed: int = 42,
    stress_ratio: float = 0.45,
    flush_every: int = 25000,
) -> tuple[int, float]:
    rng = np.random.default_rng(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    target_bytes = int(target_mb * 1024 * 1024)
    n_rows = 0

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

        while True:
            label = 1 if rng.random() < stress_ratio else 0
            writer.writerow(generate_row(rng, label))
            n_rows += 1

            if n_rows % flush_every == 0:
                f.flush()
                size = out_path.stat().st_size
                if size >= target_bytes:
                    break

        f.flush()

    final_mb = out_path.stat().st_size / (1024 * 1024)
    return n_rows, final_mb


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WESAD-like synthetic stress dataset CSV.")
    parser.add_argument("--output", default="data/processed/wesad_like_train.csv")
    parser.add_argument("--target-mb", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stress-ratio", type=float, default=0.45)
    args = parser.parse_args()

    out = Path(args.output)
    rows, size_mb = generate_csv_until_size(
        out_path=out,
        target_mb=args.target_mb,
        seed=args.seed,
        stress_ratio=args.stress_ratio,
    )

    print(f"Generated: {out}")
    print(f"Rows: {rows}")
    print(f"Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
