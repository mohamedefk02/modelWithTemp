from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["bpm", "rmssd", "sdnn", "skin_temperature", "temperature_delta", "label"]


def clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def base_row(rng: np.random.Generator, label: int) -> list[float | int]:
    # Make classes intentionally overlap.
    if label == 0:
        bpm = clip(rng.normal(82, 10), 50, 135)
        rmssd = clip(rng.normal(31, 12), 2, 120)
        sdnn = clip(rng.normal(41, 12), 2, 140)
        temp_delta = clip(rng.normal(-0.25, 0.28), -1.8, 1.2)
        base_temp = rng.normal(36.25, 0.25)
    else:
        bpm = clip(rng.normal(90, 11), 55, 145)
        rmssd = clip(rng.normal(25, 11), 1, 110)
        sdnn = clip(rng.normal(33, 11), 1, 120)
        temp_delta = clip(rng.normal(-0.45, 0.32), -2.2, 1.0)
        base_temp = rng.normal(36.20, 0.27)

    # Correlations + extra jitter
    z = (bpm - 85.0) / 18.0
    rmssd = clip(rmssd - 1.5 * z + rng.normal(0, 2.2), 0.5, 170)
    sdnn = clip(sdnn - 1.3 * z + rng.normal(0, 2.5), 0.5, 190)
    skin_temp = clip(base_temp + temp_delta + rng.normal(0, 0.12), 33.0, 38.5)

    return [round(bpm, 3), round(rmssd, 3), round(sdnn, 3), round(skin_temp, 3), round(temp_delta, 3), int(label)]


def add_noise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()

    # 1) Sensor jitter on all numeric fields
    out["bpm"] += rng.normal(0, 2.0, len(out))
    out["rmssd"] += rng.normal(0, 2.5, len(out))
    out["sdnn"] += rng.normal(0, 2.8, len(out))
    out["skin_temperature"] += rng.normal(0, 0.08, len(out))
    out["temperature_delta"] += rng.normal(0, 0.06, len(out))

    # 2) Drift segments (simulate calibration drift)
    drift_idx = rng.choice(len(out), size=max(1, len(out) // 8), replace=False)
    out.loc[drift_idx, "skin_temperature"] += rng.normal(0.18, 0.09, len(drift_idx))
    out.loc[drift_idx, "temperature_delta"] += rng.normal(0.15, 0.07, len(drift_idx))

    # 3) Missing values (small %)
    miss_frac = 0.02
    for c in ["bpm", "rmssd", "sdnn", "skin_temperature", "temperature_delta"]:
        mask = rng.random(len(out)) < miss_frac
        out.loc[mask, c] = np.nan

    # 4) Outliers (rare)
    n_out = max(1, len(out) // 100)
    out_idx = rng.choice(len(out), size=n_out, replace=False)
    out.loc[out_idx, "bpm"] = rng.uniform(45, 165, size=n_out)
    out.loc[out_idx, "rmssd"] = rng.uniform(0, 160, size=n_out)
    out.loc[out_idx, "sdnn"] = rng.uniform(0, 180, size=n_out)

    # Keep values plausible where present.
    out["bpm"] = out["bpm"].clip(40, 180)
    out["rmssd"] = out["rmssd"].clip(0, 200)
    out["sdnn"] = out["sdnn"].clip(0, 220)
    out["skin_temperature"] = out["skin_temperature"].clip(32.5, 39.0)
    out["temperature_delta"] = out["temperature_delta"].clip(-3.0, 2.0)

    return out


def generate_tricky_test(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    labels = rng.integers(0, 2, size=rows)
    base = [base_row(rng, int(lbl)) for lbl in labels]
    df = pd.DataFrame(base, columns=COLS)
    df = add_noise(df, rng)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a tricky stress-test CSV for physiological risk model.")
    ap.add_argument("--rows", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--output", default="data/test_tricky.csv")
    args = ap.parse_args()

    df = generate_tricky_test(rows=args.rows, seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"saved: {out}")
    print(f"rows: {len(df)}")
    print("label_counts:")
    print(df["label"].value_counts().sort_index().to_dict())
    print("missing_counts:")
    print(df.isna().sum().to_dict())


if __name__ == "__main__":
    main()
