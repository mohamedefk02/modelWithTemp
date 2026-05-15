from __future__ import annotations

import argparse
import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


# WESAD standard labels.
BASELINE_LABEL = 1
STRESS_LABEL = 2
AMUSEMENT_LABEL = 3
MEDITATION_LABEL = 4


@dataclass
class WindowFeatures:
    subject_id: str
    bpm: float
    rmssd: float
    sdnn: float
    skin_temperature: float
    temperature_delta: float
    label: int


def load_wesad_subject(subject_pkl: Path) -> Dict:
    with subject_pkl.open("rb") as f:
        # WESAD pickles are typically python2-origin; latin1 handles compatibility.
        return pickle.load(f, encoding="latin1")


def _to_1d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr.reshape(-1)


def detrend(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x)


def normalize(x: np.ndarray) -> np.ndarray:
    std = np.std(x)
    if std < 1e-8:
        return np.zeros_like(x)
    return (x - np.mean(x)) / std


def detect_bvp_peaks(signal: np.ndarray, fs: float, min_hr: float = 40, max_hr: float = 200) -> np.ndarray:
    """
    Lightweight local-maxima detector with refractory period.
    Works reasonably for WESAD wrist BVP after normalization.
    """
    x = normalize(detrend(signal))

    # Local maxima above a mild adaptive threshold.
    thr = np.percentile(x, 65)
    candidates = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] > thr))[0] + 1
    if candidates.size == 0:
        return candidates

    min_distance = int(fs * 60.0 / max_hr)
    max_distance = int(fs * 60.0 / min_hr)

    selected = [int(candidates[0])]
    for idx in candidates[1:]:
        if idx - selected[-1] >= min_distance:
            selected.append(int(idx))

    peaks = np.array(selected, dtype=int)

    # Remove implausible gaps by filtering RR intervals.
    if peaks.size >= 3:
        rr = np.diff(peaks) / fs
        valid = (rr >= (60.0 / max_hr)) & (rr <= (60.0 / min_hr))
        keep = np.ones(peaks.shape[0], dtype=bool)
        keep[1:] = valid
        peaks = peaks[keep]

    # Final safety guard if over-sparse.
    if peaks.size >= 2 and np.max(np.diff(peaks)) > max_distance * 3:
        return np.array([], dtype=int)

    return peaks


def compute_hrv_from_peaks(peaks: np.ndarray, fs: float) -> Tuple[float, float, float]:
    """
    Returns bpm, rmssd(ms), sdnn(ms).
    """
    if peaks.size < 3:
        raise ValueError("Not enough peaks for HRV")

    rr_s = np.diff(peaks) / fs
    rr_ms = rr_s * 1000.0

    bpm = 60.0 / float(np.mean(rr_s))
    diffs = np.diff(rr_ms)
    if diffs.size == 0:
        raise ValueError("Not enough RR differences for RMSSD")

    rmssd = float(np.sqrt(np.mean(np.square(diffs))))
    sdnn = float(np.std(rr_ms, ddof=1)) if rr_ms.size > 1 else 0.0
    return float(bpm), rmssd, sdnn


def aligned_slice(arr: np.ndarray, src_len: int, start: int, end: int) -> np.ndarray:
    """
    Map [start, end) indices from a reference timeline (length=src_len)
    into arr timeline by proportional indexing.
    """
    if end <= start:
        return np.array([], dtype=float)
    a = int(np.floor(start * len(arr) / src_len))
    b = int(np.floor(end * len(arr) / src_len))
    a = max(0, min(a, len(arr) - 1))
    b = max(a + 1, min(b, len(arr)))
    return arr[a:b]


def make_subject_rows(
    subject_data: Dict,
    subject_id: str,
    window_sec: int,
    step_sec: int,
    bvp_fs: float = 64.0,
) -> List[WindowFeatures]:
    label = _to_1d(np.asarray(subject_data["label"]))
    wrist = subject_data["signal"]["wrist"]

    bvp = _to_1d(np.asarray(wrist["BVP"]))
    temp = _to_1d(np.asarray(wrist["TEMP"]))

    total_samples = len(bvp)
    win = int(window_sec * bvp_fs)
    step = int(step_sec * bvp_fs)

    # Subject-specific baseline temp from baseline label periods.
    baseline_mask = label == BASELINE_LABEL
    if np.any(baseline_mask):
        baseline_temp_segments = temp[baseline_mask[: len(temp)]] if len(label) >= len(temp) else temp
        baseline_temp = float(np.nanmedian(baseline_temp_segments))
    else:
        baseline_temp = float(np.nanmedian(temp))

    rows: List[WindowFeatures] = []
    for start in range(0, max(1, total_samples - win + 1), step):
        end = start + win
        if end > total_samples:
            break

        bvp_win = bvp[start:end]
        label_win = aligned_slice(label, total_samples, start, end)
        temp_win = aligned_slice(temp, total_samples, start, end)

        if label_win.size == 0 or temp_win.size == 0:
            continue

        # Majority raw label in this window.
        raw_lbl = int(np.round(np.median(label_win)))

        # Keep only the canonical affective states.
        if raw_lbl not in {BASELINE_LABEL, STRESS_LABEL, AMUSEMENT_LABEL, MEDITATION_LABEL}:
            continue

        # Binary mapping requested by your API schema.
        out_label = 1 if raw_lbl == STRESS_LABEL else 0

        try:
            peaks = detect_bvp_peaks(bvp_win, fs=bvp_fs)
            bpm, rmssd, sdnn = compute_hrv_from_peaks(peaks, fs=bvp_fs)
        except Exception:
            continue

        skin_temp = float(np.nanmean(temp_win))
        temp_delta = skin_temp - baseline_temp

        # Physiological plausibility guardrails.
        if not (40 <= bpm <= 200):
            continue
        if not (0 <= rmssd <= 300 and 0 <= sdnn <= 300):
            continue

        rows.append(
            WindowFeatures(
                subject_id=subject_id,
                bpm=round(bpm, 3),
                rmssd=round(rmssd, 3),
                sdnn=round(sdnn, 3),
                skin_temperature=round(skin_temp, 3),
                temperature_delta=round(temp_delta, 3),
                label=out_label,
            )
        )

    return rows


def discover_subject_pkls(wesad_root: Path) -> List[Path]:
    # Typical WESAD layout: WESAD/S2/S2.pkl ...
    pkls = sorted(wesad_root.glob("S*/S*.pkl"))
    if pkls:
        return pkls

    # Fallback recursive search.
    return sorted(wesad_root.rglob("S*.pkl"))


def write_csv(rows: Iterable[WindowFeatures], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "bpm", "rmssd", "sdnn", "skin_temperature", "temperature_delta", "label"])
        for r in rows:
            writer.writerow(
                [r.subject_id, r.bpm, r.rmssd, r.sdnn, r.skin_temperature, r.temperature_delta, r.label]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare WESAD features into API training CSV format.")
    parser.add_argument("--wesad-root", required=True, help="Path to extracted WESAD root directory")
    parser.add_argument("--output", default="data/wesad_features.csv", help="Output CSV path")
    parser.add_argument("--window-sec", type=int, default=60, help="Window length in seconds")
    parser.add_argument("--step-sec", type=int, default=10, help="Window step in seconds")
    args = parser.parse_args()

    wesad_root = Path(args.wesad_root)
    if not wesad_root.exists():
        raise FileNotFoundError(f"WESAD root not found: {wesad_root}")

    subject_pkls = discover_subject_pkls(wesad_root)
    if not subject_pkls:
        raise FileNotFoundError("No subject .pkl files found under WESAD root")

    all_rows: List[WindowFeatures] = []
    for pkl_path in subject_pkls:
        try:
            data = load_wesad_subject(pkl_path)
            subject_id = pkl_path.stem
            rows = make_subject_rows(
                data,
                subject_id=subject_id,
                window_sec=args.window_sec,
                step_sec=args.step_sec,
            )
            all_rows.extend(rows)
            print(f"{pkl_path.name}: generated {len(rows)} rows")
        except Exception as exc:
            print(f"Skipping {pkl_path}: {exc}")

    if not all_rows:
        raise RuntimeError("No rows generated. Check WESAD path and preprocessing settings.")

    output_path = Path(args.output)
    write_csv(all_rows, output_path)

    labels = np.array([r.label for r in all_rows], dtype=int)
    n0 = int(np.sum(labels == 0))
    n1 = int(np.sum(labels == 1))

    print(f"Saved {len(all_rows)} rows to {output_path}")
    print(f"Class distribution -> normal(0): {n0}, suspicious(1): {n1}")


if __name__ == "__main__":
    main()
