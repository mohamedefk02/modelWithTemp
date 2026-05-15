from __future__ import annotations

import argparse
import re
import time
from collections import deque
from typing import Any, Dict, Optional, List

import requests
import serial


SERIAL_PATTERN = re.compile(
    r"BPM:\s*(?P<bpm>[0-9]+(?:\.[0-9]+)?)\s*\|\s*"
    r"BPM Moyen:\s*(?P<bpm_avg>[0-9]+(?:\.[0-9]+)?)\s*\|\s*"
    r"Temp:\s*(?P<temp>[0-9]+(?:\.[0-9]+)?)°C\s*\|\s*"
    r"Humidité:\s*(?P<humidity>[0-9]+(?:\.[0-9]+)?)%\s*\|\s*"
    r"IR:\s*(?P<ir>[0-9]+)\s*\|\s*Stress:\s*(?P<stress>[A-Z]+)"
)


def parse_sensor_line(line: str) -> Optional[Dict[str, Any]]:
    m = SERIAL_PATTERN.search(line)
    if not m:
        return None
    return {
        "bpm": float(m.group("bpm_avg")),  # use smoothed BPM
        "skin_temperature": float(m.group("temp")),
        "humidity": float(m.group("humidity")),
        "ir": int(m.group("ir")),
        "stress_device": m.group("stress"),
        "raw_line": line.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge ESP32 sensor serial output to model API /predict.")
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM5")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--predict-url", default="http://127.0.0.1:8000/predict", help="Model predict endpoint")
    parser.add_argument(
        "--forward-url",
        default=None,
        help="Optional second API endpoint to receive sensor+prediction payload",
    )
    parser.add_argument(
        "--baseline-samples",
        type=int,
        default=30,
        help="Number of initial temperature samples to build baseline",
    )
    parser.add_argument("--timeout", type=float, default=2.0, help="HTTP timeout seconds")
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=10.0,
        help="Aggregate samples for this many seconds, then predict once",
    )
    parser.add_argument("--min-bpm", type=float, default=45.0, help="Minimum plausible BPM to accept")
    parser.add_argument("--max-bpm", type=float, default=180.0, help="Maximum plausible BPM to accept")
    parser.add_argument("--min-ir", type=int, default=50000, help="Minimum IR signal to accept sample")
    parser.add_argument(
        "--min-valid-samples",
        type=int,
        default=20,
        help="Minimum number of valid samples required per window before predicting",
    )
    args = parser.parse_args()

    temp_window: deque[float] = deque(maxlen=max(5, args.baseline_samples))
    baseline_temp: Optional[float] = None
    sample_buffer: List[Dict[str, Any]] = []
    window_start_ts: Optional[float] = None

    print(f"[bridge] opening serial {args.port} @ {args.baud}")
    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        print("[bridge] started. waiting for sensor lines...")
        while True:
            try:
                raw = ser.readline().decode("utf-8", errors="ignore").strip()
                if not raw:
                    continue

                parsed = parse_sensor_line(raw)
                if not parsed:
                    continue

                # Filter weak/noisy samples before baseline/prediction logic.
                if parsed["ir"] < args.min_ir:
                    continue
                if parsed["bpm"] < args.min_bpm or parsed["bpm"] > args.max_bpm:
                    continue

                temp_window.append(parsed["skin_temperature"])
                if baseline_temp is None and len(temp_window) >= args.baseline_samples:
                    baseline_temp = sum(temp_window) / len(temp_window)
                    print(f"[bridge] baseline temperature set: {baseline_temp:.3f} C")

                # Before baseline is ready, keep updating and skip prediction.
                if baseline_temp is None:
                    print(
                        f"[bridge] calibrating baseline ({len(temp_window)}/{args.baseline_samples}) | "
                        f"BPM={parsed['bpm']:.2f} Temp={parsed['skin_temperature']:.2f}"
                    )
                    continue

                now = time.time()
                if window_start_ts is None:
                    window_start_ts = now
                sample_buffer.append(parsed)
                elapsed = now - window_start_ts
                if elapsed < args.window_seconds:
                    continue

                if len(sample_buffer) < args.min_valid_samples:
                    print(
                        f"[bridge] skipped window: only {len(sample_buffer)} valid samples "
                        f"(min required {args.min_valid_samples})"
                    )
                    sample_buffer = []
                    window_start_ts = None
                    continue

                # Aggregate window then predict once.
                bpm_avg = sum(x["bpm"] for x in sample_buffer) / len(sample_buffer)
                skin_temp_avg = sum(x["skin_temperature"] for x in sample_buffer) / len(sample_buffer)
                humidity_avg = sum(x["humidity"] for x in sample_buffer) / len(sample_buffer)
                ir_avg = int(sum(x["ir"] for x in sample_buffer) / len(sample_buffer))
                temp_delta = skin_temp_avg - baseline_temp
                model_input = {
                    "bpm": round(bpm_avg, 3),
                    "skin_temperature": round(skin_temp_avg, 3),
                    "temperature_delta": round(temp_delta, 3),
                }

                pred_resp = requests.post(args.predict_url, json=model_input, timeout=args.timeout)
                pred_resp.raise_for_status()
                pred = pred_resp.json()

                output = {
                    "timestamp": int(time.time()),
                    "sensor": {
                        "bpm": round(bpm_avg, 3),
                        "skin_temperature": round(skin_temp_avg, 3),
                        "humidity": round(humidity_avg, 3),
                        "ir": ir_avg,
                        "stress_device": sample_buffer[-1]["stress_device"],
                        "temperature_delta": model_input["temperature_delta"],
                        "baseline_temperature": round(baseline_temp, 3),
                        "window_seconds": args.window_seconds,
                        "window_samples": len(sample_buffer),
                    },
                    "model_prediction": pred,
                }

                print(
                    "[bridge] "
                    f"BPM={bpm_avg:.2f} Temp={skin_temp_avg:.2f} "
                    f"Delta={model_input['temperature_delta']:.2f} | "
                    f"window={args.window_seconds:.1f}s samples={len(sample_buffer)} | "
                    f"Model label={pred.get('label')} risk={pred.get('risk_score')}"
                )

                if args.forward_url:
                    fw_resp = requests.post(args.forward_url, json=output, timeout=args.timeout)
                    fw_resp.raise_for_status()
                    print(f"[bridge] forwarded to {args.forward_url}")

                # Reset window
                sample_buffer = []
                window_start_ts = None

            except requests.RequestException as exc:
                print(f"[bridge] http error: {exc}")
            except KeyboardInterrupt:
                print("\n[bridge] stopped by user")
                break
            except Exception as exc:
                print(f"[bridge] error: {exc}")


if __name__ == "__main__":
    main()
