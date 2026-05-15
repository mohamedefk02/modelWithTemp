from __future__ import annotations

import argparse
import re
import time
from collections import deque
from typing import Any, Dict, Optional

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
    args = parser.parse_args()

    temp_window: deque[float] = deque(maxlen=max(5, args.baseline_samples))
    baseline_temp: Optional[float] = None

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

                temp_delta = parsed["skin_temperature"] - baseline_temp
                model_input = {
                    "bpm": round(parsed["bpm"], 3),
                    "skin_temperature": round(parsed["skin_temperature"], 3),
                    "temperature_delta": round(temp_delta, 3),
                }

                pred_resp = requests.post(args.predict_url, json=model_input, timeout=args.timeout)
                pred_resp.raise_for_status()
                pred = pred_resp.json()

                output = {
                    "timestamp": int(time.time()),
                    "sensor": {
                        "bpm": parsed["bpm"],
                        "skin_temperature": parsed["skin_temperature"],
                        "humidity": parsed["humidity"],
                        "ir": parsed["ir"],
                        "stress_device": parsed["stress_device"],
                        "temperature_delta": model_input["temperature_delta"],
                        "baseline_temperature": round(baseline_temp, 3),
                    },
                    "model_prediction": pred,
                }

                print(
                    "[bridge] "
                    f"BPM={parsed['bpm']:.2f} Temp={parsed['skin_temperature']:.2f} "
                    f"Delta={model_input['temperature_delta']:.2f} | "
                    f"Model label={pred.get('label')} risk={pred.get('risk_score')}"
                )

                if args.forward_url:
                    fw_resp = requests.post(args.forward_url, json=output, timeout=args.timeout)
                    fw_resp.raise_for_status()
                    print(f"[bridge] forwarded to {args.forward_url}")

            except requests.RequestException as exc:
                print(f"[bridge] http error: {exc}")
            except KeyboardInterrupt:
                print("\n[bridge] stopped by user")
                break
            except Exception as exc:
                print(f"[bridge] error: {exc}")


if __name__ == "__main__":
    main()

