"""Measure the real direct-driver pipeline without network/client confounders."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from jetson_hands.camera import V4L2Camera
from jetson_hands.config import Settings
from jetson_hands.pipeline import render_raw, render_raw_lut
from jetson_hands.recognition import create_strategy


def main(args):
    settings = replace(Settings.from_env(), strategy=args.strategy, max_hands=1,
                       sensor_mode=3 if args.width == 640 else 4,
                       width=args.width, height=480 if args.width == 640 else 720,
                       fps=args.fps, exposure=min(16000, 1000000 // args.fps - 600), renderer=args.renderer)
    output = Path("runtime")
    output.mkdir(exist_ok=True)
    name = f"{args.strategy}-{settings.width}x{settings.height}-{args.fps}"
    if args.label:
        name += "-" + args.label
    cv2.setNumThreads(2)
    renderer = render_raw_lut if args.renderer == "lut" else render_raw
    strategy = create_strategy(settings)
    camera = V4L2Camera(settings)
    metrics = {key: [] for key in ("convert_ms", "inference_ms", "encode_ms", "pipeline_ms", "hands")}
    stats_file = (output / f"{name}-tegrastats.log").open("w")
    stats = subprocess.Popen(["tegrastats", "--interval", "1000"], stdout=stats_file)
    try:
        camera.start()
        previous, timestamp, measured_at, first_sequence = -1, -1, None, None
        warm_until = time.monotonic() + args.warmup
        while measured_at is None or time.monotonic() - measured_at < args.seconds:
            raw = camera.read_latest(previous)
            if raw is None:
                continue
            previous = raw.sequence
            start = time.monotonic()
            bgr = renderer(raw.pixels, settings.output_width)
            converted = time.monotonic()
            timestamp = max(timestamp + 1, int(raw.captured_at * 1000))
            result = strategy.process(bgr, timestamp)
            inferred = time.monotonic()
            ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality])
            assert ok
            finished = time.monotonic()
            if start < warm_until:
                continue
            if measured_at is None:
                measured_at, first_sequence, first_capture = start, raw.sequence, raw.captured_at
            for key, value in zip(metrics, ((converted - start) * 1000, (inferred - converted) * 1000,
                                           (finished - inferred) * 1000, (finished - raw.captured_at) * 1000,
                                           len(result["hands"]))):
                metrics[key].append(value)
        elapsed = finished - measured_at
        report = {"strategy": args.strategy, "renderer": args.renderer, "capture": [settings.width, settings.height],
                  "output": list(bgr.shape[:2][::-1]), "requested_capture_fps": args.fps,
                  "actual_capture_fps": round((raw.sequence - first_sequence) / (raw.captured_at - first_capture), 2),
                  "frames": len(metrics["hands"]), "duration_s": round(elapsed, 2),
                  "processing_fps": round(len(metrics["hands"]) / elapsed, 2),
                  "frames_with_hand": sum(value > 0 for value in metrics["hands"]),
                  "max_hands": max(metrics["hands"]), "timestamp_source": raw.timestamp_source,
                  "timings": {key: {"median": round(float(np.median(values)), 2),
                                     "p95": round(float(np.percentile(values, 95)), 2)}
                              for key, values in metrics.items() if key != "hands"}}
        (output / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
        (output / f"{name}.jpg").write_bytes(jpeg.tobytes())
        print(json.dumps(report, indent=2), flush=True)
    finally:
        camera.close()
        strategy.close()
        stats.terminate()
        stats.wait(timeout=5)
        stats_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["mediapipe", "tensorrt", "passthrough"], required=True)
    parser.add_argument("--width", type=int, choices=[640, 1280], default=1280)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--warmup", type=float, default=3)
    parser.add_argument("--label", default="")
    parser.add_argument("--renderer", choices=["standard", "lut"], default="standard")
    main(parser.parse_args())
