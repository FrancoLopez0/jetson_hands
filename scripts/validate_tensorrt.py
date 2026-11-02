"""Hardware validation: FP16 numerics, rotated ROI tracking, loss/reacquisition."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import urllib.request

import cv2
import numpy as np

from jetson_hands.config import Settings
from jetson_hands.recognition import create_strategy


def main(args):
    cv2.setNumThreads(2)
    output = Path("runtime")
    output.mkdir(exist_ok=True)
    path = output / "right_hands.jpg"
    if not path.exists():
        with urllib.request.urlopen("https://storage.googleapis.com/mediapipe-assets/right_hands.jpg", timeout=30) as response:
            path.write_bytes(response.read())
    image = cv2.imread(str(path))
    # Isolate the upright hand so both single-target strategies use the same hand.
    image = image[:, image.shape[1] // 2:].copy()
    settings = replace(Settings.from_env(), strategy=args.strategy, max_hands=1)
    strategy = create_strategy(settings)
    try:
        first_frame = image.copy()
        result = strategy.process(first_frame, 1)
        cv2.imwrite(str(output / f"{args.strategy}-acquisition.jpg"), first_frame)
        assert len(result["hands"]) == 1, result
        assert len(result["hands"][0]["landmarks"]) == 21
        reference = np.array([[p["x"], p["y"]] for p in result["hands"][0]["landmarks"]])
        report = {"strategy": args.strategy, "reference_hands": 1}
        if args.strategy == "tensorrt":
            # Compare the SAME crop against FP32 OpenCV DNN before trusting FP16.
            crop = cv2.warpAffine(image, strategy.roi.matrix(), (224, 224))
            blob = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)[None] / 255
            trt = {name: value.copy() for name, value in strategy.hand.infer(blob).items()}
            net = cv2.dnn.readNet("models/tensorrt/hand.onnx")
            net.setInput(blob)
            names = net.getUnconnectedOutLayersNames()
            cpu = dict(zip(names, net.forward(names)))
            differences = {name: float(np.max(np.abs(cpu[name] - trt[name]))) for name in trt}
            assert differences["Identity"] < 3.0, differences
            assert differences["Identity_1"] < 0.05, differences
            report["fp16_vs_fp32_max_abs_error"] = differences
        present, total, errors = 0, 0, []
        h, w = image.shape[:2]
        # Small continuous translation/rotation exercises tracking coordinates.
        for index in range(100):
            angle = 15 * np.sin(index / 20)
            transform = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            transform[:, 2] += [8 * np.sin(index / 10), 5 * np.cos(index / 10)]
            frame = cv2.warpAffine(image, transform, (w, h))
            result = strategy.process(frame, 2 + index * 33)
            total += 1
            if result["hands"]:
                present += 1
                actual = np.array([[p["x"], p["y"]] for p in result["hands"][0]["landmarks"]])
                expected = cv2.transform((reference * [w, h]).astype(np.float32)[None], transform)[0] / [w, h]
                errors.append(float(np.median(np.linalg.norm(actual - expected, axis=1))))
            if index == 50:
                cv2.imwrite(str(output / f"{args.strategy}-reference.jpg"), frame)
        assert present >= 90, ("Tracking lost", present, total)
        assert float(np.median(errors)) < 0.04, errors
        blank = np.zeros_like(image)
        for index in range(3):
            result = strategy.process(blank.copy(), 4000 + index * 33)
        assert not result["hands"], "False positive on a blank frame"
        result = strategy.process(image.copy(), 4200)
        assert len(result["hands"]) == 1, "Failed to reacquire the hand"
        warm_until = time.perf_counter() + 2
        timestamp = 5000
        while time.perf_counter() < warm_until:
            timestamp += 33
            strategy.process(image.copy(), timestamp)
        times = []
        started = time.perf_counter()
        while time.perf_counter() - started < args.seconds:
            timestamp += 33
            frame = image.copy()
            before = time.perf_counter()
            result = strategy.process(frame, timestamp)
            times.append((time.perf_counter() - before) * 1000)
        report.update({"tracked_frames": present, "transformed_frames": total,
                       "median_normalized_tracking_error": round(float(np.median(errors)), 4),
                       "blank_frame": "passed", "reacquisition": "passed",
                       "reference_inference_fps": round(len(times) / (time.perf_counter() - started), 2),
                       "reference_inference_ms": {"median": round(float(np.median(times)), 2),
                                                  "p95": round(float(np.percentile(times, 95)), 2)}})
        (output / f"{args.strategy}-validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        strategy.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["mediapipe", "tensorrt"], default="tensorrt")
    parser.add_argument("--seconds", type=float, default=30)
    main(parser.parse_args())
