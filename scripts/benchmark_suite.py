"""Exclusive-camera benchmark suite. Restores the service even if a case fails."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main(args):
    root = Path(__file__).resolve().parents[1]
    output = root / "runtime"
    output.mkdir(exist_ok=True)
    subprocess.run(["sudo", "-n", "systemctl", "stop", "jetson-hands"], check=True)
    try:
        if not args.skip_validation:
            for strategy in ("mediapipe", "tensorrt"):
                subprocess.run([sys.executable, "scripts/validate_tensorrt.py", "--strategy", strategy,
                                "--seconds", str(args.seconds)], cwd=root, check=True)
        cases = [("mediapipe", 1280, 30), ("mediapipe", 640, 30),
                 ("tensorrt", 1280, 30), ("tensorrt", 640, 30), ("tensorrt", 640, 60)]
        reports = []
        for strategy, width, fps in cases:
            if args.tensorrt_only and strategy != "tensorrt":
                continue
            subprocess.run([sys.executable, "scripts/benchmark_capture.py", "--strategy", strategy,
                            "--width", str(width), "--fps", str(fps), "--seconds", str(args.seconds),
                            "--renderer", args.renderer, "--label", args.label],
                           cwd=root, check=True)
            height = 480 if width == 640 else 720
            suffix = "-" + args.label if args.label else ""
            reports.append(json.loads((output / f"{strategy}-{width}x{height}-{fps}{suffix}.json").read_text()))
        suffix = "-" + args.label if args.label else ""
        (output / f"comparison{suffix}.json").write_text(json.dumps(reports, indent=2) + "\n")
    finally:
        subprocess.run(["sudo", "-n", "systemctl", "start", "jetson-hands"], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--renderer", choices=["standard", "lut"], default="standard")
    parser.add_argument("--tensorrt-only", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--label", default="")
    main(parser.parse_args())
