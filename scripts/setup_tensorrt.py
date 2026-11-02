"""Download checksum-pinned OpenCV Zoo models and build local FP16 engines."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
MODELS = {
    "palm": ("palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx",
             "78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c"),
    "hand": ("handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx",
             "db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c"),
}


def main(args):
    directory = ROOT / "models" / "tensorrt"
    directory.mkdir(parents=True, exist_ok=True)
    license_path = directory / "LICENSE"
    if not license_path.exists():
        url = f"https://raw.githubusercontent.com/opencv/opencv_zoo/{COMMIT}/models/palm_detection_mediapipe/LICENSE"
        with urllib.request.urlopen(url, timeout=30) as response:
            license_path.write_bytes(response.read())
    subprocess.run(["cmake", "-S", str(ROOT / "native"), "-B", str(ROOT / "native/build"),
                    "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(ROOT / "native/build"), "-j2"], check=True)
    manifest = {"source_commit": COMMIT, "precision": "FP16 internal / FP32 IO", "models": {}}
    for name, (path, digest) in MODELS.items():
        destination = directory / f"{name}.onnx"
        if not destination.exists():
            url = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{COMMIT}/models/{path}"
            with urllib.request.urlopen(url, timeout=90) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError(f"Checksum mismatch for {name}")
            temporary = destination.with_suffix(".download")
            temporary.write_bytes(data)
            temporary.replace(destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Installed ONNX checksum mismatch for {name}")
        engine = directory / f"{name}.engine"
        if args.rebuild or not engine.exists():
            temporary = engine.with_suffix(".building")
            command = ["/usr/src/tensorrt/bin/trtexec", f"--onnx={destination}",
                       f"--saveEngine={temporary}", "--fp16", "--skipInference",
                       "--memPoolSize=workspace:512", "--builderOptimizationLevel=3"]
            with (directory / f"{name}-build.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Engine build failed: {directory / (name + '-build.log')}")
            temporary.replace(engine)
        manifest["models"][name] = {"onnx_sha256": digest, "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest()}
        print(f"Ready: {engine}", flush=True)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    main(parser.parse_args())
