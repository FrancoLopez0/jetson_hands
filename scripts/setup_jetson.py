"""Run with uv on the Jetson after syncing dependencies."""
import hashlib
import io
from pathlib import Path
import secrets
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


def main():
    token = ROOT / "access.token"
    if not token.exists():
        with token.open("x") as stream:
            stream.write(secrets.token_urlsafe(32) + "\n")
    token.chmod(0o600)
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    destination = models / "hand_landmarker.task"
    if not destination.exists():
        temporary = destination.with_suffix(".download")
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            data = response.read()
        if len(data) < 1_000_000 or not zipfile.is_zipfile(io.BytesIO(data)):
            raise RuntimeError("Invalid hand landmarker model download")
        if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
            raise RuntimeError("Hand landmarker model checksum mismatch")
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if not {"hand_detector.tflite", "hand_landmarks_detector.tflite"} <= set(archive.namelist()):
                raise RuntimeError("Hand landmarker archive is missing its models")
        temporary.write_bytes(data)
        temporary.replace(destination)
    actual = hashlib.sha256(destination.read_bytes()).hexdigest()
    if actual != MODEL_SHA256:
        raise RuntimeError("Installed hand landmarker model checksum mismatch")
    print("Model SHA256:", actual)
    print("Token file:", token)


if __name__ == "__main__":
    main()
