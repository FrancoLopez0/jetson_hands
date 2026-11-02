"""Validate the real hand model against a public MediaPipe test image."""
import json
from pathlib import Path
import urllib.request

import cv2
import numpy as np

from jetson_hands.config import Settings
from jetson_hands.recognition import MediaPipeHandsStrategy


def main():
    url = "https://storage.googleapis.com/mediapipe-assets/right_hands.jpg"
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Could not decode MediaPipe test image")
    strategy = MediaPipeHandsStrategy(Settings(token="unused-for-local-inference"))
    try:
        result = strategy.process(image, 1)
        assert result["hands"], "No hands detected in the reference image"
        assert all(len(hand["landmarks"]) == 21 for hand in result["hands"])
        output = Path("runtime")
        output.mkdir(exist_ok=True)
        cv2.imwrite(str(output / "reference-hands.jpg"), image)
        print(json.dumps({"reference_image": url, "hands_detected": len(result["hands"]),
                          "landmarks_per_hand": 21, "model_test": "passed"}, indent=2))
    finally:
        strategy.close()


if __name__ == "__main__":
    main()
