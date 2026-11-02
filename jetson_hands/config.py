import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    token: str
    device: str = "/dev/video0"
    sensor_mode: int = 4
    width: int = 1280
    height: int = 720
    fps: int = 30
    exposure: int = 16000
    gain: int = 339
    output_width: int = 640
    jpeg_quality: int = 80
    strategy: str = "mediapipe"
    model: str = "models/hand_landmarker.task"
    max_hands: int = 1
    trt_library: str = "native/build/libhands_trt.so"
    palm_engine: str = "models/tensorrt/palm.engine"
    hand_engine: str = "models/tensorrt/hand.engine"
    renderer: str = "standard"

    @classmethod
    def from_env(cls):
        token_file = Path(os.environ.get("HANDS_TOKEN_FILE", "access.token"))
        token = os.environ.get("HANDS_TOKEN") or token_file.read_text().strip()
        if len(token) < 32 or not token.isascii():
            raise ValueError("Access token must contain at least 32 ASCII characters")
        integer_fields = {"sensor_mode", "width", "height", "fps", "exposure", "gain",
                          "output_width", "jpeg_quality", "max_hands"}
        values = {}
        for name in cls.__dataclass_fields__:
            value = os.environ.get("HANDS_" + name.upper())
            if name != "token" and value is not None:
                values[name] = int(value) if name in integer_fields else value
        settings = cls(token=token, **values)
        modes = {0: (2592, 1944, 15), 1: (1920, 1080, 30), 2: (1296, 972, 30),
                 3: (640, 480, 90), 4: (1280, 720, 60)}
        mode = modes.get(settings.sensor_mode)
        if not mode or (settings.width, settings.height) != mode[:2]:
            raise ValueError("Sensor mode and capture dimensions do not match")
        if not 1 <= settings.fps <= mode[2]:
            raise ValueError("Invalid capture frame rate")
        if not 131 <= settings.exposure <= min(499500, 1000000 // settings.fps - 500):
            raise ValueError("Exposure must fit within the frame interval")
        if not 16 <= settings.gain <= 1023:
            raise ValueError("Gain must be between 16 and 1023")
        if not 160 <= settings.output_width <= settings.width or not 30 <= settings.jpeg_quality <= 95:
            raise ValueError("Invalid output width or JPEG quality")
        if settings.max_hands not in (1, 2):
            raise ValueError("max_hands must be 1 or 2")
        if settings.strategy == "tensorrt" and settings.max_hands != 1:
            raise ValueError("The TensorRT strategy supports one hand")
        if settings.renderer not in ("standard", "lut"):
            raise ValueError("renderer must be standard or lut")
        return settings
