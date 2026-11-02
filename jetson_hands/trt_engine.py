"""ctypes bridge to the system TensorRT runtime; works with Python 3.14."""
import ctypes as c
from pathlib import Path

import numpy as np


class TensorRTEngine:
    def __init__(self, engine_path, library_path):
        self.lib = c.CDLL(str(Path(library_path).resolve()))
        signatures = {
            "error": ([], c.c_char_p),
            "open": ([c.c_char_p], c.c_void_p),
            "close": ([c.c_void_p], None),
            "count": ([c.c_void_p], c.c_int),
            "name": ([c.c_void_p, c.c_int], c.c_char_p),
            "input": ([c.c_void_p, c.c_int], c.c_int),
            "shape": ([c.c_void_p, c.c_int, c.POINTER(c.c_int64)], c.c_int),
            "output": ([c.c_void_p, c.c_int], c.POINTER(c.c_float)),
            "run": ([c.c_void_p, c.POINTER(c.c_float), c.c_size_t], c.c_int),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.lib, "hands_trt_" + name)
            function.argtypes, function.restype = args, result
        self.handle = self.lib.hands_trt_open(str(Path(engine_path).resolve()).encode())
        if not self.handle:
            raise RuntimeError(self.lib.hands_trt_error().decode())
        self.outputs = {}
        try:
            for index in range(self.lib.hands_trt_count(self.handle)):
                dimensions = (c.c_int64 * 8)()
                rank = self.lib.hands_trt_shape(self.handle, index, dimensions)
                shape = tuple(dimensions[:rank])
                name = self.lib.hands_trt_name(self.handle, index).decode()
                if self.lib.hands_trt_input(self.handle, index):
                    self.input_shape = shape
                else:
                    pointer = self.lib.hands_trt_output(self.handle, index)
                    self.outputs[name] = np.ctypeslib.as_array(pointer, shape=(int(np.prod(shape)),)).reshape(shape)
        except BaseException:
            self.close()
            raise

    def infer(self, image):
        if not self.handle:
            raise RuntimeError("TensorRT engine is closed")
        image = np.ascontiguousarray(image, dtype=np.float32)
        if image.shape != self.input_shape:
            raise ValueError(f"Input shape {image.shape}, expected {self.input_shape}")
        if self.lib.hands_trt_run(self.handle, image.ctypes.data_as(c.POINTER(c.c_float)), image.size):
            raise RuntimeError(self.lib.hands_trt_error().decode())
        # These views are valid until next inference or close; caller must consume now.
        return self.outputs

    def close(self):
        if self.handle:
            self.outputs.clear()
            self.lib.hands_trt_close(self.handle)
            self.handle = None
