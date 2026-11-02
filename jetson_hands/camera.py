"""Linux V4L2 MMAP capture. No network or OpenCV VideoCapture intermediary."""
import ctypes as c
import errno
import fcntl
import mmap
import os
import select
import struct
import subprocess
import threading
import time
from dataclasses import dataclass

import numpy as np


class Timeval(c.Structure):
    _fields_ = [("seconds", c.c_long), ("microseconds", c.c_long)]


class Timecode(c.Structure):
    _fields_ = [("type", c.c_uint32), ("flags", c.c_uint32),
                ("frames", c.c_uint8), ("seconds", c.c_uint8),
                ("minutes", c.c_uint8), ("hours", c.c_uint8), ("userbits", c.c_uint8 * 4)]


class BufferMemory(c.Union):
    _fields_ = [("offset", c.c_uint32), ("userptr", c.c_ulong), ("fd", c.c_int32)]


class Buffer(c.Structure):
    _fields_ = [("index", c.c_uint32), ("type", c.c_uint32), ("bytesused", c.c_uint32),
                ("flags", c.c_uint32), ("field", c.c_uint32), ("timestamp", Timeval),
                ("timecode", Timecode), ("sequence", c.c_uint32), ("memory", c.c_uint32),
                ("m", BufferMemory), ("length", c.c_uint32), ("reserved2", c.c_uint32),
                ("request_fd", c.c_int32)]


def iowr(number, size):
    return (3 << 30) | (size << 16) | (ord("V") << 8) | number


@dataclass(frozen=True)
class RawFrame:
    pixels: np.ndarray
    sequence: int
    captured_at: float
    timestamp_source: str
    driver_timestamp: float


class V4L2Camera:
    def __init__(self, settings):
        self.settings = settings
        self.fd = None
        self.buffers = []
        self.streaming = False
        self.stopping = threading.Event()
        self.condition = threading.Condition()
        self.latest = None
        self.error = None
        self.thread = None

    def start(self):
        s = self.settings
        # The controls are specific to the installed NVIDIA OV5647 driver.
        subprocess.run([
            "v4l2-ctl", "-d", s.device,
            f"--set-ctrl=sensor_mode={s.sensor_mode},override_enable=1,"
            f"frame_rate={s.fps * 1000000},exposure={s.exposure},gain={s.gain}",
        ], check=True, timeout=5)
        try:
            self.fd = os.open(s.device, os.O_RDWR | os.O_NONBLOCK)
            # v4l2_format has an 8-byte-aligned, 200-byte union on aarch64.
            if c.sizeof(c.c_void_p) != 8 or c.sizeof(Buffer) != 88:
                raise RuntimeError("V4L2 capture requires the Linux 64-bit ABI")
            fmt = bytearray(208)
            struct.pack_into("I", fmt, 0, 1)  # VIDEO_CAPTURE
            fourcc = int.from_bytes(b"BG10", "little")
            stride = ((s.width * 2 + 63) // 64) * 64
            struct.pack_into("6I", fmt, 8, s.width, s.height, fourcc, 1, stride, stride * s.height)
            fcntl.ioctl(self.fd, iowr(5, len(fmt)), fmt)  # S_FMT
            width, height, pixel, _, self.stride, size = struct.unpack_from("6I", fmt, 8)
            if (width, height, pixel) != (s.width, s.height, fourcc):
                raise RuntimeError("Driver negotiated an unexpected Bayer format")
            if self.stride < s.width * 2 or size < self.stride * s.height:
                raise RuntimeError("Driver returned an invalid frame layout")
            request = bytearray(struct.pack("5I", 4, 1, 1, 0, 0))
            fcntl.ioctl(self.fd, iowr(8, 20), request)  # REQBUFS, MMAP
            count = struct.unpack_from("I", request)[0]
            if count < 2:
                raise RuntimeError("Driver did not provide enough capture buffers")
            for index in range(count):
                buffer = Buffer(index=index, type=1, memory=1)
                fcntl.ioctl(self.fd, iowr(9, c.sizeof(Buffer)), buffer)  # QUERYBUF
                self.buffers.append(mmap.mmap(self.fd, buffer.length, mmap.MAP_SHARED,
                                             mmap.PROT_READ | mmap.PROT_WRITE, offset=buffer.m.offset))
                fcntl.ioctl(self.fd, iowr(15, c.sizeof(Buffer)), buffer)  # QBUF
            fcntl.ioctl(self.fd, 0x40045612, struct.pack("I", 1))  # STREAMON
            self.streaming = True
            self.thread = threading.Thread(target=self._capture, name="v4l2-capture", daemon=True)
            self.thread.start()
        except BaseException:
            self.close()
            raise

    def _capture(self):
        last_frame = time.monotonic()
        try:
            while not self.stopping.is_set():
                if not select.select([self.fd], [], [], 0.5)[0]:
                    if time.monotonic() - last_frame > 5:
                        raise TimeoutError("No frames received from the camera for 5 seconds")
                    continue
                # Drain queued driver buffers; only publish the most recent frame.
                newest = None
                for _ in self.buffers:
                    buffer = Buffer(type=1, memory=1)
                    try:
                        fcntl.ioctl(self.fd, iowr(17, c.sizeof(Buffer)), buffer)  # DQBUF
                    except OSError as error:
                        if error.errno == errno.EAGAIN:
                            break
                        raise
                    try:
                        if buffer.flags & 0x40:  # V4L2_BUF_FLAG_ERROR
                            continue
                        needed = self.stride * self.settings.height
                        if buffer.bytesused < needed:
                            raise RuntimeError("Driver returned an incomplete RAW frame")
                        raw = np.ndarray((self.settings.height, self.stride // 2), dtype="<u2",
                                         buffer=self.buffers[buffer.index])[:, :self.settings.width].copy()
                        # Tegra VI can label its hardware timestamp MONOTONIC while
                        # using a different clock epoch. Never subtract it from the
                        # host clock: measure from dequeue and expose the raw value.
                        timestamp = buffer.timestamp.seconds + buffer.timestamp.microseconds / 1e6
                        newest = RawFrame(raw, buffer.sequence, time.monotonic(),
                                          "dequeue-monotonic", timestamp)
                    finally:
                        fcntl.ioctl(self.fd, iowr(15, c.sizeof(Buffer)), buffer)
                if newest is not None:
                    last_frame = time.monotonic()
                    with self.condition:
                        self.latest = newest
                        self.condition.notify_all()
        except Exception as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()

    def read_latest(self, previous):
        with self.condition:
            self.condition.wait_for(lambda: self.stopping.is_set() or self.error or
                                    (self.latest is not None and self.latest.sequence != previous), 1)
            if self.error:
                raise self.error
            if self.stopping.is_set() or self.latest is None or self.latest.sequence == previous:
                return None
            return self.latest

    def close(self):
        self.stopping.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread:
            self.thread.join(timeout=2)
        if self.fd is not None:
            try:
                if self.streaming:
                    fcntl.ioctl(self.fd, 0x40045613, struct.pack("I", 1))  # STREAMOFF
            finally:
                for buffer in self.buffers:
                    buffer.close()
                self.buffers.clear()
                os.close(self.fd)
                self.fd = None
