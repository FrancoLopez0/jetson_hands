"""Low-rate board telemetry, separate from camera capture and inference."""
import asyncio
from contextlib import suppress
from pathlib import Path
import re
import time


GPU_CLOCK_PATH = Path("/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq")


def parse_tegrastats(line, gpu_clock_mhz=None):
    cpu_match = re.search(r"\bCPU\s+\[([^]]+)\]", line)
    if not cpu_match:
        raise ValueError("No CPU sample in tegrastats output")
    cores = []
    for index, value in enumerate(cpu_match[1].split(",")):
        match = re.fullmatch(r"\s*([\d.]+)%@([\d.]+)\s*", value)
        cores.append({"id": index, "online": match is not None,
                      "usage_pct": float(match[1]) if match else None,
                      "clock_mhz": float(match[2]) if match else None})
    online = [core for core in cores if core["online"]]
    gpu_match = re.search(r"\bGR3D_FREQ\s+([\d.]+)%(?:@(\[[\d,\s]+\]|[\d.]+))?", line)
    if gpu_clock_mhz is None and gpu_match and gpu_match[2]:
        clocks = [float(value) for value in gpu_match[2].strip("[]").split(",")]
        gpu_clock_mhz = sum(clocks) / len(clocks)
    rails = {}
    for name, value, unit, average, average_unit in re.findall(
        r"\b((?:VDD|POM)_[A-Z0-9_]+)\s+([\d.]+)(mW|W)/([\d.]+)(mW|W)", line
    ):
        rails[name] = float(value) / (1000 if unit == "mW" else 1)
    return {
        "cpu": {"usage_pct": round(sum(core["usage_pct"] for core in online) / len(online), 2) if online else None,
                "clock_mhz": round(sum(core["clock_mhz"] for core in online) / len(online), 2) if online else None,
                "cores": cores},
        "gpu": {"usage_pct": float(gpu_match[1]) if gpu_match else None, "clock_mhz": gpu_clock_mhz},
        "power": {"input_w": rails.get("VDD_IN", rails.get("POM_5V_IN")), "rails_w": rails},
    }


def read_gpu_clock():
    try:
        return float(GPU_CLOCK_PATH.read_text().strip()) / 1_000_000
    except (OSError, ValueError):
        return None


class TelemetryCollector:
    """One tegrastats child at 1 Hz, only the latest sample retained on the server."""
    def __init__(self):
        self.latest = None
        self.error = None
        self.task = None
        self.process = None
        self.sequence = 0

    async def start(self):
        self.task = asyncio.create_task(self._run(), name="jetson-telemetry")

    async def close(self):
        if self.task:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task

    async def _stop_process(self):
        process, self.process = self.process, None
        if process is not None and process.returncode is None:
            with suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 2)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    process.kill()
                await process.wait()

    async def _run(self):
        while True:
            try:
                self.process = await asyncio.create_subprocess_exec(
                    "tegrastats", "--interval", "1000", stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL, limit=16384,
                )
                while True:
                    line = await asyncio.wait_for(self.process.stdout.readline(), timeout=4)
                    if not line:
                        raise RuntimeError("tegrastats stopped producing samples")
                    clock = await asyncio.to_thread(read_gpu_clock)
                    sample = parse_tegrastats(line.decode(errors="replace"), clock)
                    self.sequence += 1
                    self.latest = {"type": "telemetry", "schema_version": 1,
                                   "sample_id": self.sequence, "timestamp_ms": time.time_ns() // 1_000_000,
                                   "monotonic_ms": time.monotonic_ns() // 1_000_000,
                                   "interval_ms": 1000, "source": "tegrastats+sysfs", **sample}
                    self.error = None
            except (OSError, ValueError, RuntimeError, TimeoutError) as error:
                self.error = f"Telemetry unavailable: {error}"
                self.latest = None
            finally:
                await self._stop_process()
            await asyncio.sleep(2)
