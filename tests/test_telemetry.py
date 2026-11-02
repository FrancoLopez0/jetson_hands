import asyncio

import pytest

from jetson_hands import telemetry
from jetson_hands.telemetry import parse_tegrastats


def test_real_orin_sample_units_and_cpu_mean():
    line = ("CPU [2%@729,4%@729,6%@1036,8%@1036,10%@1574,72%@1574] GR3D_FREQ 30% "
            "VDD_IN 7621mW/7600mW VDD_CPU_GPU_CV 1695mW/1700mW VDD_SOC 2566mW/2566mW")
    sample = parse_tegrastats(line, gpu_clock_mhz=306)
    assert sample["cpu"]["usage_pct"] == 17
    assert sample["cpu"]["clock_mhz"] == 1113
    assert sample["gpu"] == {"usage_pct": 30, "clock_mhz": 306}
    assert sample["power"]["input_w"] == 7.621
    assert sample["power"]["rails_w"]["VDD_CPU_GPU_CV"] == 1.695


def test_offline_cores_and_missing_sensors_are_not_reported_as_zero():
    sample = parse_tegrastats("CPU [off,50%@1000,0%@1000] GR3D_FREQ 0%")
    assert sample["cpu"]["usage_pct"] == 25
    assert sample["cpu"]["cores"][0]["online"] is False
    assert sample["cpu"]["cores"][0]["clock_mhz"] is None
    assert sample["power"]["input_w"] is None
    assert sample["gpu"]["clock_mhz"] is None


def test_gpu_clock_from_tegrastats_and_watt_units():
    sample = parse_tegrastats("CPU [10%@1500] GR3D_FREQ 35%@[306,306] VDD_IN 8.5W/8W")
    assert sample["gpu"]["clock_mhz"] == 306
    assert sample["power"]["input_w"] == 8.5


def test_missing_cpu_sample_is_rejected():
    with pytest.raises(ValueError):
        parse_tegrastats("tegrastats failed to read sensors")


def test_collector_samples_and_terminates_its_child(monkeypatch):
    class Process:
        returncode = None
        sent = False
        terminated = False

        @property
        def stdout(self):
            return self

        async def readline(self):
            if not self.sent:
                self.sent = True
                return b"CPU [50%@1000] GR3D_FREQ 30% VDD_IN 7500mW/7500mW\n"
            await asyncio.Event().wait()

        def terminate(self):
            self.terminated = True
            self.returncode = 0

        async def wait(self):
            return self.returncode

    process = Process()

    async def create_process(*args, **kwargs):
        assert args == ("tegrastats", "--interval", "1000")
        return process

    monkeypatch.setattr(telemetry.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(telemetry, "read_gpu_clock", lambda: 306.0)

    async def run():
        collector = telemetry.TelemetryCollector()
        await collector.start()
        try:
            async with asyncio.timeout(2):
                while collector.latest is None:
                    await asyncio.sleep(0.001)
            assert collector.latest["power"]["input_w"] == 7.5
            assert collector.latest["gpu"]["clock_mhz"] == 306
            assert collector.latest["sample_id"] == 1
        finally:
            await collector.close()
        assert process.terminated
        assert collector.process is None

    asyncio.run(run())


def test_unavailable_tegrastats_is_reported_without_crashing(monkeypatch):
    async def unavailable(*args, **kwargs):
        raise FileNotFoundError("tegrastats")

    monkeypatch.setattr(telemetry.asyncio, "create_subprocess_exec", unavailable)

    async def run():
        collector = telemetry.TelemetryCollector()
        await collector.start()
        try:
            async with asyncio.timeout(2):
                while collector.error is None:
                    await asyncio.sleep(0)
            assert collector.latest is None
            assert "tegrastats" in collector.error
        finally:
            await collector.close()

    asyncio.run(run())
