"""End-to-end smoke check of the real Jetson camera and receiver exclusivity."""
import argparse
import asyncio
import json
from pathlib import Path
import time

import cv2
import numpy as np
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed


async def main(args):
    token = Path(args.token_file).read_text().strip()
    async with connect(args.url, max_size=4_000_000) as bad:
        await bad.send(json.dumps({"token": "invalid"}))
        assert json.loads(await bad.recv())["type"] == "error"
        try:
            await bad.recv()
        except ConnectionClosed as error:
            assert error.code == 4401
    timings = {key: [] for key in ("inference_ms", "pipeline_ms", "convert_ms", "encode_ms", "server_frame_age_ms")}
    hands = []
    received_times = []
    async with connect(args.url, max_size=4_000_000) as ws:
        await ws.send(json.dumps({"token": token}))
        assert json.loads(await ws.recv())["type"] == "ready"
        async with connect(args.url) as other:
            await other.send(json.dumps({"token": token}))
            assert json.loads(await other.recv())["type"] == "error"
            try:
                await other.recv()
            except ConnectionClosed as error:
                assert error.code == 4409
        warm_until = time.monotonic() + args.warmup
        measured_at = None
        while True:
            while True:
                meta = json.loads(await asyncio.wait_for(ws.recv(), 15))
                if meta["type"] == "ping":
                    await ws.send(json.dumps({"type": "pong"}))
                    continue
                assert meta["type"] == "frame", meta
                break
            jpeg = await ws.recv()
            image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            assert image is not None and image.shape[1] == meta["width"]
            received = time.monotonic()
            await ws.send(json.dumps({"type": "ack", "sequence": meta["sequence"]}))
            if received < warm_until:
                continue
            if measured_at is None:
                measured_at = received
            received_times.append(received)
            for key in timings:
                timings[key].append(meta[key])
            hands.append(len(meta["hands"]))
            if args.seconds is not None:
                if received - measured_at >= args.seconds:
                    break
            elif len(hands) >= args.frames:
                break
        if args.save:
            cv2.imwrite(args.save, image)
    # A newly authenticated receiver must be able to claim the released slot.
    for _ in range(30):
        async with connect(args.url) as ws:
            await ws.send(json.dumps({"token": token}))
            response = json.loads(await ws.recv())
            if response["type"] == "ready":
                break
        await asyncio.sleep(0.1)
    else:
        raise AssertionError("Receiver slot was not released")
    report = {"frames": len(hands), "strategy": meta["strategy"], "output": [meta["width"], meta["height"]],
              "received_fps": round((len(hands) - 1) / (received_times[-1] - received_times[0]), 2),
              "timings": {key: {"median": round(float(np.median(values)), 2),
                                 "p95": round(float(np.percentile(values, 95)), 2)}
                          for key, values in timings.items()},
              "max_hands_detected": max(hands), "frames_with_hand": sum(bool(value) for value in hands),
              "timestamp_source": meta["timestamp_source"], "authentication": "passed",
              "single_receiver": "passed", "reconnect": "passed"}
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8765/stream")
    parser.add_argument("--token-file", default="access.token")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--save")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--warmup", type=float, default=0)
    parser.add_argument("--report")
    args = parser.parse_args()
    if args.frames < 2 or args.warmup < 0 or (args.seconds is not None and args.seconds <= 0):
        parser.error("frames must be >=2, warmup >=0 and seconds >0")
    asyncio.run(main(args))
