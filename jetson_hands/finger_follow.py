"""Turn in place using the horizontal error of the recognized pointing gesture."""
import asyncio
import hmac
import json
import math
from pathlib import Path
from time import monotonic

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

KEY_FILE = Path("/var/lib/robot-finger-pan/key")
BRIDGE = "ws://127.0.0.1:8000/ws/finger-pan"


class PanOptions(BaseModel):
    token: str
    enabled: bool = False
    zone_width: int = Field(ge=4, le=2000)
    margin: int = Field(ge=0, le=250)
    smoothing_ms: int = Field(ge=0, le=1000)
    max_rate: float = Field(ge=.2, le=.65)
    invert: bool = False


class HorizontalPan:
    def __init__(self, options):
        self.options = options
        self.reset()

    def reset(self):
        self.inside = None
        self.offset = None
        self.last_stamp = None

    def target(self, sample, now):
        if sample is None or sample[0] is None or now-sample[2] > .32:
            self.reset()
            return None
        x, width, timestamp = sample
        half = width / 2
        zone = min(self.options.zone_width / 2, half)
        outer = min(zone + self.options.margin, half)
        raw = (x-.5)*width
        if self.last_stamp is None or not self.options.smoothing_ms:
            self.offset = raw
        elif timestamp > self.last_stamp:
            alpha = 1 - math.exp(-(timestamp-self.last_stamp)*1000/self.options.smoothing_ms)
            self.offset += alpha * (raw-self.offset)
        self.last_stamp = timestamp
        magnitude = abs(self.offset)
        self.inside = magnitude <= (outer if self.inside else zone)
        if self.inside or magnitude <= outer or outer == half:
            return 0
        demand = min(1., (magnitude-outer) / (half-outer))
        sign = 1 if self.options.invert else -1
        return round(sign * math.copysign(self.options.max_rate*1000*demand, self.offset))


def register_follow(app, hub, settings):
    @app.websocket("/follow")
    async def follow(ws: WebSocket):
        origin = ws.headers.get("origin")
        if origin not in ("http://" + ws.headers.get("host", ""), "https://" + ws.headers.get("host", "")):
            await ws.close(code=1008)
            return
        await ws.accept()
        try:
            options = PanOptions.model_validate(await asyncio.wait_for(ws.receive_json(), .6))
            if not options.enabled or not hmac.compare_digest(options.token.encode(), settings.token.encode()):
                await ws.close(code=1008)
                return
            if hub.receiver is None:
                raise RuntimeError("Conectá primero el visor de la cámara.")
            tracker = HorizontalPan(options)

            async def watch():
                while await asyncio.wait_for(ws.receive_text(), .28) == "keepalive":
                    pass

            async def track():
                secret = KEY_FILE.read_text().strip()
                async with connect(BRIDGE, open_timeout=.5, close_timeout=.10, proxy=None) as bridge:
                    await bridge.send(json.dumps({"token":secret}))
                    ready = json.loads(await asyncio.wait_for(bridge.recv(), .65))
                    if ready.get("kind")!="ready":
                        raise RuntimeError(ready.get("message","El STM32 no confirmó el seguimiento."))
                    await ws.send_json({"kind":"ready"})
                    previous = None
                    while hub.receiver is not None:
                        rate = tracker.target(hub.latest_pointing, monotonic())
                        await bridge.send(json.dumps({"rate":rate}))
                        state = json.loads(await asyncio.wait_for(bridge.recv(), .28))
                        if state["kind"]=="error":
                            raise RuntimeError(state["message"])
                        if state["kind"] != previous:
                            await asyncio.wait_for(ws.send_json(state), .15)
                            previous = state["kind"]
                        await asyncio.sleep(.065)

            drive = asyncio.create_task(track())
            watcher = asyncio.create_task(watch())
            try:
                done,_ = await asyncio.wait((drive,watcher),return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            finally:
                for task in (drive,watcher):
                    if not task.done(): task.cancel()
                await asyncio.gather(drive,watcher,return_exceptions=True)
        except WebSocketDisconnect:
            pass
        except (RuntimeError, ValueError, asyncio.TimeoutError, OSError, WebSocketException) as error:
            try:
                await ws.send_json({"kind":"error","message":str(error) or "Se interrumpió el enlace de giro."})
            except (WebSocketDisconnect,RuntimeError):
                pass
        finally:
            try:
                await ws.close()
            except (WebSocketDisconnect,RuntimeError):
                pass
