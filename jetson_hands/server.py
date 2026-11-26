import asyncio
from contextlib import asynccontextmanager
import hmac
from pathlib import Path
import subprocess
import time

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings
from .finger_follow import register_follow


class ImageControls(BaseModel):
    exposure_us: int = Field(ge=131, le=499500)
    gain: int = Field(ge=16, le=1023)
    gamma: float = Field(ge=0.2, le=4)
    fps: int | None = Field(default=None, ge=2, le=90)
    black: int | None = Field(default=None, ge=0, le=65000)
    white: int | None = Field(default=None, ge=64, le=65535)
    red: float | None = Field(default=None, ge=.25, le=4)
    green: float | None = Field(default=None, ge=.25, le=4)
    blue: float | None = Field(default=None, ge=.25, le=4)
    auto_levels: bool | None = None
    auto_wb: bool | None = None
    denoise: bool | None = None
    flip: bool | None = None
    jpeg_quality: int | None = Field(default=None, ge=40, le=95)
    preview_fps: int | None = Field(default=None, ge=1, le=60)


class FrameHub:
    """Owned by the event loop. Memory is bounded to a single processed frame."""
    def __init__(self):
        self.latest = None
        self.latest_pointing = None
        self.error = None
        self.changed = asyncio.Event()
        self.receiver = None

    def publish(self, metadata, jpeg, captured_at):
        self.latest = (metadata, jpeg, captured_at)
        self.error = None
        self.changed.set()

    def publish_pointing(self, x, width, captured_at):
        self.latest_pointing = (x, width, captured_at)

    def fail(self, error):
        self.error = error
        self.latest_pointing = None
        self.changed.set()


def create_app(settings=None, pipeline_factory=None, telemetry_factory=None):
    settings = settings or Settings.from_env()
    hub = FrameHub()
    if telemetry_factory is None:
        from .telemetry import TelemetryCollector
        telemetry_factory = TelemetryCollector
    telemetry = telemetry_factory()

    @asynccontextmanager
    async def lifespan(app):
        if pipeline_factory is None:
            from .pipeline import Pipeline
            factory = Pipeline
        else:
            factory = pipeline_factory
        pipeline = factory(settings, asyncio.get_running_loop(), hub)
        app.state.pipeline = pipeline
        pipeline.start()
        try:
            await telemetry.start()
            yield
        finally:
            await telemetry.close()
            await asyncio.to_thread(pipeline.close)

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.hub = hub
    app.state.telemetry = telemetry
    register_follow(app, hub, settings)

    @app.get("/health")
    async def health():
        age = time.monotonic() - hub.latest[2] if hub.latest else None
        ready = not hub.error and age is not None and age < 3
        # No image or recognition data is exposed without authentication.
        return JSONResponse({"ready": ready, "receiver_connected": hub.receiver is not None},
                            status_code=200 if ready else 503)

    image_lock = asyncio.Lock()
    max_fps = (15, 30, 30, 90, 60)[settings.sensor_mode]

    def image_state():
        controls = app.state.pipeline.controls
        return {**controls, "max_fps": max_fps,
                "max_exposure_us": min(499500, 1_000_000 // controls["fps"] - 500),
                "capture_width": settings.width, "capture_height": settings.height}

    def authorize_image(authorization):
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(token.encode(), settings.token.encode()):
            raise HTTPException(401, "Invalid access token")

    @app.get("/api/image-controls")
    async def image_controls(authorization: str | None = Header(default=None)):
        authorize_image(authorization)
        return image_state()

    @app.put("/api/image-controls")
    async def update_image_controls(values: ImageControls, authorization: str | None = Header(default=None)):
        authorize_image(authorization)
        async with image_lock:
            controls = {**app.state.pipeline.controls, **values.model_dump(exclude_none=True)}
            if controls["fps"] > max_fps:
                raise HTTPException(422, "FPS exceeds the current sensor mode")
            if controls["exposure_us"] > min(499500, 1_000_000 // controls["fps"] - 500):
                raise HTTPException(422, "La exposición supera el intervalo del cuadro")
            if controls["white"] <= controls["black"] + 32:
                raise HTTPException(422, "El nivel blanco debe superar al negro por más de 32")
            try:
                await asyncio.to_thread(app.state.pipeline.set_image_controls, controls)
            except (RuntimeError, OSError, subprocess.SubprocessError) as error:
                raise HTTPException(503, "Could not apply camera controls") from error
        return image_state()

    @app.websocket("/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        owns_slot = False
        try:
            auth = await asyncio.wait_for(ws.receive_json(), timeout=5)
            token = auth.get("token") if isinstance(auth, dict) else None
            if not isinstance(token, str) or not hmac.compare_digest(token.encode(), settings.token.encode()):
                await ws.send_json({"type": "error", "message": "Invalid access token"})
                await ws.close(code=4401)
                return
            # Check and claim without an await: atomic within this event loop.
            if hub.receiver is not None:
                await ws.send_json({"type": "error", "message": "Another receiver is connected"})
                await ws.close(code=4409)
                return
            hub.receiver = ws
            owns_slot = True
            await ws.send_json({"type": "ready"})
            previous = -1
            telemetry_id = None
            telemetry_error = None
            include_telemetry = auth.get("telemetry") is True
            while True:
                # Optional messages on the same authenticated/exclusive socket.
                # Never interleave a telemetry message between a frame JSON and JPEG.
                if include_telemetry:
                    sample = telemetry.latest
                    if sample is not None and sample["sample_id"] != telemetry_id:
                        await asyncio.wait_for(ws.send_json(sample), 5)
                        telemetry_id = sample["sample_id"]
                        telemetry_error = None
                    elif telemetry.error and telemetry.error != telemetry_error:
                        await asyncio.wait_for(ws.send_json({"type": "telemetry_status", "available": False,
                                                            "message": telemetry.error}), 5)
                        telemetry_error = telemetry.error
                if hub.error:
                    await ws.send_json({"type": "error", "message": "Camera pipeline unavailable"})
                    await ws.close(code=1011)
                    return
                latest = hub.latest
                if latest is None or latest[0]["sequence"] == previous:
                    hub.changed.clear()
                    # Also detect disconnected clients when the camera stalls.
                    try:
                        await asyncio.wait_for(hub.changed.wait(), 1)
                    except TimeoutError:
                        await ws.send_json({"type": "ping"})
                        reply = await asyncio.wait_for(ws.receive_json(), 5)
                        if not isinstance(reply, dict) or reply.get("type") != "pong":
                            await ws.close(code=4400)
                            return
                    continue
                metadata, jpeg, captured_at = latest
                if time.monotonic() - captured_at > 3:
                    await ws.close(code=1011, reason="Camera frames are stale")
                    return
                previous = metadata["sequence"]
                message = dict(metadata, server_frame_age_ms=round((time.monotonic() - captured_at) * 1000, 1))
                async with asyncio.timeout(10):
                    await ws.send_json(message)
                    await ws.send_bytes(jpeg)
                    # One frame in flight: slow viewers never accumulate a video queue.
                    ack = await ws.receive_json()
                if not isinstance(ack, dict) or ack.get("type") != "ack" or ack.get("sequence") != previous:
                    await ws.close(code=4400, reason="Invalid frame acknowledgement")
                    return
        except WebSocketDisconnect:
            pass
        except (TimeoutError, ValueError, TypeError):
            await ws.close(code=4408, reason="Authentication or acknowledgement timed out/invalid")
        finally:
            if owns_slot and hub.receiver is ws:
                hub.receiver = None

    web = Path(__file__).resolve().parents[1] / "web"
    if web.is_dir():
        app.mount("/", StaticFiles(directory=web, html=True), name="viewer")
    return app
