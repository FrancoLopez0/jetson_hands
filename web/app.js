const $ = (id) => document.getElementById(id);
const canvas = $("video");
const context = canvas.getContext("2d");
let socket = null;
let metadata = null;
let lastAck = null;
let lastDisplayed = null;
let displayedFps = 0;
const telemetryDashboard = new TelemetryDashboard();

function status(message, live = false) {
  $("status").textContent = message;
  $("indicator").classList.toggle("live", live);
}

$("connection").addEventListener("submit", (event) => {
  event.preventDefault();
  if (socket) return;
  let url;
  try {
    url = new URL($("address").value);
    if (!["ws:", "wss:"].includes(url.protocol) || url.username || url.password || url.search || url.hash) throw new Error();
  } catch {
    status("Enter a ws:// or wss:// URL without credentials or query parameters.");
    return;
  }
  const token = $("token").value;
  const ws = new WebSocket(url);
  socket = ws;
  metadata = null;
  lastAck = lastDisplayed = null;
  displayedFps = 0;
  $("connect").disabled = true;
  $("disconnect").disabled = false;
  status("Connecting…");
  ws.onopen = () => ws.send(JSON.stringify({ token, telemetry: true }));
  ws.onmessage = async (event) => {
    if (socket !== ws) return;
    if (typeof event.data === "string") {
      const message = JSON.parse(event.data);
      if (message.type === "error") status(message.message);
      if (message.type === "ready") {
        status("Authenticated · waiting for camera", true);
        telemetryDashboard.beginSession(url.href);
      }
      if (message.type === "telemetry") telemetryDashboard.add(message);
      if (message.type === "telemetry_status") telemetryDashboard.unavailable(message.message);
      if (message.type === "ping") ws.send(JSON.stringify({ type: "pong" }));
      if (message.type === "frame") metadata = message;
      return;
    }
    const frame = metadata;
    if (!frame) return;
    try {
      const bitmap = await createImageBitmap(event.data);
      if (socket !== ws || ws.readyState !== WebSocket.OPEN) { bitmap.close(); return; }
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      canvas.parentElement.style.aspectRatio = `${bitmap.width} / ${bitmap.height}`;
      context.drawImage(bitmap, 0, 0);
      bitmap.close();
      $("placeholder").hidden = true;
      const now = performance.now();
      if (lastDisplayed !== null) {
        const fps = 1000 / (now - lastDisplayed);
        displayedFps = displayedFps ? displayedFps * .85 + fps * .15 : fps;
      }
      $("fps").textContent = displayedFps.toFixed(1);
      $("hands").textContent = frame.hands.length;
      $("inference").textContent = `${frame.inference_ms} ms`;
      $("pipeline").textContent = `${frame.pipeline_ms} ms`;
      $("roundtrip").textContent = lastAck === null ? "—" : `${Math.round(now - lastAck)} ms`;
      lastDisplayed = now;
      status(`Live · ${frame.width} × ${frame.height} · ${frame.strategy}`, true);
      ws.send(JSON.stringify({ type: "ack", sequence: frame.sequence }));
      lastAck = performance.now();
    } catch {
      status("Image decoding failed");
      ws.close();
    }
  };
  ws.onerror = () => status("Connection failed. Check the Jetson address and service.");
  ws.onclose = (event) => {
    if (socket !== ws) return;
    socket = null;
    telemetryDashboard.disconnect();
    $("connect").disabled = false;
    $("disconnect").disabled = true;
    $("indicator").classList.remove("live");
    context.clearRect(0, 0, canvas.width, canvas.height);
    $("placeholder").hidden = false;
    if (event.code === 4401) status("Invalid access token");
    else if (event.code === 4409) status("Another receiver is connected");
    else if (event.code === 4408) status("Connection timed out");
    else if (event.code === 1011) status("Camera pipeline unavailable");
    else status(`Disconnected (${event.code})`);
  };
});

$("disconnect").addEventListener("click", () => socket?.close(1000, "Viewer disconnected"));
window.addEventListener("pagehide", () => socket?.close(1000));
