const $ = (id) => document.getElementById(id);
const canvas = $("video");
const context = canvas.getContext("2d");
let socket = null;
let metadata = null;
let lastAck = null;
let lastDisplayed = null;
let displayedFps = 0;
const telemetryDashboard = new TelemetryDashboard();
const streamProtocol = location.protocol === "https:" ? "wss:" : "ws:";
$("address").value = streamProtocol + "//" + location.host + "/stream";

const imageForm = $("image-controls");
const imagePanel = $("image-panel");
let imageToken = null;
let pendingImage = null, applyingImage = false;
const imageNumbers = {gain:"gain", gamma:"gamma", fps:"image-fps", black:"black", white:"white",
  red:"red", green:"green", blue:"blue", preview_fps:"preview-fps", jpeg_quality:"jpeg-quality"};
const imageToggles = {auto_levels:"auto-levels", auto_wb:"auto-wb", denoise:"denoise", flip:"flip"};
const sliders = new Map();
function syncSlider(field) {
  const slider = sliders.get(field);
  if (!slider) return;
  for (const attr of ["min","max","step","value"]) slider[attr] = field[attr];
}
function addSlider(field, onChange) {
  const label = field.closest("label");
  const slider = document.createElement("input");
  slider.type = "range";
  slider.setAttribute("aria-label", label.firstChild.textContent.trim());
  label.classList.add("slider-control");
  label.append(slider);
  sliders.set(field, slider);
  syncSlider(field);
  slider.addEventListener("input", () => {
    field.value = slider.value;
    field.dispatchEvent(new Event("input"));
  });
  slider.addEventListener("change", onChange);
  field.addEventListener("input", () => syncSlider(field));
}
function showControls(settings) {
  $("exposure").value = settings.exposure_us / 1000;
  $("exposure").max = settings.max_exposure_us / 1000;
  $("image-fps").max = settings.max_fps;
  $("sensor-mode").textContent = settings.capture_width + " × " + settings.capture_height;
  for (const [key,id] of Object.entries(imageNumbers)) $(id).value = settings[key];
  for (const [key,id] of Object.entries(imageToggles)) $(id).checked = settings[key];
  for (const field of imageForm.querySelectorAll('input[type="number"]')) syncSlider(field);
  $("image-fields").disabled = false;
  $("image-status").textContent = "Ajustes disponibles. Arrastrá una barra y soltá para aplicar.";
}
$("image-fps").addEventListener("input", () => {
  const fps = Number($("image-fps").value);
  if (fps >= 2) $("exposure").max = Math.min(499500, Math.floor(1000000 / fps) - 500) / 1000;
  syncSlider($("exposure"));
});
async function imageRequest(method, token, values) {
  const headers = {Authorization: "Bearer " + token};
  if (values) headers["Content-Type"] = "application/json";
  const response = await fetch("/api/image-controls", {
    method, headers, body: values ? JSON.stringify(values) : undefined, cache: "no-store",
  });
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : "Revisá los valores");
  return body;
}
async function applyImage() {
  if (!imageToken || !imageForm.reportValidity()) return;
  pendingImage = {token:imageToken, settings: {
    exposure_us: Math.round(Number($("exposure").value) * 1000),
    ...Object.fromEntries(Object.entries(imageNumbers).map(([key,id]) => [key,Number($(id).value)])),
    ...Object.fromEntries(Object.entries(imageToggles).map(([key,id]) => [key,$(id).checked])),
  }};
  if (applyingImage) return;
  applyingImage = true;
  try {
    while (pendingImage) {
      const {token, settings} = pendingImage;
      pendingImage = null;
      if (token !== imageToken) continue;
      $("image-status").textContent = "Aplicando ajustes…";
      try {
        await imageRequest("PUT", token, settings);
        if (token === imageToken) $("image-status").textContent = "Ajustes aplicados";
      } catch (error) {
        if (token === imageToken) $("image-status").textContent = error.message;
      }
    }
  } finally {
    applyingImage = false;
  }
}
for (const field of imageForm.querySelectorAll('input[type="number"]')) addSlider(field, applyImage);
for (const id of Object.values(imageToggles)) $(id).addEventListener("change", applyImage);
imageForm.addEventListener("submit", event => {
  event.preventDefault();
  applyImage();
});
document.querySelector(".camera-link").addEventListener("click", () => { imagePanel.open = true; });

const zoneForm = $("zone-form");
const configKey = "jetson-hands-pointer-v1";
const pointer = new PointerCalibration(POINTER_DEFAULTS);
function applyZone(options) {
  for (const [key,value] of Object.entries(POINTER_DEFAULTS)) {
    const field = zoneForm.elements.namedItem(key);
    field.value = options[key] ?? value;
    if (!field.checkValidity()) field.value = value;
    syncSlider(field);
  }
  pointer.configure(Object.fromEntries(Object.keys(POINTER_DEFAULTS).map(
    key => [key,Number(zoneForm.elements.namedItem(key).value)])));
}
try { applyZone(JSON.parse(localStorage.getItem(configKey)) || POINTER_DEFAULTS); }
catch { applyZone(POINTER_DEFAULTS); }
function saveZone() {
  applyZone(Object.fromEntries(new FormData(zoneForm)));
  try { localStorage.setItem(configKey,JSON.stringify(pointer.options)); } catch {}
}
for (const field of zoneForm.querySelectorAll('input[type="number"]')) addSlider(field, saveZone);
zoneForm.addEventListener("submit", event => {
  event.preventDefault();
  saveZone();
});
$("reset-zone").addEventListener("click", () => {
  applyZone(POINTER_DEFAULTS);
  try { localStorage.removeItem(configKey); } catch {}
});
const signedPixels = (x,y) => [Math.round(x),Math.round(y)]
  .map(value => value > 0 ? "+" + value : String(value)).join(" / ");

const panEnabled = $("pan-enabled");
let panSocket = null;
const panMessage = text => { $("pan-status").textContent = text; };
$("pan-speed").oninput = () => {
  $("pan-speed-value").textContent = Number($("pan-speed").value).toFixed(2).replace(".",",");
};
function lockPanOptions(active) {
  $("pan-speed").disabled = active;
  $("pan-invert").disabled = active;
  zoneForm.inert = imageForm.inert = active;
}
function stopPan() {
  panEnabled.checked = false;
  if (panSocket) {
    if (panSocket.readyState === WebSocket.OPEN) panSocket.send("stop");
    panSocket.close();
  }
}
panEnabled.onchange = () => {
  if (!panEnabled.checked) return stopPan();
  if (!imageToken || panSocket) { panEnabled.checked = false; return; }
  const drive = new WebSocket(streamProtocol + "//" + location.host + "/follow");
  panSocket = drive;
  let timer, failed = false;
  lockPanOptions(true);
  panMessage("Comprobando el STM32…");
  drive.onopen = () => {
    if (!panEnabled.checked || !imageToken) return drive.close();
    drive.send(JSON.stringify({token:imageToken, enabled:true, zone_width:pointer.options.zoneWidth,
      margin:pointer.options.margin, smoothing_ms:pointer.options.smoothing,
      max_rate:Number($("pan-speed").value), invert:$("pan-invert").checked}));
    timer = setInterval(() => { if (drive.readyState === WebSocket.OPEN) drive.send("keepalive"); }, 75);
  };
  drive.onmessage = ({data}) => {
    const msg = JSON.parse(data);
    if (msg.kind === "error") {
      failed = true;
      panMessage(msg.message);
      stopPan();
      return;
    }
    panMessage({ready:"Seguimiento habilitado.", no_gesture:"Esperando el gesto del índice.",
      neutral:"Zona neutra · robot detenido.", tracking:"Girando para centrar el índice.",
      braking:"Frenando…", settling:"Esperando que se detengan ambas ruedas."}[msg.kind] || "");
  };
  drive.onclose = () => {
    clearInterval(timer);
    if (panSocket !== drive) return;
    panSocket = null;
    panEnabled.checked = false;
    lockPanOptions(false);
    if (!failed) panMessage("Seguimiento desactivado.");
  };
};
window.addEventListener("pagehide",stopPan);
window.addEventListener("blur",stopPan);
document.addEventListener("visibilitychange", () => { if (document.hidden) stopPan(); });

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
  imageToken = null;
  $("image-fields").disabled = true;
  $("image-status").textContent = "Conectando con la cámara…";
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
        if (url.host === location.host && url.protocol === streamProtocol) {
          imageToken = token;
          panEnabled.disabled = false;
          panMessage("Seguimiento desactivado.");
          imageRequest("GET", token).then(result => {
            if (socket === ws) showControls(result);
          }).catch(error => {
            if (socket === ws) $("image-status").textContent = "No se pudieron leer los ajustes. " + error.message;
          });
        } else {
          $("image-status").textContent = "Abrí el visor desde la misma dirección de la cámara para ajustar la imagen.";
        }
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
      const tip = frame.index_tip;
      const display = pointer.update(tip, canvas.width, canvas.height, now);
      drawCalibratedPointer(context, display, pointer.options);
      $("index-offset").textContent = tip
        ? signedPixels((tip.x - .5) * frame.width, (.5 - tip.y) * frame.height) : "—";
      $("safe-offset").textContent = display.tip
        ? signedPixels(display.tip.usefulX, display.tip.usefulY) : "—";
      $("zone-state").textContent = display.tip
        ? (display.tip.inside ? "Zona neutra · útil (px)" : "Fuera de zona · útil (px)")
        : "Desplazamiento útil (px)";
      $("index-tip").textContent = tip ? `${tip.x.toFixed(2)} / ${tip.y.toFixed(2)}` : frame.hands?.length ? "Sin gesto" : "Sin mano";
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
    stopPan();
    panEnabled.disabled = true;
    panMessage("Conectá la cámara para habilitar el seguimiento.");
    telemetryDashboard.disconnect();
    $("connect").disabled = false;
    $("disconnect").disabled = true;
    $("indicator").classList.remove("live");
    $("index-tip").textContent = "—";
    $("index-offset").textContent = $("safe-offset").textContent = "—";
    $("zone-state").textContent = "Desplazamiento útil (px)";
    pointer.reset();
    imageToken = null;
    pendingImage = null;
    $("image-fields").disabled = true;
    $("image-status").textContent = "Conectate con el token para habilitar los ajustes.";
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
