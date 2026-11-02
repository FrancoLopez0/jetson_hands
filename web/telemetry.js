/* Native-canvas telemetry charts: no CDN, credentials, or video in exports. */
class TelemetryDashboard {
  constructor() {
    this.samples = [];
    this.limit = 3600;
    this.windowMs = 120000;
    this.connected = false;
    this.receivedAt = null;
    this.lastIdentity = null;
    this.error = "";
    this.palette = ["#6fe4b3", "#78b7ff", "#ffc875", "#ed94c9", "#b49bff", "#73d8de"];
    this.charts = [
      { id: "chart-power", title: "Module power", unit: "W", series: () => [
        this.series("Module input (VDD_IN)", 0, s => s.power.input_w),
        this.series("CPU + GPU + CV rail", 1, s => s.power.rails_w.VDD_CPU_GPU_CV),
        this.series("SoC rail", 2, s => s.power.rails_w.VDD_SOC),
      ] },
      { id: "chart-usage", title: "CPU / GPU utilization", unit: "%", maximum: 100, series: () => [
        this.series("CPU mean (online)", 0, s => s.cpu.usage_pct),
        this.series("GPU", 1, s => s.gpu.usage_pct),
      ] },
      { id: "chart-clocks", title: "CPU / GPU clocks", unit: "MHz", series: () => [
        this.series("CPU mean (online)", 0, s => s.cpu.clock_mhz),
        this.series("GPU", 1, s => s.gpu.clock_mhz),
      ] },
      { id: "chart-cores", title: "CPU utilization by core", unit: "%", maximum: 100, series: () =>
        this.coreIds().map((id, index) => this.series(`CPU ${id}`, index,
          s => s.cpu.cores.find(core => core.id === id)?.usage_pct)),
      },
    ];
    document.getElementById("export-csv").addEventListener("click", () => this.exportCSV());
    document.getElementById("export-json").addEventListener("click", () => this.exportJSON());
    document.getElementById("export-charts").addEventListener("click", () => this.exportPNG());
    document.getElementById("clear-telemetry").addEventListener("click", () => this.clear());
    this.resizeObserver = new ResizeObserver(() => this.render());
    for (const chart of this.charts) this.resizeObserver.observe(document.getElementById(chart.id));
    this.timer = setInterval(() => this.updateStatus(), 1000);
    this.clear();
  }

  series(label, color, value) { return { label, color: this.palette[color % this.palette.length], value }; }
  coreIds() { return [...new Set(this.samples.flatMap(s => s.cpu.cores.map(c => c.id)))].sort((a, b) => a - b); }

  beginSession(source) {
    const url = new URL(source);
    this.source = `${url.protocol}//${url.host}${url.pathname}`;
    this.sessionId = crypto.randomUUID();
    this.connected = true;
    this.receivedAt = null;
    this.lastIdentity = null;
    this.error = "";
    this.updateStatus();
  }

  disconnect() { this.connected = false; this.updateStatus(); }
  unavailable(message) { this.error = message; this.updateStatus(); }

  add(sample) {
    if (!this.connected || !Number.isFinite(sample.timestamp_ms) || !sample.cpu || !sample.gpu ||
        !sample.power || !Array.isArray(sample.cpu.cores)) return;
    const identity = `${this.sessionId}:${sample.sample_id}`;
    if (identity === this.lastIdentity) return;
    this.lastIdentity = identity;
    this.receivedAt = performance.now();
    this.error = "";
    // Explicit whitelist: exports cannot accidentally include authentication or frames.
    const entry = { session_id: this.sessionId, source_url: this.source,
      sample_id: sample.sample_id, timestamp_ms: sample.timestamp_ms,
      monotonic_ms: sample.monotonic_ms, interval_ms: sample.interval_ms,
      source: sample.source, cpu: sample.cpu, gpu: sample.gpu, power: sample.power };
    this.samples.push(entry);
    if (this.samples.length > this.limit) this.samples.shift();
    document.getElementById("telemetry-sample-count").textContent = this.samples.length;
    this.setValue("power-now", entry.power.input_w, "W", 2);
    this.setValue("cpu-now", entry.cpu.usage_pct, "%", 1);
    this.setValue("gpu-now", entry.gpu.usage_pct, "%", 1);
    this.setValue("cpu-clock-now", entry.cpu.clock_mhz, "MHz", 0);
    this.setValue("gpu-clock-now", entry.gpu.clock_mhz, "MHz", 0);
    this.buttons(false);
    this.updateStatus();
    this.render();
  }

  setValue(id, value, unit, digits) {
    document.getElementById(id).textContent = Number.isFinite(value) ? `${value.toFixed(digits)} ${unit}` : "—";
  }

  buttons(disabled) {
    for (const id of ["export-csv", "export-json", "export-charts", "clear-telemetry"])
      document.getElementById(id).disabled = disabled;
  }

  clear() {
    this.samples = [];
    document.getElementById("telemetry-sample-count").textContent = "0";
    for (const id of ["power-now", "cpu-now", "gpu-now", "cpu-clock-now", "gpu-clock-now"])
      document.getElementById(id).textContent = "—";
    this.buttons(true);
    this.render();
    this.updateStatus();
  }

  updateStatus() {
    const text = !this.connected ? "Disconnected · history retained for export" :
      this.error || (this.receivedAt === null ? "Waiting for telemetry…" :
        performance.now() - this.receivedAt > 3500 ? "Telemetry stale · waiting for a fresh sample" : "Live · 1 sample / second");
    document.getElementById("telemetry-status").textContent = text;
  }

  render() {
    for (const chart of this.charts) {
      const canvas = document.getElementById(chart.id);
      const width = Math.max(canvas.clientWidth, 280);
      const height = 260;
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      const context = canvas.getContext("2d");
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      this.draw(context, width, height, chart);
    }
  }

  draw(context, width, height, chart) {
    context.fillStyle = "#141d2a";
    context.fillRect(0, 0, width, height);
    context.font = "600 15px system-ui";
    context.fillStyle = "#e8edf5";
    context.fillText(`${chart.title} (${chart.unit})`, 16, 25);
    const series = chart.series();
    let legendX = 16, legendY = 46;
    context.font = "11px system-ui";
    for (const item of series) {
      const itemWidth = context.measureText(item.label).width + 26;
      if (legendX + itemWidth > width - 12) { legendX = 16; legendY += 17; }
      context.fillStyle = item.color;
      context.fillRect(legendX, legendY - 7, 12, 3);
      context.fillStyle = "#b9c7da";
      context.fillText(item.label, legendX + 17, legendY - 2);
      legendX += itemWidth;
    }
    const plot = { left: 54, right: width - 16, top: legendY + 17, bottom: height - 34 };
    const end = this.samples.length ? this.samples[this.samples.length - 1].timestamp_ms : Date.now();
    const first = this.samples.length ? this.samples[0].timestamp_ms : end;
    const span = Math.min(this.windowMs, Math.max(10000, end - first));
    const start = end - span;
    const visible = this.samples.filter(s => s.timestamp_ms >= start && s.timestamp_ms <= end);
    let peak = 0;
    for (const sample of visible) for (const item of series) {
      const value = item.value(sample);
      if (Number.isFinite(value)) peak = Math.max(peak, value);
    }
    const magnitude = 10 ** Math.floor(Math.log10(Math.max(peak * 1.15, 1)));
    const maximum = chart.maximum || Math.ceil(Math.max(peak * 1.15, 1) / magnitude) * magnitude;
    context.font = "10px system-ui";
    for (let i = 0; i <= 4; i++) {
      const y = plot.bottom - i / 4 * (plot.bottom - plot.top);
      context.strokeStyle = "#29374b";
      context.beginPath(); context.moveTo(plot.left, y); context.lineTo(plot.right, y); context.stroke();
      context.fillStyle = "#9aaac0";
      context.textAlign = "right";
      context.fillText((maximum * i / 4).toLocaleString("en", { maximumFractionDigits: 2 }), plot.left - 7, y + 3);
      const x = plot.left + i / 4 * (plot.right - plot.left);
      context.textAlign = "center";
      context.fillText(i === 4 ? "latest" : `−${Number(((4 - i) * span / 4000).toFixed(1))}s`, x, height - 14);
    }
    context.textAlign = "left";
    context.save();
    context.beginPath(); context.rect(plot.left, plot.top, plot.right - plot.left, plot.bottom - plot.top); context.clip();
    for (const item of series) {
      context.strokeStyle = item.color;
      context.lineWidth = 1.8;
      context.beginPath();
      let previous = null;
      for (const sample of visible) {
        const value = item.value(sample);
        if (!Number.isFinite(value)) { previous = null; continue; }
        const x = plot.left + (sample.timestamp_ms - start) / span * (plot.right - plot.left);
        const y = plot.bottom - value / maximum * (plot.bottom - plot.top);
        if (!previous || previous.session_id !== sample.session_id || sample.timestamp_ms - previous.timestamp_ms > 4000)
          context.moveTo(x, y);
        else context.lineTo(x, y);
        previous = sample;
      }
      context.stroke();
    }
    context.restore();
    if (!visible.length) {
      context.fillStyle = "#9aaac0";
      context.fillText("Connect to collect board telemetry", plot.left + 12, (plot.top + plot.bottom) / 2);
    }
  }

  filename(extension) { return `jetson-telemetry-${new Date().toISOString().replace(/[:.]/g, "-")}.${extension}`; }

  download(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url; link.download = filename;
    document.body.appendChild(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  exportJSON() {
    if (!this.samples.length) return;
    const report = { schema_version: 1, exported_at: new Date().toISOString(),
      sampling_interval_ms: 1000, retained_sample_limit: this.limit, chart_window_seconds: 120,
      units: { power: "W", clocks: "MHz", utilization: "%" },
      definitions: { cpu: "Mean utilization/clock of online cores; per-core values included",
        input_power: "VDD_IN total module power, not the complete developer kit or battery-side power",
        cpu_gpu_cv_power: "Combined CPU/GPU/CV rail, not separate CPU and GPU power" }, samples: this.samples };
    this.download(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }), this.filename("json"));
  }

  exportCSV() {
    if (!this.samples.length) return;
    const ids = this.coreIds();
    const header = ["timestamp_utc", "session_id", "source_url", "monotonic_ms", "power_input_w",
      "power_cpu_gpu_cv_w", "power_soc_w", "cpu_usage_pct", "cpu_clock_mean_mhz", "gpu_usage_pct", "gpu_clock_mhz",
      ...ids.flatMap(id => [`cpu${id}_usage_pct`, `cpu${id}_clock_mhz`])];
    const rows = this.samples.map(s => [new Date(s.timestamp_ms).toISOString(), s.session_id, s.source_url,
      s.monotonic_ms, s.power.input_w, s.power.rails_w.VDD_CPU_GPU_CV, s.power.rails_w.VDD_SOC,
      s.cpu.usage_pct, s.cpu.clock_mhz, s.gpu.usage_pct, s.gpu.clock_mhz,
      ...ids.flatMap(id => { const core = s.cpu.cores.find(c => c.id === id); return [core?.usage_pct, core?.clock_mhz]; })]);
    const quote = value => `"${String(value ?? "").replaceAll('"', '""')}"`;
    const csv = [header, ...rows].map(row => row.map(quote).join(",")).join("\r\n");
    this.download(new Blob([csv], { type: "text/csv;charset=utf-8" }), this.filename("csv"));
  }

  exportPNG() {
    if (!this.samples.length) return;
    const canvas = document.createElement("canvas");
    canvas.width = 1600; canvas.height = 760;
    const context = canvas.getContext("2d");
    context.fillStyle = "#0b1018"; context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#e8edf5"; context.font = "600 24px system-ui";
    context.fillText("Jetson telemetry", 24, 38);
    context.font = "13px system-ui"; context.fillStyle = "#9aaac0";
    context.fillText(`Up to last 120 seconds · 1 Hz · ${new Date(this.samples[this.samples.length - 1].timestamp_ms).toISOString()}`, 24, 64);
    this.charts.forEach((chart, index) => {
      context.save(); context.translate(24 + index % 2 * 784, 90 + Math.floor(index / 2) * 300);
      this.draw(context, 768, 282, chart); context.restore();
    });
    context.font = "12px system-ui"; context.fillStyle = "#9aaac0";
    context.fillText("CPU clocks/utilization are online-core means. VDD_IN is module power, not full-kit/battery input. CPU/GPU/CV share a rail. Missing values are gaps.", 24, 730);
    canvas.toBlob(blob => { if (blob) this.download(blob, this.filename("png")); }, "image/png");
  }
}
