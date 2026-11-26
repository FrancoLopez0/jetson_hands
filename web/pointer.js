const POINTER_DEFAULTS = {
  zoneWidth: 96, zoneHeight: 72, margin: 8, smoothing: 100, markerWidth: 24, markerHeight: 24,
};

class PointerCalibration {
  constructor(options) {
    this.configure(options);
  }

  configure(options) {
    this.options = options;
    this.reset();
  }

  reset() {
    this.last = null;
    this.inside = null;
  }

  update(tip, width, height, now) {
    const {zoneWidth, zoneHeight, margin, smoothing} = this.options;
    const cx = width / 2, cy = height / 2;
    const hx = Math.min(zoneWidth / 2, cx), hy = Math.min(zoneHeight / 2, cy);
    const mx = Math.min(margin, cx - hx), my = Math.min(margin, cy - hy);
    if (!tip || !Number.isFinite(tip.x) || !Number.isFinite(tip.y)) {
      this.reset();
      return {cx, cy, hx, hy, mx, my, tip: null};
    }
    const alpha = this.last && smoothing > 0
      ? 1 - Math.exp(-(now - this.last.time) / smoothing) : 1;
    const x = this.last ? this.last.x + alpha * (tip.x * width - this.last.x) : tip.x * width;
    const y = this.last ? this.last.y + alpha * (tip.y * height - this.last.y) : tip.y * height;
    this.last = {x, y, time: now};
    const dx = x - cx, dy = cy - y;
    this.inside = Math.abs(dx) <= hx + (this.inside ? mx : 0) &&
                  Math.abs(dy) <= hy + (this.inside ? my : 0);
    const usefulX = this.inside ? 0 : Math.sign(dx) * Math.max(0, Math.abs(dx) - hx - mx);
    const usefulY = this.inside ? 0 : Math.sign(dy) * Math.max(0, Math.abs(dy) - hy - my);
    return {cx, cy, hx, hy, mx, my, tip: {x, y, dx, dy, usefulX, usefulY, inside: this.inside}};
  }
}

function drawCalibratedPointer(ctx, display, options) {
  const {cx, cy, hx, hy, mx, my, tip} = display;
  ctx.save();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = tip?.inside ? "rgba(111,228,179,.90)" : "rgba(232,237,245,.72)";
  ctx.strokeRect(cx - hx, cy - hy, hx * 2, hy * 2);
  if (mx || my) {
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "rgba(232,237,245,.48)";
    ctx.strokeRect(cx - hx - mx, cy - hy - my, 2 * (hx + mx), 2 * (hy + my));
    ctx.setLineDash([]);
  }
  if (tip) {
    ctx.strokeStyle = tip.inside ? "#6fe4b3" : "#ffcd70";
    ctx.lineWidth = 2.5;
    ctx.shadowColor = "#000";
    ctx.shadowBlur = 3;
    const halfWidth = options.markerWidth / 2, halfHeight = options.markerHeight / 2;
    const scale = Math.max(Math.abs(tip.x - cx) / halfWidth, Math.abs(tip.y - cy) / halfHeight);
    const ratio = scale > 1 ? 1 - 1 / scale : 1;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + (tip.x - cx) * ratio, cy + (tip.y - cy) * ratio);
    ctx.stroke();
    ctx.strokeRect(tip.x - halfWidth, tip.y - halfHeight, options.markerWidth, options.markerHeight);
    const signed = value => {
      const integer = Math.round(value);
      return integer > 0 ? "+" + integer : String(integer);
    };
    const coordinates = "X " + signed(tip.dx) + "  Y " + signed(tip.dy) + " px";
    ctx.shadowBlur = 0;
    ctx.font = "600 13px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    const pad = 6, labelHeight = 23;
    const frameWidth = cx * 2, frameHeight = cy * 2;
    const labelWidth = Math.min(frameWidth, Math.ceil(ctx.measureText(coordinates).width) + pad * 2);
    const left = Math.max(0, Math.min(tip.x - labelWidth / 2, frameWidth - labelWidth));
    const below = tip.y + halfHeight + 4;
    const top = below + labelHeight <= frameHeight ? below
      : Math.max(0, tip.y - halfHeight - labelHeight - 4);
    ctx.fillStyle = "rgba(10,16,24,.94)";
    ctx.fillRect(left, top, labelWidth, labelHeight);
    ctx.fillStyle = "#f3f7fd";
    ctx.fillText(coordinates, left + pad, top + labelHeight / 2, Math.max(1, labelWidth - pad * 2));
  }
  ctx.restore();
}
