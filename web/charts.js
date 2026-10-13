// Small dependency-free line-chart renderer for the training dashboard and win-rate graph.
"use strict";

const Charts = (() => {
  function niceStep(range, target) {
    const raw = range / Math.max(1, target);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const n = raw / mag;
    return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
  }
  function fmt(v) {
    const a = Math.abs(v);
    if (a >= 1e6) return (v / 1e6).toFixed(1) + "M";
    if (a >= 1e4) return (v / 1e3).toFixed(0) + "k";
    if (a >= 100) return v.toFixed(0);
    if (a >= 1) return v.toFixed(2).replace(/\.?0+$/, "");
    if (a === 0) return "0";
    if (a < 1e-3) return v.toExponential(0);
    return v.toPrecision(2);
  }
  function ema(ys, alpha) {
    const out = [];
    let m = null;
    for (const y of ys) {
      if (y === null || Number.isNaN(y)) { out.push(null); continue; }
      m = m === null ? y : alpha * y + (1 - alpha) * m;
      out.push(m);
    }
    return out;
  }
  function range(series) {
    let lo = Infinity, hi = -Infinity;
    for (const s of series) for (const y of s.y) if (y !== null && Number.isFinite(y)) { lo = Math.min(lo, y); hi = Math.max(hi, y); }
    if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
    if (lo === hi) { const d = Math.abs(lo) * 0.1 || 0.5; lo -= d; hi += d; }
    const pad = (hi - lo) * 0.06;
    return [lo - pad, hi + pad];
  }

  // series: [{name, color, x: [], y: [], axis: "left"|"right", dash: bool, smooth: bool}]
  function draw(canvas, cfg) {
    canvas._chart = cfg;
    const dpr = window.devicePixelRatio || 1;
    const W = Math.max(100, Math.round(canvas.clientWidth * dpr));
    const H = Math.max(60, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== W) canvas.width = W;
    if (canvas.height !== H) canvas.height = H;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    const series = cfg.series.map((s) => ({ ...s, y: cfg.smooth && s.smooth !== false ? ema(s.y, cfg.smoothAlpha || 0.25) : s.y }));
    const left = series.filter((s) => s.axis !== "right"), right = series.filter((s) => s.axis === "right");
    let xs = [];
    for (const s of series) xs = xs.concat(s.x);
    if (!xs.length) {
      ctx.fillStyle = "#8b93a1";
      ctx.font = `${12 * dpr}px system-ui`;
      ctx.fillText(cfg.empty || "no data yet", 12 * dpr, H / 2);
      return;
    }
    const x0 = Math.min(...xs), x1 = Math.max(...xs) === x0 ? x0 + 1 : Math.max(...xs);
    const [l0, l1] = cfg.yRange || range(left);
    const [r0, r1] = right.length ? (cfg.yRangeRight || range(right)) : [0, 1];
    const pad = { l: 46 * dpr, r: (right.length ? 46 : 12) * dpr, t: (cfg.legend === false ? 8 : 22) * dpr, b: 22 * dpr };
    const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
    const X = (x) => pad.l + ((x - x0) / (x1 - x0)) * pw;
    const YL = (y) => pad.t + (1 - (y - l0) / (l1 - l0)) * ph;
    const YR = (y) => pad.t + (1 - (y - r0) / (r1 - r0)) * ph;
    canvas._geom = { X, x0, x1, pad, pw, W, H, dpr };

    ctx.font = `${10.5 * dpr}px system-ui, sans-serif`;
    ctx.lineWidth = 1;
    // Horizontal grid + left ticks
    const step = niceStep(l1 - l0, 4);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let v = Math.ceil(l0 / step) * step; v <= l1; v += step) {
      const y = YL(v);
      ctx.strokeStyle = "rgba(255,255,255,0.06)";
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
      ctx.fillStyle = "#8b93a1";
      ctx.fillText(fmt(v), pad.l - 6 * dpr, y);
    }
    if (cfg.hline !== undefined) {
      ctx.strokeStyle = "rgba(239,106,91,0.6)";
      ctx.setLineDash([4 * dpr, 4 * dpr]);
      ctx.beginPath(); ctx.moveTo(pad.l, YL(cfg.hline)); ctx.lineTo(W - pad.r, YL(cfg.hline)); ctx.stroke();
      ctx.setLineDash([]);
    }
    if (right.length) {
      const rs = niceStep(r1 - r0, 4);
      ctx.textAlign = "left";
      for (let v = Math.ceil(r0 / rs) * rs; v <= r1; v += rs) {
        ctx.fillStyle = "#8b93a1";
        ctx.fillText(fmt(v), W - pad.r + 6 * dpr, YR(v));
      }
    }
    // X ticks
    const xstep = Math.max(1, niceStep(x1 - x0, 6));
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let v = Math.ceil(x0 / xstep) * xstep; v <= x1; v += xstep) {
      ctx.fillStyle = "#8b93a1";
      ctx.fillText(fmt(v), X(v), H - pad.b + 5 * dpr);
    }
    // Area fill (optional, for the win-rate graph)
    if (cfg.area) {
      const s = series[0];
      const base = YL(cfg.area.base);
      ctx.save();
      ctx.beginPath();
      let started = false;
      s.x.forEach((x, i) => {
        if (s.y[i] === null) return;
        if (!started) { ctx.moveTo(X(x), base); started = true; }
        ctx.lineTo(X(x), YL(s.y[i]));
      });
      if (started) {
        const lastX = s.x[s.x.length - 1];
        ctx.lineTo(X(lastX), base);
        ctx.closePath();
        ctx.clip();
        ctx.fillStyle = cfg.area.above;
        ctx.fillRect(pad.l, pad.t, pw, base - pad.t);
        ctx.fillStyle = cfg.area.below;
        ctx.fillRect(pad.l, base, pw, pad.t + ph - base);
      }
      ctx.restore();
    }
    // Lines
    for (const s of series) {
      const Y = s.axis === "right" ? YR : YL;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = (s.width || 1.8) * dpr;
      ctx.setLineDash(s.dash ? [5 * dpr, 4 * dpr] : []);
      ctx.beginPath();
      let pen = false;
      s.x.forEach((x, i) => {
        const y = s.y[i];
        if (y === null || !Number.isFinite(y)) { pen = false; return; }
        if (!pen) { ctx.moveTo(X(x), Y(y)); pen = true; } else ctx.lineTo(X(x), Y(y));
      });
      ctx.stroke();
      if (s.points) {
        ctx.fillStyle = s.color;
        s.x.forEach((x, i) => {
          if (s.y[i] === null) return;
          ctx.beginPath(); ctx.arc(X(x), Y(s.y[i]), 2.6 * dpr, 0, Math.PI * 2); ctx.fill();
        });
      }
    }
    ctx.setLineDash([]);
    // Marker (e.g. current move)
    if (cfg.marker !== undefined && cfg.marker !== null) {
      ctx.strokeStyle = "rgba(54,194,240,0.8)";
      ctx.lineWidth = 1.5 * dpr;
      ctx.beginPath(); ctx.moveTo(X(cfg.marker), pad.t); ctx.lineTo(X(cfg.marker), pad.t + ph); ctx.stroke();
    }
    // Legend
    if (cfg.legend !== false) {
      let lx = pad.l;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      for (const s of series) {
        ctx.fillStyle = s.color;
        ctx.fillRect(lx, 9 * dpr, 12 * dpr, 3 * dpr);
        ctx.fillStyle = "#c8ccd4";
        ctx.fillText(s.name, lx + 16 * dpr, 10.5 * dpr);
        lx += ctx.measureText(s.name).width + 32 * dpr;
      }
    }
    // Hover crosshair + values
    if (canvas._hoverX !== undefined && canvas._hoverX !== null) {
      const hx = canvas._hoverX;
      let best = null;
      for (const s of series) s.x.forEach((x) => { if (best === null || Math.abs(X(x) - hx) < Math.abs(X(best) - hx)) best = x; });
      if (best !== null) {
        ctx.strokeStyle = "rgba(255,255,255,0.25)";
        ctx.beginPath(); ctx.moveTo(X(best), pad.t); ctx.lineTo(X(best), pad.t + ph); ctx.stroke();
        const lines = [`${cfg.xLabel || "x"} ${fmt(best)}`];
        for (const s of series) {
          const i = s.x.indexOf(best);
          if (i >= 0 && s.y[i] !== null) lines.push(`${s.name}: ${fmt(s.y[i])}`);
        }
        ctx.font = `${11 * dpr}px system-ui, sans-serif`;
        const tw = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 14 * dpr;
        const th = lines.length * 15 * dpr + 8 * dpr;
        let tx = X(best) + 10 * dpr;
        if (tx + tw > W) tx = X(best) - tw - 10 * dpr;
        ctx.fillStyle = "rgba(15,17,20,0.92)";
        ctx.fillRect(tx, pad.t + 4 * dpr, tw, th);
        ctx.fillStyle = "#e6e8ec";
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        lines.forEach((l, i) => ctx.fillText(l, tx + 7 * dpr, pad.t + 8 * dpr + i * 15 * dpr));
      }
    }
  }

  function attachHover(canvas) {
    canvas.addEventListener("mousemove", (e) => {
      const r = canvas.getBoundingClientRect();
      canvas._hoverX = (e.clientX - r.left) * (canvas.width / r.width);
      if (canvas._chart) draw(canvas, canvas._chart);
    });
    canvas.addEventListener("mouseleave", () => {
      canvas._hoverX = null;
      if (canvas._chart) draw(canvas, canvas._chart);
    });
  }

  function parseCsv(text) {
    const lines = text.trim().split(/\r?\n/);
    if (lines.length < 1) return { cols: [], rows: [] };
    const cols = lines[0].split(",");
    const rows = lines.slice(1).map((l) => {
      const parts = l.split(",");
      const o = {};
      cols.forEach((c, i) => { o[c] = parts[i] === undefined || parts[i] === "" ? null : Number(parts[i]); });
      return o;
    });
    return { cols, rows };
  }

  return { draw, attachHover, parseCsv, fmt };
})();
