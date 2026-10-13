// Canvas Go board renderer: wood texture, shaded stones, KataGo/Lizzie-style analysis overlays.
"use strict";

const N = 9;
const COLS = "ABCDEFGHJ";
const STAR_POINTS = [[2, 2], [6, 2], [2, 6], [6, 6], [4, 4]];

function actionToGtp(a) {
  if (a === 81) return "pass";
  return COLS[a % N] + (N - Math.floor(a / N));
}
function gtpToAction(v) {
  if (!v) return -1;
  v = v.toUpperCase();
  if (v === "PASS") return 81;
  const x = COLS.indexOf(v[0]);
  const row = parseInt(v.slice(1), 10);
  if (x < 0 || !(row >= 1 && row <= N)) return -1;
  return (N - row) * N + x;
}
function fmtVisits(v) {
  if (v >= 10000) return (v / 1000).toFixed(0) + "k";
  if (v >= 1000) return (v / 1000).toFixed(1) + "k";
  return String(v);
}
// Colour for a candidate move: best = cyan, then green -> yellow -> red by win-rate loss.
function candidateColor(rank, loss) {
  if (rank === 0) return [54, 194, 240];
  const t = Math.max(0, Math.min(1, loss / 0.12));
  const hue = 125 * (1 - t);
  return hslToRgb(hue / 360, 0.78, 0.55);
}
function hslToRgb(h, s, l) {
  const f = (n) => {
    const k = (n + h * 12) % 12;
    const a = s * Math.min(l, 1 - l);
    return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))));
  };
  return [f(0), f(8), f(4)];
}
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

class BoardView {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.state = null;
    this.analysis = null;
    this.opts = { cand: true, own: false, policy: false, nums: false, score: false, coords: true };
    this.hover = -1;
    this.pv = null;          // {moves: [actions], color: 1|2}
    this.interactive = true; // human may click
    this.onClick = null;
    this.onHoverCandidate = null;
    this.wood = null;
    canvas.addEventListener("mousemove", (e) => this._mouse(e));
    canvas.addEventListener("mouseleave", () => { this.hover = -1; this.pv = null; this.draw(); });
    canvas.addEventListener("click", (e) => {
      const a = this._pointFromEvent(e);
      if (a >= 0 && this.onClick) this.onClick(a);
    });
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(200, Math.round(this.canvas.clientWidth * dpr));
    if (this.canvas.width !== w) {
      this.canvas.width = w;
      this.canvas.height = w;
      this.wood = null;
    }
    this.draw();
  }

  geometry() {
    const size = this.canvas.width;
    const marginCells = this.opts.coords ? 1.08 : 0.62;
    const cell = size / (N - 1 + 2 * marginCells);
    const origin = marginCells * cell;
    return { size, cell, origin };
  }
  xy(a) {
    const g = this.geometry();
    return [g.origin + (a % N) * g.cell, g.origin + Math.floor(a / N) * g.cell];
  }
  _pointFromEvent(e) {
    const r = this.canvas.getBoundingClientRect();
    const scale = this.canvas.width / r.width;
    const px = (e.clientX - r.left) * scale, py = (e.clientY - r.top) * scale;
    const g = this.geometry();
    const x = Math.round((px - g.origin) / g.cell), y = Math.round((py - g.origin) / g.cell);
    if (x < 0 || y < 0 || x >= N || y >= N) return -1;
    const dx = px - (g.origin + x * g.cell), dy = py - (g.origin + y * g.cell);
    if (dx * dx + dy * dy > (g.cell * 0.55) ** 2) return -1;
    return y * N + x;
  }
  _mouse(e) {
    const a = this._pointFromEvent(e);
    if (a === this.hover) return;
    this.hover = a;
    this.pv = null;
    if (a >= 0 && this.analysis && this.opts.cand) {
      const m = (this.analysis.moves || []).find((mv) => mv.a === a);
      if (m && this.onHoverCandidate) this.onHoverCandidate(m);
      if (m) this.pv = { moves: m.pv.map(gtpToAction), color: this.state.to_move };
    }
    this.draw();
  }
  setPV(pvMoves) {
    this.pv = pvMoves ? { moves: pvMoves.map(gtpToAction), color: this.state ? this.state.to_move : 1 } : null;
    this.draw();
  }

  _makeWood(size) {
    const c = document.createElement("canvas");
    c.width = c.height = size;
    const g = c.getContext("2d");
    const base = g.createLinearGradient(0, 0, size, size);
    base.addColorStop(0, "#e9c27a");
    base.addColorStop(0.45, "#ddb062");
    base.addColorStop(1, "#d3a352");
    g.fillStyle = base;
    g.fillRect(0, 0, size, size);
    const rnd = mulberry32(9);
    // Wood grain: long gently waving strokes.
    for (let i = 0; i < 140; i++) {
      const y0 = rnd() * size * 1.2 - size * 0.1;
      const amp = size * (0.004 + rnd() * 0.012);
      const freq = 1 + rnd() * 3;
      const phase = rnd() * Math.PI * 2;
      g.beginPath();
      for (let x = 0; x <= size; x += size / 60) {
        const y = y0 + Math.sin((x / size) * Math.PI * freq + phase) * amp + (x / size) * size * 0.04;
        x === 0 ? g.moveTo(x, y) : g.lineTo(x, y);
      }
      const dark = rnd() < 0.7;
      g.strokeStyle = dark ? `rgba(120,72,20,${0.03 + rnd() * 0.07})` : `rgba(255,236,190,${0.04 + rnd() * 0.06})`;
      g.lineWidth = size * (0.0015 + rnd() * 0.006);
      g.stroke();
    }
    // Fine noise.
    const img = g.getImageData(0, 0, size, size);
    for (let i = 0; i < img.data.length; i += 4) {
      const n = (rnd() - 0.5) * 10;
      img.data[i] += n; img.data[i + 1] += n; img.data[i + 2] += n * 0.6;
    }
    g.putImageData(img, 0, 0);
    // Soft vignette.
    const v = g.createRadialGradient(size / 2, size / 2, size * 0.25, size / 2, size / 2, size * 0.75);
    v.addColorStop(0, "rgba(0,0,0,0)");
    v.addColorStop(1, "rgba(60,30,0,0.18)");
    g.fillStyle = v;
    g.fillRect(0, 0, size, size);
    return c;
  }

  _stone(ctx, x, y, r, color, alpha = 1) {
    ctx.save();
    ctx.globalAlpha = alpha;
    if (alpha >= 1) {
      ctx.shadowColor = "rgba(0,0,0,0.45)";
      ctx.shadowBlur = r * 0.28;
      ctx.shadowOffsetX = r * 0.07;
      ctx.shadowOffsetY = r * 0.12;
    }
    const grad = ctx.createRadialGradient(x - r * 0.35, y - r * 0.38, r * 0.08, x, y, r * 1.05);
    if (color === 1) {
      grad.addColorStop(0, "#6b6b6b");
      grad.addColorStop(0.35, "#262626");
      grad.addColorStop(1, "#050505");
    } else {
      grad.addColorStop(0, "#ffffff");
      grad.addColorStop(0.55, "#ececea");
      grad.addColorStop(1, "#b4b4ae");
    }
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();
    ctx.restore();
    if (color === 2 && alpha >= 1) {
      // Subtle clamshell lines on white stones.
      ctx.save();
      ctx.beginPath();
      ctx.arc(x, y, r * 0.97, 0, Math.PI * 2);
      ctx.clip();
      ctx.strokeStyle = "rgba(160,160,150,0.18)";
      ctx.lineWidth = Math.max(1, r * 0.03);
      for (let k = -3; k <= 3; k++) {
        ctx.beginPath();
        ctx.ellipse(x + r * 0.15, y + k * r * 0.28 + r * 1.6, r * 2.2, r * 1.6, 0, Math.PI * 1.15, Math.PI * 1.85);
        ctx.stroke();
      }
      ctx.restore();
    }
  }

  draw() {
    const ctx = this.ctx;
    const { size, cell, origin } = this.geometry();
    if (!this.wood || this.wood.width !== size) this.wood = this._makeWood(size);
    ctx.clearRect(0, 0, size, size);
    ctx.drawImage(this.wood, 0, 0);

    // Grid
    const end = origin + (N - 1) * cell;
    ctx.strokeStyle = "rgba(40,24,6,0.88)";
    ctx.lineWidth = Math.max(1, size / 650);
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const p = origin + i * cell;
      ctx.moveTo(origin, p); ctx.lineTo(end, p);
      ctx.moveTo(p, origin); ctx.lineTo(p, end);
    }
    ctx.stroke();
    ctx.lineWidth = Math.max(2, size / 300);
    ctx.strokeRect(origin, origin, end - origin, end - origin);
    ctx.fillStyle = "rgba(40,24,6,0.95)";
    for (const [x, y] of STAR_POINTS) {
      ctx.beginPath();
      ctx.arc(origin + x * cell, origin + y * cell, Math.max(2.5, cell * 0.085), 0, Math.PI * 2);
      ctx.fill();
    }
    if (this.opts.coords) {
      ctx.fillStyle = "rgba(55,32,8,0.85)";
      ctx.font = `600 ${Math.round(cell * 0.3)}px "Segoe UI", system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (let i = 0; i < N; i++) {
        const p = origin + i * cell;
        ctx.fillText(COLS[i], p, origin * 0.36);
        ctx.fillText(COLS[i], p, size - origin * 0.36);
        ctx.fillText(String(N - i), origin * 0.36, p);
        ctx.fillText(String(N - i), size - origin * 0.36, p);
      }
    }
    const s = this.state;
    if (!s) return;
    const r = cell * 0.475;
    const an = this.analysis;
    const showAnalysis = an && an.visits > 0;

    // Territory / ownership under the stones (empty points).
    if (this.opts.own && an && an.ownership) {
      for (let a = 0; a < N * N; a++) {
        const v = an.ownership[a];
        if (v === null || s.board[a] !== 0 || Math.abs(v) < 0.08) continue;
        const [x, y] = this.xy(a);
        const h = cell * 0.36 * Math.sqrt(Math.abs(v));
        ctx.fillStyle = v > 0 ? `rgba(10,10,10,${0.25 + 0.55 * Math.abs(v)})` : `rgba(250,250,248,${0.3 + 0.6 * Math.abs(v)})`;
        ctx.fillRect(x - h, y - h, 2 * h, 2 * h);
      }
    }

    // Stones
    for (let a = 0; a < N * N; a++) {
      const c = s.board[a];
      if (!c) continue;
      const [x, y] = this.xy(a);
      this._stone(ctx, x, y, r, c);
    }

    // Dead-stone marks from ownership (stone owned by the other colour).
    if (this.opts.own && an && an.ownership) {
      for (let a = 0; a < N * N; a++) {
        const c = s.board[a], v = an.ownership[a];
        if (!c || v === null) continue;
        const dead = (c === 1 && v < -0.4) || (c === 2 && v > 0.4);
        if (!dead) continue;
        const [x, y] = this.xy(a);
        const h = cell * 0.2;
        ctx.fillStyle = c === 1 ? "rgba(250,250,248,0.9)" : "rgba(10,10,10,0.9)";
        ctx.fillRect(x - h, y - h, 2 * h, 2 * h);
      }
    }

    // Move numbers or last-move marker.
    const lastMove = s.last_move;
    if (this.opts.nums && s.move_actions) {
      const num = new Map();
      s.move_actions.forEach((a, i) => { if (a !== 81) num.set(a, i + 1); });
      ctx.font = `600 ${Math.round(cell * 0.34)}px "Segoe UI", system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (const [a, n] of num) {
        if (!s.board[a]) continue;
        const [x, y] = this.xy(a);
        ctx.fillStyle = a === lastMove ? "#e34b3b" : (s.board[a] === 1 ? "#f0f0f0" : "#151515");
        ctx.fillText(String(n), x, y + 1);
      }
    } else if (lastMove >= 0 && lastMove < 81 && s.board[lastMove]) {
      const [x, y] = this.xy(lastMove);
      ctx.beginPath();
      ctx.arc(x, y, r * 0.42, 0, Math.PI * 2);
      ctx.lineWidth = Math.max(2, r * 0.12);
      ctx.strokeStyle = s.board[lastMove] === 1 ? "rgba(255,255,255,0.9)" : "rgba(0,0,0,0.85)";
      ctx.stroke();
    }

    // Simple-ko point.
    if (s.ko >= 0 && !s.board[s.ko]) {
      const [x, y] = this.xy(s.ko);
      ctx.strokeStyle = "rgba(30,20,10,0.85)";
      ctx.lineWidth = Math.max(1.5, cell * 0.04);
      ctx.strokeRect(x - cell * 0.18, y - cell * 0.18, cell * 0.36, cell * 0.36);
    }

    // Policy heatmap.
    if (this.opts.policy && an && an.policy && !this.pv) {
      let maxp = 0;
      for (let a = 0; a < 81; a++) maxp = Math.max(maxp, an.policy[a] || 0);
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (let a = 0; a < 81; a++) {
        const p = an.policy[a] || 0;
        if (s.board[a] || p < 0.002) continue;
        const [x, y] = this.xy(a);
        const rel = p / maxp;
        ctx.beginPath();
        ctx.arc(x, y, r * (0.25 + 0.7 * Math.sqrt(rel)), 0, Math.PI * 2);
        ctx.fillStyle = `rgba(232,${Math.round(90 + 110 * (1 - rel))},60,${0.3 + 0.55 * rel})`;
        ctx.fill();
        if (p >= 0.01) {
          ctx.fillStyle = "#1a1208";
          ctx.font = `600 ${Math.round(cell * 0.26)}px "Segoe UI", system-ui, sans-serif`;
          ctx.fillText((p * 100).toFixed(p >= 0.1 ? 0 : 1), x, y + 1);
        }
      }
    }

    // Candidate moves (analysis).
    if (this.opts.cand && showAnalysis && !this.pv && !s.over) {
      const moves = (an.moves || []).filter((m) => m.a < 81);
      const maxV = Math.max(1, ...moves.map((m) => m.visits));
      const best = an.moves.length ? an.moves[0].winrate : 0;
      moves.forEach((m, rank) => {
        if (rank > 0 && (rank >= 15 || m.visits < Math.max(2, maxV * 0.02))) return;
        const [x, y] = this.xy(m.a);
        const col = candidateColor(an.moves.indexOf(m), best - m.winrate);
        const alpha = 0.55 + 0.4 * Math.sqrt(m.visits / maxV);
        ctx.beginPath();
        ctx.arc(x, y, r * 0.93, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${col[0]},${col[1]},${col[2]},${alpha})`;
        ctx.fill();
        if (an.moves.indexOf(m) === 0) {
          ctx.lineWidth = Math.max(2, r * 0.1);
          ctx.strokeStyle = "rgba(10,60,90,0.9)";
          ctx.stroke();
        }
        ctx.fillStyle = "#101010";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.font = `700 ${Math.round(cell * 0.27)}px "Segoe UI", system-ui, sans-serif`;
        ctx.fillText((m.winrate * 100).toFixed(1), x, y - cell * 0.11);
        ctx.font = `500 ${Math.round(cell * 0.21)}px "Segoe UI", system-ui, sans-serif`;
        const sub = this.opts.score ? (m.score_lead >= 0 ? "+" : "") + m.score_lead.toFixed(1) : fmtVisits(m.visits);
        ctx.fillText(sub, x, y + cell * 0.17);
      });
      // A pass recommendation is shown as a label.
      if (an.moves.length && an.moves[0].a === 81) {
        ctx.font = `700 ${Math.round(cell * 0.3)}px "Segoe UI", system-ui, sans-serif`;
        ctx.fillStyle = "rgba(10,60,90,0.9)";
        ctx.textAlign = "center";
        ctx.fillText(`AI suggests PASS (${(an.moves[0].winrate * 100).toFixed(1)}%)`, size / 2, origin * 0.36 + (this.opts.coords ? cell * 0.35 : 0));
      }
    }

    // Principal variation preview.
    if (this.pv && this.pv.moves.length) {
      let color = this.pv.color;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      const placed = new Set();
      this.pv.moves.forEach((a, i) => {
        if (a >= 0 && a < 81 && !s.board[a] && !placed.has(a)) {
          placed.add(a);
          const [x, y] = this.xy(a);
          this._stone(ctx, x, y, r, color, 0.82);
          ctx.fillStyle = color === 1 ? "#f2f2f2" : "#111";
          ctx.font = `700 ${Math.round(cell * 0.32)}px "Segoe UI", system-ui, sans-serif`;
          ctx.fillText(String(i + 1), x, y + 1);
        }
        color = 3 - color;
      });
    }

    // Hover ghost stone.
    if (this.interactive && this.hover >= 0 && !this.pv && !s.over && s.legal[this.hover]) {
      const [x, y] = this.xy(this.hover);
      this._stone(ctx, x, y, r, s.to_move, 0.45);
    }
  }
}
