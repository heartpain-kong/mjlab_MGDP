/* 蜡烛图画布引擎 —— 没有第三方图表库，全部手画。
 *
 * 支持三种形态：
 *   k     蜡烛 + 均线 + 波动柱（默认）
 *   line  分时曲线（带渐变面积、基准线、最新值标签）
 *   bars  副图（柱 + 均值线）
 *
 * 交互：滚轮缩放（以光标为锚点）、拖动平移、双击复位、悬停十字光标。
 * 缩放/平移只动 view.start / view.count，不重新请求数据。
 */
(function () {
  "use strict";

  const PAD = { r: 64, t: 22, b: 20, l: 8 };
  const MA_STYLES = [
    { n: 5, key: "ma5" },
    { n: 10, key: "ma10" },
    { n: 20, key: "ma20" },
    { n: 60, key: "ma60" },
  ];

  const DEFAULT_PALETTE = {
    up: "#ff4d4f",
    down: "#00b578",
    grid: "#232833",
    gridStrong: "#2c3441",
    axis: "#6f7787",
    text: "#9aa2b1",
    accent: "#f0b90b",
    base: "#6b7280",
    overlay: "#4fc3f7",
    ma: ["#e9edf4", "#f0b90b", "#a970ff", "#2bd6a0"],
    cross: "#c9cdd6",
    bg: "#181c24",
  };

  class TerminalChart {
    constructor(canvas, options) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.options = options || {};
      this.palette = Object.assign({}, DEFAULT_PALETTE);
      this.data = null;
      this.view = { start: 0, count: 0 };
      this.hover = -1;
      this.reveal = 0;          // 回放：只画前 reveal 根（0 = 全部）
      this.dragging = null;
      this.dpr = 1;
      this.width = 0;
      this.height = 0;
      this._bind();
      this.resize();
      if (window.ResizeObserver) {
        this._observer = new ResizeObserver(() => this.resize());
        this._observer.observe(canvas.parentElement || canvas);
      } else {
        window.addEventListener("resize", () => this.resize());
      }
    }

    setPalette(palette) {
      this.palette = Object.assign({}, DEFAULT_PALETTE, palette);
      this.draw();
    }

    setOverlay(overlay) {
      this.overlay = overlay || null;
      if (this.data) this.data.overlay = this.overlay;
      this.draw();
    }

    setData(data) {
      const sameSeries = this.data && data && this.data.seriesKey === data.seriesKey;
      this.data = data;
      if (data) this.overlay = data.overlay || null;
      const n = this.length();
      if (!sameSeries || this.view.count === 0) {
        this.view.count = Math.min(n, this.options.defaultCount || 140);
        this.view.start = Math.max(0, this.length() - this.view.count);
      }
      this.clampView();
      this.draw();
    }

    length() {
      if (!this.data) return 0;
      if (this.data.mode === "line") return this.data.points ? this.data.points.y.length : 0;
      return this.data.candles ? this.data.candles.length : 0;
    }

    visible() {
      const n = this.length();
      const shown = this.reveal > 0 ? Math.min(n, this.reveal) : n;
      const start = Math.max(0, Math.min(this.view.start, shown - 1));
      const count = Math.max(2, Math.min(this.view.count, shown - start));
      return { start, count, shown };
    }

    clampView() {
      const n = this.length();
      if (!n) { this.view.start = 0; this.view.count = 0; return; }
      this.view.count = Math.max(2, Math.min(this.view.count, n));
      this.view.start = Math.max(0, Math.min(this.view.start, n - this.view.count));
    }

    reset() {
      const n = this.length();
      this.view.count = Math.min(n, this.options.defaultCount || 140);
      this.view.start = Math.max(0, n - this.view.count);
      this.draw();
    }

    setReveal(value) {
      this.reveal = value;
      if (value > 0) {
        this.view.start = 0;
        this.view.count = Math.max(2, Math.min(value, 140));
      }
      this.draw();
    }

    viewAll() {
      this.view.start = 0;
      this.view.count = this.length();
      this.draw();
    }

    resize() {
      const rect = this.canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      this.dpr = window.devicePixelRatio || 1;
      this.width = rect.width;
      this.height = rect.height;
      this.canvas.width = Math.round(rect.width * this.dpr);
      this.canvas.height = Math.round(rect.height * this.dpr);
      this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      this.draw();
    }

    // ---------------------------------------------------------------- 绘制
    draw() {
      const ctx = this.ctx;
      const W = this.width;
      const H = this.height;
      if (!W || !H) return;
      ctx.clearRect(0, 0, W, H);
      const data = this.data;
      if (!data || !this.length()) {
        ctx.fillStyle = this.palette.axis;
        ctx.font = "12px 'Microsoft YaHei UI', sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("没有数据", W / 2, H / 2);
        return;
      }
      if (data.mode === "line") return this._drawLine();
      if (data.mode === "bars") return this._drawBars();
      return this._drawCandles();
    }

    _plot() {
      return {
        left: PAD.l,
        top: PAD.t,
        right: this.width - PAD.r,
        bottom: this.height - PAD.b,
      };
    }

    _niceTicks(min, max, count) {
      if (!(max > min)) return [min];
      const raw = (max - min) / Math.max(1, count);
      const mag = Math.pow(10, Math.floor(Math.log10(raw)));
      const norm = raw / mag;
      // 1/2/5×10ⁿ 取整：阈值放宽一点，窄区间才不会只剩两根刻度
      const step = (norm <= 1.5 ? 1 : norm <= 3.5 ? 2 : norm <= 7.5 ? 5 : 10) * mag;
      const ticks = [];
      for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) ticks.push(v);
      return ticks;
    }

    _format(value, range) {
      const span = Math.abs(range || 0);
      let digits = 2;
      if (span >= 10000) digits = 0;
      else if (span >= 1000) digits = 1;
      else if (span >= 10) digits = 2;
      else if (span >= 1) digits = 2;
      else if (span >= 0.01) digits = 4;
      else digits = 6;
      if (Math.abs(value) >= 100000) return (value / 1000).toFixed(0) + "k";
      if (Math.abs(value) >= 10000) return value.toFixed(0);
      return value.toFixed(digits);
    }

    _axis(ctx, plot, min, max, count) {
      const p = this.palette;
      ctx.font = "10.5px 'Bahnschrift','Consolas',monospace";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      const ticks = this._niceTicks(min, max, count || this.options.ticks || 4);
      const span = max - min || 1;
      for (const tick of ticks) {
        const y = plot.bottom - ((tick - min) / span) * (plot.bottom - plot.top);
        if (y < plot.top - 2 || y > plot.bottom + 2) continue;
        ctx.strokeStyle = p.grid;
        ctx.beginPath();
        ctx.moveTo(plot.left, Math.round(y) + 0.5);
        ctx.lineTo(plot.right, Math.round(y) + 0.5);
        ctx.stroke();
        ctx.fillStyle = p.axis;
        ctx.fillText(this._format(tick, span), plot.right + 7, y);
      }
      return ticks;
    }

    _xLabels(ctx, plot, count, labelOf) {
      const p = this.palette;
      ctx.font = "10px 'Bahnschrift','Consolas',monospace";
      ctx.fillStyle = p.axis;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      const step = Math.max(1, Math.ceil(count / 8));
      for (let i = 0; i < count; i += step) {
        const x = plot.left + ((i + 0.5) / count) * (plot.right - plot.left);
        const text = labelOf(i);
        if (text == null) continue;
        ctx.strokeStyle = p.grid;
        ctx.beginPath();
        ctx.moveTo(Math.round(x) + 0.5, plot.top);
        ctx.lineTo(Math.round(x) + 0.5, plot.bottom);
        ctx.stroke();
        ctx.fillText(text, x, plot.bottom + 5);
      }
    }

    _drawCandles() {
      const ctx = this.ctx;
      const p = this.palette;
      const data = this.data;
      const rows = data.candles;
      const view = this.visible();
      const plot = this._plot();
      const volHeight = Math.max(24, (plot.bottom - plot.top) * 0.2);
      const volTop = plot.bottom - volHeight;
      const priceBottom = volTop - 8;
      const slice = rows.slice(view.start, view.start + view.count);
      let min = Infinity;
      let max = -Infinity;
      for (const row of slice) {
        if (row.l < min) min = row.l;
        if (row.h > max) max = row.h;
      }
      const pad = (max - min) * 0.08 || Math.abs(max || 1) * 0.02 || 1;
      min -= pad;
      max += pad;

      const plotPrice = { left: plot.left, top: plot.top, right: plot.right, bottom: priceBottom };
      this._axis(ctx, plotPrice, min, max);
      // x 轴标签画在整张图的最底下（波动柱下面），不然会和「振幅」那行字撞在一起
      this._xLabels(ctx, plot, view.count, (i) => {
        const row = slice[i];
        return row ? "步 " + compact(row.t) : null;
      });

      const cw = (plot.right - plot.left) / view.count;
      const bodyWidth = Math.max(1, Math.min(14, cw * 0.68));
      const yOf = (v) => priceBottom - ((v - min) / (max - min || 1)) * (priceBottom - plot.top);
      const p2 = this.palette;

      // 基准线：图上第一根的收盘（相当于「昨收」）
      const base = data.base != null ? data.base : slice.length ? slice[0].c : null;
      if (base != null && base >= min && base <= max) {
        ctx.save();
        ctx.strokeStyle = p2.base;
        ctx.setLineDash([4, 4]);
        ctx.globalAlpha = 0.8;
        ctx.beginPath();
        ctx.moveTo(plot.left, yOf(base));
        ctx.lineTo(plot.right, yOf(base));
        ctx.stroke();
        ctx.restore();
      }

      slice.forEach((row, i) => {
        const x = plot.left + (i + 0.5) * cw;
        const rising = row.c >= row.o;
        const color = rising ? p2.up : p2.down;
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 1;
        // 影线
        const xr = Math.round(x) + 0.5;
        ctx.beginPath();
        ctx.moveTo(xr, yOf(row.h));
        ctx.lineTo(xr, yOf(row.l));
        ctx.stroke();
        // 实体
        const yo = yOf(Math.max(row.o, row.c));
        const yc = yOf(Math.min(row.o, row.c));
        const height = Math.max(1, yc - yo);
        const left = Math.round(x - bodyWidth / 2);
        if (rising) {
          ctx.globalAlpha = 0.95;
          ctx.fillRect(left, yo, Math.max(1, Math.round(bodyWidth)), height);
          ctx.globalAlpha = 1;
        } else {
          ctx.fillRect(left, yo, Math.max(1, Math.round(bodyWidth)), height);
        }
      });

      // 均线
      this._movingAverages(ctx, slice, view, plot.left, cw, yOf);

      // 波动柱
      this._volume(ctx, slice, view, plot, volTop, volHeight, cw, bodyWidth);

      // 叠到主图的副图曲线
      this._drawOverlay(ctx, plot, plot.top, priceBottom, slice, cw);

      // 最新价标签
      const last = slice[slice.length - 1];
      if (last) {
        const y = yOf(last.c);
        const rising = last.c >= last.o;
        this._priceTag(ctx, plot.right, y, last.c, rangeOf(slice), rising ? p2.up : p2.down);
      }
      this._maLegend(ctx, slice, view, plot);
      this._crosshair(ctx, plot, slice, view, cw, yOf, "k");
    }

    _drawOverlay(ctx, plot, top, bottom, slice, cw) {
      const overlay = this.overlay;
      if (!overlay || !overlay.values || !overlay.values.length) return;
      const slot = this.visible();
      const values = overlay.values.slice(slot.start, slot.start + slot.count);
      let min = Infinity;
      let max = -Infinity;
      let seen = false;
      for (const value of values) {
        if (value == null || !isFinite(value)) continue;
        seen = true;
        if (value < min) min = value;
        if (value > max) max = value;
      }
      if (!seen) return;
      const pad = (max - min) * 0.1 || Math.abs(max || 1) * 0.02 || 1;
      min -= pad;
      max += pad;
      const yOf = (value) => bottom - ((value - min) / (max - min)) * (bottom - top);
      ctx.save();
      ctx.strokeStyle = overlay.color || this.palette.overlay;
      ctx.lineWidth = 1.1;
      ctx.globalAlpha = 0.85;
      ctx.beginPath();
      let started = false;
      values.forEach((value, index) => {
        if (value == null || !isFinite(value)) { started = false; return; }
        const x = plot.left + (index + 0.5) * cw;
        const y = yOf(value);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.restore();
      const text = "副图 " + (overlay.label || "") + " " + this._format(values[values.length - 1] || max, max - min);
      ctx.font = "10px 'Bahnschrift','Consolas',monospace";
      const width = ctx.measureText(text).width;
      ctx.fillStyle = overlay.color || this.palette.overlay;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(text, Math.max(plot.left + 2, plot.right - width - 4), 10);
    }

    _movingAverages(ctx, slice, view, left, cw, yOf) {
      const p = this.palette;
      const data = this.data;
      ctx.lineWidth = 1.2;
      MA_STYLES.forEach((style, index) => {
        if (data.ma && data.ma[index] === false) return;
        ctx.strokeStyle = p.ma[index];
        ctx.globalAlpha = 0.9;
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < slice.length; i++) {
          const value = data.ma5 && average(slice, i, style.n);
          if (value == null) continue;
          const x = left + (i + 0.5) * cw;
          const y = yOf(value);
          if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.globalAlpha = 1;
      });
    }

    _maLegend(ctx, slice, view, plot) {      const p = this.palette;
      ctx.font = "10px 'Bahnschrift','Consolas',monospace";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      let x = plot.left + 2;
      const y = 10;
      MA_STYLES.forEach((style, index) => {
        const value = average(slice, slice.length - 1, style.n);
        const text = "MA" + style.n + (value == null ? " —" : " " + this._format(value, rangeOf(slice)));
        ctx.fillStyle = p.ma[index];
        ctx.fillText(text, x, y);
        x += ctx.measureText(text).width + 12;
      });
    }

    _volume(ctx, slice, view, plot, volTop, volHeight, cw, bodyWidth) {
      const p = this.palette;
      const kind = this.data.volumeKind || "range";
      let max = 0;
      for (const row of slice) {
        const v = row.vol ? row.vol[kind] : 0;
        if (v > max) max = v;
      }
      if (!(max > 0)) max = 1;
      ctx.fillStyle = "rgba(255,255,255,.03)";
      ctx.fillRect(plot.left, volTop, plot.right - plot.left, volHeight);
      ctx.font = "9.5px 'Bahnschrift','Consolas',monospace";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillStyle = p.axis;
      ctx.fillText(VOLUME_LABELS[kind] || "波动", plot.left + 3, volTop + 2);
      for (let i = 0; i < slice.length; i++) {
        const row = slice[i];
        const value = row.vol ? row.vol[kind] : 0;
        const h = Math.max(1, (value / max) * (volHeight - 10));
        const x = plot.left + (i + 0.5) * cw;
        const rising = row.c >= row.o;
        ctx.fillStyle = rising ? p.up : p.down;
        ctx.globalAlpha = 0.55;
        ctx.fillRect(Math.round(x - bodyWidth / 2), volTop + volHeight - h, Math.max(1, Math.round(bodyWidth)), h);
        ctx.globalAlpha = 1;
      }
    }

    _title(ctx, plot) {
      const label = this.data && this.data.label;
      if (!label) return;
      ctx.font = "10.5px 'Microsoft YaHei UI',sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillStyle = this.palette.accent;
      ctx.fillText(label, plot.left + 4, 4);
      const sub = this.data.subtitle;
      if (sub) {
        const width = ctx.measureText(label).width;
        ctx.fillStyle = this.palette.axis;
        ctx.font = "9.5px 'Bahnschrift','Consolas',monospace";
        ctx.fillText(sub, plot.left + 10 + width, 5.5);
      }
    }

    _priceTag(ctx, x, y, value, range, color) {      const p = this.palette;
      const text = this._format(value, range);
      ctx.font = "10.5px 'Bahnschrift','Consolas',monospace";
      const w = ctx.measureText(text).width + 10;
      ctx.fillStyle = color;
      ctx.fillRect(x + 1, y - 8, w, 16);
      ctx.fillStyle = "#fff";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(text, x + 6, y);
    }

    _drawLine() {
      const ctx = this.ctx;
      const p = this.palette;
      const data = this.data;
      const points = data.points;
      const view = this.visible();
      const plot = this._plot();
      const xs = points.y.slice(view.start, view.start + view.count);
      let min = Infinity;
      let max = -Infinity;
      for (const v of xs) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
      if (!isFinite(min)) return;
      const pad = (max - min) * 0.1 || Math.abs(max || 1) * 0.02 || 1;
      min -= pad;
      max += pad;
      this._axis(ctx, plot, min, max);
      this._title(ctx, plot);
      this._xLabels(ctx, plot, view.count, (i) => {
        const step = points.step[view.start + i];
        return step == null ? null : "步 " + compact(step);
      });
      const yOf = (v) => plot.bottom - ((v - min) / (max - min || 1)) * (plot.bottom - plot.top);
      const cw = (plot.right - plot.left) / view.count;
      const base = data.base != null ? data.base : xs[0];
      ctx.save();
      ctx.strokeStyle = p.base;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(plot.left, yOf(base));
      ctx.lineTo(plot.right, yOf(base));
      ctx.stroke();
      ctx.restore();

      const rising = xs[xs.length - 1] >= base;
      const color = rising ? p.up : p.down;
      const gradient = ctx.createLinearGradient(0, plot.top, 0, plot.bottom);
      gradient.addColorStop(0, hexAlpha(color, 0.28));
      gradient.addColorStop(1, hexAlpha(color, 0.02));
      ctx.beginPath();
      ctx.moveTo(plot.left, plot.bottom);
      xs.forEach((v, i) => ctx.lineTo(plot.left + (i + 0.5) * cw, yOf(v)));
      ctx.lineTo(plot.right, plot.bottom);
      ctx.closePath();
      ctx.fillStyle = gradient;
      ctx.fill();

      ctx.beginPath();
      xs.forEach((v, i) => {
        const x = plot.left + (i + 0.5) * cw;
        const y = yOf(v);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.4;
      ctx.stroke();

      this._priceTag(ctx, plot.right, yOf(xs[xs.length - 1]), xs[xs.length - 1], max - min, color);
      this._crosshair(ctx, plot, xs, view, cw, yOf, "line");
    }

    _drawBars() {
      const ctx = this.ctx;
      const p = this.palette;
      const data = this.data;
      const rows = data.candles;
      const view = this.visible();
      const plot = this._plot();
      const slice = rows.slice(view.start, view.start + view.count);
      let min = 0;
      let max = 0;
      for (const row of slice) {
        const v = data.value(row);
        if (v < min) min = v;
        if (v > max) max = v;
      }
      if (min === max) max = min + 1;
      const pad = (max - min) * 0.1;
      max += pad;
      if (min < 0) min -= pad;
      this._axis(ctx, plot, min, max);
      this._title(ctx, plot);
      this._xLabels(ctx, plot, view.count, (i) => (slice[i] ? "步 " + compact(slice[i].t) : null));
      const cw = (plot.right - plot.left) / view.count;
      const width = Math.max(1, Math.min(10, cw * 0.6));
      const zero = plot.bottom - ((0 - min) / (max - min)) * (plot.bottom - plot.top);
      ctx.strokeStyle = p.gridStrong;
      ctx.beginPath();
      ctx.moveTo(plot.left, zero);
      ctx.lineTo(plot.right, zero);
      ctx.stroke();
      const yOf = (v) => plot.bottom - ((v - min) / (max - min)) * (plot.bottom - plot.top);
      slice.forEach((row, i) => {
        const value = data.value(row);
        const x = plot.left + (i + 0.5) * cw;
        const y = yOf(value);
        const rising = value >= 0 ? row.c >= row.o : row.c < row.o;
        ctx.fillStyle = rising ? p.up : p.down;
        ctx.globalAlpha = 0.6;
        ctx.fillRect(Math.round(x - width / 2), Math.min(y, zero), Math.max(1, Math.round(width)), Math.max(1, Math.abs(zero - y)));
        ctx.globalAlpha = 1;
      });
      ctx.beginPath();
      slice.forEach((row, i) => {
        const x = plot.left + (i + 0.5) * cw;
        const y = yOf(data.value(row));
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = p.accent;
      ctx.lineWidth = 1.2;
      ctx.stroke();
      this._crosshair(ctx, plot, slice, view, cw, yOf, "bars");
    }

    _crosshair(ctx, plot, slice, view, cw, yOf, mode) {
      if (this.hover < 0 || this.hover >= slice.length) return;
      const p = this.palette;
      const index = this.hover;
      const x = plot.left + (index + 0.5) * cw;
      const row = slice[index];
      const value = mode === "line" ? row : (mode === "k" ? row.c : this.data.value(row));
      if (!isFinite(value)) return;
      const y = yOf(value);
      ctx.save();
      ctx.strokeStyle = p.cross;
      ctx.globalAlpha = 0.45;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(Math.round(x) + 0.5, plot.top);
      ctx.lineTo(Math.round(x) + 0.5, plot.bottom);
      ctx.moveTo(plot.left, Math.round(y) + 0.5);
      ctx.lineTo(plot.right, Math.round(y) + 0.5);
      ctx.stroke();
      ctx.restore();
      const text = this._format(value, 1);
      ctx.font = "10.5px 'Bahnschrift','Consolas',monospace";
      const w = ctx.measureText(text).width + 10;
      ctx.fillStyle = "#2b3240";
      ctx.fillRect(plot.right + 1, y - 8, w, 16);
      ctx.fillStyle = "#e9edf4";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(text, plot.right + 6, y);
      const label = "步 " + compact(mode === "line"
        ? this.data.points.step[view.start + index]
        : row.t);
      const lw = ctx.measureText(label).width + 12;
      ctx.fillStyle = "#2b3240";
      ctx.fillRect(x - lw / 2, plot.bottom + 1, lw, 14);
      ctx.fillStyle = "#e9edf4";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(label, x, plot.bottom + 4);
    }

    // ---------------------------------------------------------------- 交互
    _indexAt(clientX) {
      const rect = this.canvas.getBoundingClientRect();
      const plot = this._plot();
      const view = this.visible();
      const cw = (plot.right - plot.left) / view.count;
      const x = clientX - rect.left - plot.left;
      const index = Math.floor(x / cw);
      return { index, rect, plot, view, cw };
    }

    _bind() {
      const canvas = this.canvas;
      canvas.addEventListener("wheel", (event) => {
        if (!this.length()) return;
        event.preventDefault();
        const { index, plot, view, cw } = this._indexAt(event.clientX);
        const anchor = Math.max(0, Math.min(view.count - 1, index));
        const factor = event.deltaY > 0 ? 1.18 : 0.85;
        const next = Math.round(Math.max(6, Math.min(this.length(), view.count * factor)));
        const ratio = view.count ? anchor / view.count : 0.5;
        this.view.count = next;
        this.view.start = Math.round(view.start + anchor - next * ratio);
        this.clampView();
        this._updateHover(event.clientX, event.clientY);
        this.draw();
      }, { passive: false });

      canvas.addEventListener("pointerdown", (event) => {
        this.dragging = { x: event.clientX, start: this.view.start, cw: this._indexAt(event.clientX).cw };
        canvas.setPointerCapture(event.pointerId);
      });
      canvas.addEventListener("pointermove", (event) => {
        if (this.dragging) {
          const moved = Math.round((this.dragging.x - event.clientX) / this.dragging.cw);
          this.view.start = this.dragging.start + moved;
          this.clampView();
        }
        this._updateHover(event.clientX, event.clientY);
        this.draw();
      });
      canvas.addEventListener("pointerup", (event) => {
        this.dragging = null;
        try { canvas.releasePointerCapture(event.pointerId); } catch (error) { /* 已释放 */ }
      });
      canvas.addEventListener("pointerleave", () => {
        this.hover = -1;
        if (this.options.onLeave) this.options.onLeave();
        this.draw();
      });
      canvas.addEventListener("dblclick", () => this.reset());
    }

    _updateHover(clientX, clientY) {
      const { index, rect, view } = this._indexAt(clientX);
      if (index < 0 || index >= view.count) {
        this.hover = -1;
        if (this.options.onLeave) this.options.onLeave();
        return;
      }
      this.hover = index;
      if (this.options.onHover) {
        let row;
        if (this.data.mode === "line") {
          const absolute = view.start + index;
          row = { v: this.data.points.y[absolute], step: this.data.points.step[absolute] };
        } else {
          row = this.data.candles[view.start + index];
        }
        this.options.onHover(row, index, clientX - rect.left, clientY - rect.top);
      }
    }
  }

  // ------------------------------------------------------------------ 工具
  const VOLUME_LABELS = { range: "振幅", std: "标准差", count: "点数", rel: "相对变化" };

  function average(rows, end, n) {
    if (end < n - 1) return null;
    let total = 0;
    for (let i = end - n + 1; i <= end; i++) total += rows[i].c;
    return total / n;
  }

  function rangeOf(rows) {
    let min = Infinity;
    let max = -Infinity;
    for (const row of rows) {
      if (row.l < min) min = row.l;
      if (row.h > max) max = row.h;
    }
    return max - min || 1;
  }

  function compact(value) {
    const abs = Math.abs(value);
    if (abs >= 1e6) return (value / 1e6).toFixed(1) + "M";
    if (abs >= 1e4) return (value / 1e3).toFixed(1) + "k";
    return String(Math.round(value));
  }

  function hexAlpha(hex, alpha) {
    const clean = hex.replace("#", "");
    const r = parseInt(clean.slice(0, 2), 16);
    const g = parseInt(clean.slice(2, 4), 16);
    const b = parseInt(clean.slice(4, 6), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  window.TerminalChart = TerminalChart;
  window.ChartUtils = { compact, hexAlpha, average };
})();
