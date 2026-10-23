/* 曲线战士 · 前端主逻辑
 *
 * 一句话：把 /api/run + /api/series 拿到的标量曲线交给 TerminalChart 画，
 * 顺手把「行情头 / 指数条 / 自选 / 盘口 / 流水」这些格子填满。
 *
 * 状态都收在 state 里，渲染函数都从 state 读 —— 这样刷新、切轮次、切指标
 * 只需要改 state 再 renderAll()，不用到处手动改 DOM。
 */
(function () {
  "use strict";

  const SAVE_KEY = "lgdog.kline.settings";
  const KEY_TAGS_FALLBACK = [
    "Train/mean_reward",
    "Curriculum/terrain_levels/level_mean",
    "Curriculum/terrain_levels/tracking_ratio",
    "Curriculum/terrain_levels/reset_batch_failure_fraction",
    "Episode_Metrics/velocity_error_xy_mps",
    "Policy/mean_std",
    "Loss/entropy",
    "Perf/total_fps",
  ];
  const SUB_GROUPS = {
    reward: ["Episode_Reward/track_linear_velocity", "Episode_Reward/action_rate_l2", "Episode_Reward/foot_gait"],
    curriculum: ["Curriculum/terrain_levels/level_mean", "Curriculum/terrain_levels/tracking_ratio",
      "Curriculum/terrain_levels/promote_fraction", "Curriculum/terrain_levels/reset_batch_failure_fraction"],
    loss: ["Loss/value", "Loss/entropy", "Loss/surrogate", "Loss/him_velocity_mse"],
    metrics: ["Episode_Metrics/velocity_error_xy_mps", "Metrics/twist/error_vel_xy", "Episode_Metrics/mean_action_acc"],
    perf: ["Perf/total_fps", "Perf/collection_time", "Perf/learning_time"],
  };

  const state = {
    runs: [],
    roots: [],
    hasTB: true,
    runId: null,
    run: null,
    tag: null,
    subTag: null,
    subGroup: "reward",
    period: "mid",
    bucket: 100,
    volumeKind: "range",
    colorMode: "cn",
    refreshMs: 60000,
    overlay: false,
    favorites: [],
    sort: "default",
    series: null,
    subSeries: null,
    overview: null,
    health: null,
    keyTags: KEY_TAGS_FALLBACK,
    loading: false,
    replayTimer: null,
    token: 0,
    nextRefresh: 0,
  };

  const tickerScroll = { frame: 0, paused: false };
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  };

  // ------------------------------------------------------------------ 工具
  function fmt(value, digits) {
    if (value == null || !isFinite(value)) return "—";
    if (digits != null) return Number(value).toFixed(digits);
    const abs = Math.abs(value);
    if (abs >= 1e6) return (value / 1e6).toFixed(2) + "M";
    if (abs >= 1e4) return (value / 1e3).toFixed(1) + "k";
    if (abs >= 100) return value.toFixed(1);
    if (abs >= 1) return value.toFixed(2);
    if (abs >= 0.0001) return value.toFixed(4);
    if (abs === 0) return "0";
    return value.toExponential(2);
  }

  function fmtSigned(value) {
    if (value == null || !isFinite(value)) return "—";
    return (value > 0 ? "+" : "") + fmt(value);
  }

  function fmtPct(value) {
    if (value == null || !isFinite(value)) return "—";
    return (value > 0 ? "+" : "") + value.toFixed(2) + "%";
  }

  function folder(value) {
    if (value == null || !isFinite(value)) return "flat";
    return value > 0 ? "up" : value < 0 ? "down" : "flat";
  }

  function bytes(size) {
    if (!size) return "—";
    if (size > 1024 * 1024) return (size / 1024 / 1024).toFixed(1) + " MB";
    return (size / 1024).toFixed(0) + " KB";
  }

  function duration(seconds) {
    if (seconds == null || !isFinite(seconds)) return "—";
    if (seconds < 60) return seconds.toFixed(0) + " 秒";
    if (seconds < 3600) return (seconds / 60).toFixed(0) + " 分";
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.round((seconds % 3600) / 60);
    return hours + " 小时 " + minutes + " 分";
  }

  function clockTime(seconds) {
    if (!seconds) return "—";
    const date = new Date(seconds * 1000);
    const pad2 = (n) => String(n).padStart(2, "0");
    return `${pad2(date.getMonth() + 1)}-${pad2(date.getDate())} ${pad2(date.getHours())}:${pad2(date.getMinutes())}`;
  }

  function compactStep(value) {
    const abs = Math.abs(value || 0);
    if (abs >= 1e4) return (value / 1e3).toFixed(1) + "k";
    return String(Math.round(value || 0));
  }

  function toast(message, ms) {
    const node = $("toast");
    node.textContent = message;
    node.hidden = false;
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => { node.hidden = true; }, ms || 2600);
  }

  async function getJSON(url) {
    const response = await fetch(url, { cache: "no-store" });
    const data = await response.json().catch(() => ({ error: "服务返回的不是 JSON" }));
    if (!response.ok) throw new Error(data.error || ("请求失败 " + response.status));
    return data;
  }

  function setStatus(text, kind) {
    const node = $("statusLeft");
    node.textContent = text;
    node.style.color = kind === "bad" ? "#ff9a9b" : kind === "ok" ? "" : "";
    node.className = kind === "bad" ? "" : "dim";
  }

  // ------------------------------------------------------------------ 设置
  function applySaved() {
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem(SAVE_KEY) || "{}"); } catch (error) { saved = {}; }
    state.colorMode = saved.colorMode === "us" ? "us" : "cn";
    state.refreshMs = [0, 3000, 5000, 10000, 30000, 60000, 300000].includes(saved.refreshMs) ? saved.refreshMs : 60000;
    state.volumeKind = ["range", "std", "count", "rel"].includes(saved.volumeKind) ? saved.volumeKind : "range";
    state.period = ["trend", "short", "mid", "long", "full"].includes(saved.period) ? saved.period : "mid";
    state.overlay = !!saved.overlay;
    state.sort = ["default", "value", "name"].includes(saved.sort) ? saved.sort : "default";
    state.favorites = Array.isArray(saved.favorites) ? saved.favorites : [];
    state.tag = saved.tag || null;
    state.subTag = saved.subTag || null;
    state.subGroup = saved.subGroup || "reward";
    document.body.classList.toggle("us", state.colorMode === "us");
    $("colorMode").value = state.colorMode;
    $("volumeMetric").value = state.volumeKind;
    $("refreshRate").value = String(state.refreshMs);
  }

  function saveSettings() {
    try {
      localStorage.setItem(SAVE_KEY, JSON.stringify({
        colorMode: state.colorMode, refreshMs: state.refreshMs, volumeKind: state.volumeKind,
        period: state.period, overlay: state.overlay, sort: state.sort, tag: state.tag,
        subTag: state.subTag, subGroup: state.subGroup, favorites: state.favorites,
      }));
    } catch (error) { /* 隐私模式下写不进去就算了 */ }
  }

  // ------------------------------------------------------------------ 数据
  async function loadState() {
    const data = await getJSON("/api/state");
    state.runs = data.runs;
    state.roots = data.roots;
    state.hasTB = data.has_tensorboard;
    state.keyTags = data.key_tags && data.key_tags.length ? data.key_tags : KEY_TAGS_FALLBACK;
    $("wxHint").textContent = data.roots.map((root) => root.path + (root.exists ? "" : "（不存在）")).join("  ·  ");
    if (!state.hasTB) {
      setStatus("当前 Python 没有 tensorboard，读不了 event 文件", "bad");
      toast("当前 Python 环境没有 tensorboard，只能用训练环境的 Python 启动", 8000);
    }
    $("statusSrc").textContent = data.roots.length + " 个数据源 · " + state.runs.length + " 个轮次";
  }

  async function loadRun(token) {
    const meta = await getJSON(`/api/run?id=${encodeURIComponent(state.runId)}`);
    if (token !== state.token) return "stale";
    state.run = meta;
    if (meta.state === "error") {
      showLoading(meta.error || "解析失败");
      setStatus(meta.error || "解析失败", "bad");
      return "error";
    }
    if (meta.state !== "ready") {
      showLoading("正在解析 event 文件…");
      return "loading";
    }
    hideLoading();
    if (!state.tag || !meta.tags.includes(state.tag)) state.tag = meta.default_tag;
    if (!state.subTag || !meta.tags.includes(state.subTag)) state.subTag = meta.default_sub;
    state.bucket = meta.plan[state.period] || meta.plan.mid;
    fillSubSelect();
    return "ready";
  }

  function modeOf(period) {
    return period === "trend" ? "line" : "k";
  }

  async function loadSeries() {
    if (!state.runId || !state.tag) return;
    const token = state.token;
    const mode = modeOf(state.period);
    const query = `/api/series?id=${encodeURIComponent(state.runId)}&tag=${encodeURIComponent(state.tag)}` +
      `&mode=${mode}&bucket=${state.bucket}`;
    let data;
    try {
      data = await getJSON(query);
    } catch (error) {
      setStatus(error.message, "bad");
      return;
    }
    if (token !== state.token) return;
    if (data.state !== "ready") {
      showLoading("正在解析 event 文件…（首次大约 10 秒）");
      setTimeout(() => { if (token === state.token) loadSeries(); }, 1500);
      return;
    }
    hideLoading();
    state.series = data;
    const seriesKey = state.runId + "|" + state.tag + "|" + mode + "|" + state.bucket;
    mainChart.setData({
      mode: data.mode,
      seriesKey,
      candles: data.candles,
      points: data.points,
      bucket: data.bucket || state.bucket,
      volumeKind: state.volumeKind,
      base: data.base,
      overlay: overlayLine(data, state.subSeries),
    });
    renderQuote();
    renderOrderbook();
    renderTape();
    renderHint();
    $("statusNext").textContent = "上次刷新 " + new Date().toLocaleTimeString("zh-CN", { hour12: false });
  }

  async function loadSub(forced) {
    const tag = forced || state.subTag;
    if (!state.runId || !tag) return;
    state.subTag = tag;
    const token = state.token;
    const mode = modeOf(state.period);
    const query = `/api/series?id=${encodeURIComponent(state.runId)}&tag=${encodeURIComponent(tag)}` +
      `&mode=${mode}&bucket=${state.bucket}`;
    let data;
    try {
      data = await getJSON(query);
    } catch (error) {
      markSubMissing(error.message);
      return;
    }
    if (token !== state.token) return;
    if (data.state !== "ready") { setTimeout(() => { if (token === state.token) loadSub(tag); }, 1500); return; }
    state.subSeries = data;
    subChart.setData({
      mode: data.mode === "line" ? "line" : "bars",
      seriesKey: state.runId + "|sub|" + tag + "|" + mode + "|" + state.bucket,
      candles: data.candles,
      points: data.points,
      value: (row) => (row && row.mean != null ? row.mean : row && row.c != null ? row.c : 0),
      base: data.base,
      label: tagLabel(tag),
      subtitle: tag,
    });
    $("subSelect").value = tag;
    $("btnOverlay").classList.toggle("disabled", false);
    refreshOverlay();
    markSubMissing(null);
  }

  function markSubMissing(message) {
    const hint = $("chartHint");
    if (message) hint.textContent = "副图读不到：" + message;
  }

  async function loadOverview(extraTags) {
    const tags = Array.from(new Set([...state.keyTags, state.tag, ...(extraTags || [])].filter(Boolean)));
    const data = await getJSON("/api/overview?tags=" + encodeURIComponent(tags.join(",")));
    state.overview = data;
    renderIndexStrip();
    renderWatchlist();
  }

  async function loadHealth() {
    if (!state.runId) return;
    try {
      state.health = await getJSON(`/api/health?id=${encodeURIComponent(state.runId)}`);
    } catch (error) {
      state.health = { items: [] };
    }
    renderTicker();
  }

  // ------------------------------------------------------------------ 渲染
  function tagLabel(tag) {
    const info = (state.run && state.run.tag_info || []).find((item) => item.tag === tag);
    return info ? info.label : (tag || "").split("/").pop();
  }

  function tagInfo(tag) {
    return ((state.run && state.run.tag_info) || []).find((item) => item.tag === tag) || null;
  }

  function summaryOf(tag) {
    if (!state.overview) return null;
    const row = state.overview.runs.find((item) => item.id === state.runId);
    return row && row.values ? row.values[tag] : null;
  }

  function showLoading(text) {
    $("loadingText").textContent = text;
    $("loadingMask").hidden = false;
  }

  function hideLoading() {
    $("loadingMask").hidden = true;
  }

  function renderAll() {
    renderRunInfo();
    renderQuote();
    renderIndexStrip();
    renderWatchlist();
    renderTicker();
    renderHint();
    $("qStar").classList.toggle("on", state.favorites.includes(state.runId));
    $("qStar").textContent = state.favorites.includes(state.runId) ? "★" : "☆";
    document.querySelectorAll("#periodTabs .tab").forEach((button) => {
      button.classList.toggle("active", button.dataset.period === state.period);
      const key = button.dataset.period;
      let suffix = button.querySelector("em");
      if (!suffix) {
        suffix = document.createElement("em");
        button.appendChild(suffix);
      }
      suffix.textContent = key === "trend" || !state.run || !state.run.plan
        ? "" : compactStep(state.run.plan[key] || "");
    });
    document.querySelectorAll("#subTabs .tab[data-subgroup]").forEach((button) => {
      button.classList.toggle("active", button.dataset.subgroup === state.subGroup);
    });
    $("btnOverlay").classList.toggle("active", state.overlay);
    document.body.classList.toggle("us", state.colorMode === "us");
  }

  function renderRunInfo() {
    const box = $("runInfo");
    box.innerHTML = "";
    const run = state.run;
    if (!run) { box.appendChild(el("div", "empty", "还没有选中轮次")); return; }
    const rows = [
      ["实验", run.experiment],
      ["轮次", run.label],
      ["状态", run.state === "ready" ? (run.running ? "训练中" : "已完成") : run.state === "loading" ? "读取中" : "读取失败"],
      ["迭代", run.last_step == null ? "—" : compactStep(run.last_step)],
      ["记录点", run.points ? String(run.points) : "—"],
      ["指标", run.tags ? String(run.tags.length) : "—"],
      ["日志", bytes(run.size)],
      ["开始", clockTime(run.started)],
      ["更新", clockTime(run.updated)],
      ["时长", duration(run.duration)],
      ["读取", run.load_seconds ? run.load_seconds.toFixed(1) + " 秒" : "缓存"],
      ["路径", run.path],
    ];
    for (const [name, value, wrap] of rows) {
      const row = el("div", "info-row");
      row.appendChild(el("b", "", name));
      const span = el("span", wrap ? "wrap" : "", value == null || value === "" ? "—" : String(value));
      span.title = String(value == null ? "" : value);
      row.appendChild(span);
      box.appendChild(row);
    }
  }

  function renderQuote() {
    const run = state.run;
    const data = state.series;
    if (!run) return;
    $("qName").textContent = run.label;
    $("qCode").textContent = run.experiment.replace(/^lgdog1_/, "").slice(0, 22);
    const tags = $("qTags");
    tags.innerHTML = "";
    tags.appendChild(el("span", "tag " + (run.running ? "tag-warn" : "tag-src"), run.running ? "训练中" : "归档"));
    const info = tagInfo(state.tag);
    if (info) tags.appendChild(el("span", "tag tag-info", info.group_label));
    if (info && info.direction) tags.appendChild(el("span", "tag tag-mute", info.direction === "up" ? "越大越好" : "越小越好"));
    if (state.period !== "trend" && data) tags.appendChild(el("span", "tag tag-mute", "窗口 " + compactStep(state.bucket) + " 步"));

    $("qTagName").textContent = info ? info.label : "—";
    $("qTagRaw").textContent = state.tag || "";
    if (!data || !data.summary) return;
    const summary = data.summary;
    const base = data.base != null ? data.base : summary.first;
    const last = summary.last;
    const change = last - base;
    const pct = Math.abs(base) > 1e-9 ? (change / Math.abs(base)) * 100 : null;
    $("qPrice").textContent = fmt(last);
    $("qUnit").textContent = "";
    $("qPrice").className = "qh-last " + folder(change);
    $("qChange").textContent = fmtSigned(change);
    $("qChange").className = "qh-chg " + folder(change);
    $("qPct").textContent = fmtPct(pct);
    $("qPct").className = "qh-pct " + folder(change);

    const candles = data.candles || [];
    const lastCandle = candles[candles.length - 1];
    const firstCandle = candles[0];
    const trendMode = state.period === "trend";
    const stats = [
      {
        title: "区间",
        rows: [
          [trendMode ? "首值" : "开盘", firstCandle ? fmt(firstCandle.o) : fmt(summary.first)],
          ["最高", fmt(summary.max)],
          ["最低", fmt(summary.min)],
          [trendMode ? "末值" : "收盘", fmt(last)],
          ["均值", fmt(summary.mean)],
        ],
      },
      {
        title: "对比",
        rows: [
          ["较基准", fmtSigned(change)],
          ["涨跌幅", fmtPct(pct)],
          ["基准", fmt(base)],
          ["振幅", lastCandle ? fmt(lastCandle.h - lastCandle.l) : "—"],
          ["根数", candles.length ? String(candles.length) : "—"],
        ],
      },
      {
        title: "训练",
        rows: [
          ["迭代", run.last_step == null ? "—" : compactStep(run.last_step)],
          ["记录点", summary.n ? String(summary.n) : "—"],
          ["窗口", state.period === "trend" ? "逐点" : compactStep(state.bucket) + " 步"],
          ["窗口点数", lastCandle ? String(lastCandle.n) : "—"],
          ["时长", duration(run.duration)],
        ],
      },
      {
        title: "读取",
        rows: [
          ["日志", bytes(run.size)],
          ["指标数", run.tags ? String(run.tags.length) : "—"],
          ["更新时间", clockTime(run.updated)],
          ["读取耗时", run.load_seconds ? run.load_seconds.toFixed(1) + "s" : "缓存"],
          ["状态", run.running ? "实时写入" : "归档"],
        ],
      },
    ];
    const box = $("qStats");
    box.innerHTML = "";
    for (const group of stats) {
      const grid = el("div", "qg");
      for (const [name, value] of group.rows) {
        const cell = el("div", "qs");
        cell.appendChild(el("b", "", name));
        const span = el("span", "");
        span.textContent = value;
        if (/^较基准|^涨跌幅/.test(name)) span.className = folder(change);
        cell.appendChild(span);
        grid.appendChild(cell);
      }
      box.appendChild(grid);
    }
  }

  function renderIndexStrip() {
    const list = $("indexList");
    list.innerHTML = "";
    const row = state.overview && state.overview.runs.find((item) => item.id === state.runId);
    if (!row) { list.appendChild(el("span", "idx-empty", "等待这个轮次的数据…")); return; }
    if (row.state !== "ready") { list.appendChild(el("span", "idx-empty", "正在读取…")); return; }
    for (const tag of state.keyTags) {
      const values = row.values[tag];
      if (!values) continue;
      const info = tagInfo(tag) || { label: tag.split("/").pop() };
      const item = el("div", "strip-item" + (tag === state.tag ? " on" : ""));
      item.title = tag;
      item.appendChild(el("span", "si-name", info.label));
      const price = el("span", "si-val " + folder(values.change));
      price.textContent = fmt(values.last);
      item.appendChild(price);
      const pct = el("span", "si-chg " + folder(values.change));
      pct.textContent = fmtPct(values.pct);
      item.appendChild(pct);
      item.addEventListener("click", () => {
        if (state.tag === tag) return;
        state.tag = tag;
        saveSettings();
        loadSeries();
        renderIndexStrip();
        renderWatchlist();
        renderQuote();
        loadSub();
      });
      list.appendChild(item);
    }
    if (!list.children.length) list.appendChild(el("span", "idx-empty", "这个轮次没有关键指标"));
  }

  function sortedRuns() {
    const runs = state.runs.slice();
    const priceOf = (run) => {
      if (!state.overview) return null;
      const row = state.overview.runs.find((item) => item.id === run.id);
      const values = row && row.values && row.values[state.tag];
      return values ? values.last : null;
    };
    if (state.sort === "name") runs.sort((a, b) => a.label.localeCompare(b.label, "zh"));
    else if (state.sort === "value") runs.sort((a, b) => (priceOf(b) || -Infinity) - (priceOf(a) || -Infinity));
    else runs.sort((a, b) => b.mtime - a.mtime);
    runs.sort((a, b) => (state.favorites.includes(b.id) ? 1 : 0) - (state.favorites.includes(a.id) ? 1 : 0));
    return runs;
  }

  function renderWatchlist() {
    const box = $("watchlist");
    box.innerHTML = "";
    const all = sortedRuns();
    // 有收藏时只列收藏，但正在看的那个无论如何都要在列表里 —— 从「全部轮次」点进来一个
    // 没收藏的轮次，结果左边看不到自己，会很迷惑。
    const shown = state.favorites.length
      ? all.filter((run) => state.favorites.includes(run.id) || run.id === state.runId)
      : all;
    if (!shown.length) {
      box.appendChild(el("div", "empty", "还没有收藏的轮次。<br>点右上角的 ＋ 打开全部轮次。"));
      return;
    }
    for (const run of shown) {
      const row = el("div", "stock-row" + (run.id === state.runId ? " on" : "") +
        (run.state === "loading" ? " loading" : "") + (run.state === "error" ? " error" : ""));
      const main = el("div", "sw-main");
      const nameRow = el("div", "sw-name-row");
      const name = el("div", "sw-name", run.label);
      name.title = run.label;
      nameRow.appendChild(name);
      const star = el("span", "sw-star" + (state.favorites.includes(run.id) ? " on" : ""),
        state.favorites.includes(run.id) ? "★" : "☆");
      star.title = "收藏";
      star.addEventListener("click", (event) => {
        event.stopPropagation();
        toggleFavorite(run.id);
      });
      nameRow.appendChild(star);
      main.appendChild(nameRow);
      const sub = el("div", "sw-sub", run.stamp || run.experiment);
      sub.title = run.path;
      main.appendChild(sub);
      row.appendChild(main);

      const values = (() => {
        const found = state.overview && state.overview.runs.find((item) => item.id === run.id);
        return found && found.values ? found.values[state.tag] : null;
      })();
      const price = el("div", "sw-price " + (values ? folder(values.change) : "flat"));
      price.textContent = values ? fmt(values.last) : (run.state === "error" ? "读取失败" : "…");
      row.appendChild(price);
      const pct = el("div", "sw-pct " + (values ? "p-" + folder(values.change) : "p-flat"));
      pct.textContent = values ? fmtPct(values.pct) : "—";
      row.appendChild(pct);
      row.addEventListener("click", () => selectRun(run.id));
      box.appendChild(row);
    }
  }

  function toggleFavorite(id) {
    const index = state.favorites.indexOf(id);
    if (index >= 0) state.favorites.splice(index, 1);
    else state.favorites.push(id);
    saveSettings();
    renderWatchlist();
    renderAll();
  }

  function renderTicker() {
    const ticker = $("warnTicker");
    const items = [];
    if (state.health && state.health.items) {
      for (const item of state.health.items.slice(0, 24)) {
        for (const finding of item.findings) {
          if (finding.level === "ok") continue;
          items.push({ level: finding.level, text: `${item.label} · ${finding.title}`, detail: finding.detail, tag: item.tag });
        }
      }
    }
    if (!items.length && state.run && state.run.running) {
      items.push({ level: "info", text: "训练进行中", detail: "最后一个窗口还没收盘，那根蜡烛会继续长", tag: "" });
    }
    // 坏消息优先；同一档里关键指标优先（地形等级的一次台阶，比某个奖励项的常量占位值得看）
    const rank = { bad: 0, warn: 1, info: 2, ok: 3 };
    const keyRank = (tag) => {
      const index = state.keyTags.indexOf(tag);
      return index < 0 ? 50 : index;
    };
    items.sort((a, b) => ((rank[a.level] || 3) - (rank[b.level] || 3)) || (keyRank(a.tag) - keyRank(b.tag)));
    if (!items.length) { ticker.hidden = true; stopTickerScroll(); return; }
    ticker.hidden = false;
    ticker.classList.toggle("bad", items.some((item) => item.level === "bad"));
    $("tickerTag").textContent = items.some((item) => item.level === "bad") ? "⚠ 体检"
      : items[0].level === "warn" ? "● 提示" : "● 状态";
    const run = $("tickerRun");
    run.innerHTML = "";
    for (const item of items.slice(0, 5)) {
      const node = el("span", "ticker-it");
      node.title = item.detail;
      node.appendChild(el("b", "", item.text));
      run.appendChild(node);
    }
    startTickerScroll();
  }

  function stopTickerScroll() {
    if (tickerScroll.frame) cancelAnimationFrame(tickerScroll.frame);
    tickerScroll.frame = 0;
  }

  function startTickerScroll() {
    stopTickerScroll();
    const view = $("tickerView");
    const run = $("tickerRun");
    const step = () => {
      const max = run.scrollWidth - view.clientWidth;
      if (max > 0 && !tickerScroll.paused) {
        view.scrollLeft += 0.45;
        if (view.scrollLeft >= max) view.scrollLeft = 0;
      }
      tickerScroll.frame = requestAnimationFrame(step);
    };
    view.onmouseenter = () => { tickerScroll.paused = true; };
    view.onmouseleave = () => { tickerScroll.paused = false; };
    if (run.scrollWidth > view.clientWidth) tickerScroll.frame = requestAnimationFrame(step);
  }

  function renderOrderbook() {
    const box = $("orderbook");
    box.innerHTML = "";
    const candles = (state.series && state.series.candles) || [];
    if (!candles.length) {
      const hint = el("div", "empty");
      hint.innerHTML = "分时视图没有蜡烛<br>切到 K 线看盘口";
      box.appendChild(hint);
      return;
    }
    const tail = candles.slice(-5);
    let min = Infinity;
    let max = -Infinity;
    for (const candle of tail) {
      if (candle.l < min) min = candle.l;
      if (candle.h > max) max = candle.h;
    }
    const span = max - min || 1;
    const barOf = (value) => Math.max(2, Math.min(100, ((value - min) / span) * 100));
    for (let i = tail.length - 1; i >= 0; i--) {
      const candle = tail[i];
      const row = el("div", "ob-row");
      row.appendChild(el("span", "ob-label", "窗口 " + Math.round(candle.t)));
      const high = el("span", "ob-price up", fmt(candle.h));
      row.appendChild(high);
      const bar = el("span", "ob-bar");
      const fill = el("i");
      fill.style.width = barOf(candle.h) + "%";
      fill.style.background = "var(--up)";
      bar.appendChild(fill);
      row.appendChild(bar);
      row.appendChild(el("span", "ob-val", candle.n + "点"));
      box.appendChild(row);
    }
    const last = tail[tail.length - 1];
    const first = tail[0];
    const mid = el("div", "ob-mid");
    const base = state.series.base != null ? state.series.base : first.o;
    const pct = base ? ((last.c - base) / Math.abs(base)) * 100 : null;
    const now = el("span", "", "现在 ");
    now.appendChild(el("b", folder(pct), fmt(last.c)));
    mid.appendChild(now);
    const change = el("span", " " + folder(pct), "较基准 " + fmtPct(pct));
    mid.appendChild(change);
    box.appendChild(mid);
    for (let i = tail.length - 1; i >= 0; i--) {
      const candle = tail[i];
      const row = el("div", "ob-row");
      row.appendChild(el("span", "ob-label", "窗口 " + Math.round(candle.t)));
      const low = el("span", "ob-price down", fmt(candle.l));
      row.appendChild(low);
      const bar = el("span", "ob-bar");
      const fill = el("i");
      fill.style.width = barOf(candle.l) + "%";
      fill.style.background = "var(--down)";
      bar.appendChild(fill);
      row.appendChild(bar);
      row.appendChild(el("span", "ob-val", candle.n + "点"));
      box.appendChild(row);
    }
  }

  function renderTape() {
    const box = $("tape");
    box.innerHTML = "";
    const tail = (state.series && state.series.tail) || [];
    if (!tail.length) {
      box.appendChild(el("div", "empty", "还没有记录点"));
      return;
    }
    const reversed = tail.slice().reverse();
    let previous = null;
    for (const [step, value] of reversed) {
      const row = el("div", "tape-row");
      row.appendChild(el("span", "t-time", String(Math.round(step))));
      row.appendChild(el("span", "t-val " + (previous == null ? "" : folder(value - previous)), fmt(value)));
      row.appendChild(el("span", "t-amt " + (previous == null ? "" : folder(value - previous)),
        previous == null ? "—" : fmtSigned(value - previous)));
      box.appendChild(row);
      previous = value;
    }
    $("tapeHint").textContent = "最近 " + tail.length + " 条，倒序";
  }

  function renderHint() {
    const parts = [];
    if (state.period === "trend") {
      parts.push("逐点曲线 · " + (state.series && state.series.points ? state.series.points.n : 0) + " 个记录点");
    } else if (state.series && state.series.candles) {
      parts.push("窗口 " + compactStep(state.bucket) + " 步 · " + state.series.candles.length + " 根");
      parts.push("滚轮缩放 / 拖动平移 / 双击复位");
    }
    $("chartHint").textContent = parts.join("  ·  ");
  }

  function overlayLine(main, sub) {
    if (!state.overlay || !sub) return null;
    if (main.mode !== sub.mode) return null;
    if (main.mode === "line") {
      return { values: sub.points ? sub.points.y : [], label: tagLabel(state.subTag), color: "#4fc3f7" };
    }
    const byStep = new Map();
    for (const row of sub.candles || []) byStep.set(row.t, row.mean != null ? row.mean : row.c);
    const values = (main.candles || []).map((row) => {
      const value = byStep.get(row.t);
      return value == null ? null : value;
    });
    return { values, label: tagLabel(state.subTag), color: "#4fc3f7" };
  }

  function refreshOverlay() {
    if (!state.series) return;
    mainChart.setOverlay(overlayLine(state.series, state.subSeries));
  }

  // ------------------------------------------------------------------ 抽屉
  function openDrawer(id) {
    const drawer = $(id);
    if (!drawer) return;
    drawer.hidden = false;
    if (id === "runDrawer") renderRunDrawer();
    if (id === "tagDrawer") renderTagDrawer();
    if (id === "compareDrawer") renderCompareDrawer();
    if (id === "healthDrawer") renderHealthDrawer();
    if (id === "dictDrawer") renderDictDrawer();
  }

  function closeDrawers() {
    document.querySelectorAll(".drawer").forEach((drawer) => { drawer.hidden = true; });
  }

  function renderRunDrawer() {
    const body = $("drawerBody");
    body.innerHTML = "";
    const filter = ($("drawerFilter").value || "").trim().toLowerCase();
    const groups = new Map();
    for (const run of sortedRuns()) {
      const text = (run.label + " " + run.experiment + " " + run.stamp).toLowerCase();
      if (filter && !text.includes(filter)) continue;
      if (!groups.has(run.experiment)) groups.set(run.experiment, []);
      groups.get(run.experiment).push(run);
    }
    if (!groups.size) { body.appendChild(el("div", "empty", "没有匹配的轮次")); return; }
    for (const [experiment, runs] of groups) {
      body.appendChild(el("div", "dw-group", experiment));
      const grid = el("div", "dw-grid");
      for (const run of runs) {
        const item = el("div", "dw-tag" + (run.id === state.runId ? " on" : ""));
        item.appendChild(el("span", "", run.label));
        item.appendChild(el("i", "", run.running ? "训练中" : run.state === "ready" ? "归档" : run.state));
        const star = el("u", "", state.favorites.includes(run.id) ? "★" : "☆");
        star.title = "收藏";
        star.addEventListener("click", (event) => {
          event.stopPropagation();
          toggleFavorite(run.id);
          renderRunDrawer();
        });
        item.appendChild(star);
        item.addEventListener("click", () => { closeDrawers(); selectRun(run.id); });
        grid.appendChild(item);
      }
      body.appendChild(grid);
    }
  }

  function renderTagDrawer() {
    const body = $("tagDrawerBody");
    body.innerHTML = "";
    const info = state.run && state.run.tag_info;
    if (!info || !info.length) { body.appendChild(el("div", "empty", "这个轮次还没读出来")); return; }
    $("tagDrawerSub").textContent = state.run.label + " · " + info.length + " 个指标";
    const groups = new Map();
    for (const entry of info) {
      if (!groups.has(entry.group_label)) groups.set(entry.group_label, []);
      groups.get(entry.group_label).push(entry);
    }
    for (const [group, entries] of groups) {
      body.appendChild(el("div", "dw-group", group));
      const grid = el("div", "dw-grid");
      for (const entry of entries) {
        const item = el("div", "dw-tag" + (entry.tag === state.tag ? " on" : ""));
        item.title = entry.tag + "\n" + entry.desc;
        item.appendChild(el("span", "", entry.label));
        item.appendChild(el("i", "", entry.direction === "up" ? "↑好" : entry.direction === "down" ? "↓好" : ""));
        const sub = el("u", "", "副");
        sub.title = "设为副图";
        sub.addEventListener("click", (event) => {
          event.stopPropagation();
          state.subTag = entry.tag;
          saveSettings();
          loadSub(entry.tag);
          closeDrawers();
        });
        item.appendChild(sub);
        item.addEventListener("click", () => {
          closeDrawers();
          if (entry.tag !== state.tag) {
            state.tag = entry.tag;
            saveSettings();
            loadSeries();
            loadOverview();
            renderQuote();
            renderIndexStrip();
            renderWatchlist();
          }
        });
        grid.appendChild(item);
      }
      body.appendChild(grid);
    }
  }

  async function renderCompareDrawer() {
    const body = $("compareBody");
    const extra = ["Train/mean_episode_length"];
    if (!state.overview || !extra.every((tag) => state.overview.tags.includes(tag))) {
      body.innerHTML = "";
      body.appendChild(el("div", "empty", "正在读取各轮次的最后一个记录点…"));
      try { await loadOverview(extra); } catch (error) { body.innerHTML = ""; body.appendChild(el("div", "empty", error.message)); return; }
    }
    body.innerHTML = "";
    if (!state.overview) { body.appendChild(el("div", "empty", "还没有数据")); return; }
    const tags = state.overview.tags;
    const table = el("table", "compare-table");
    const head = el("thead");
    const headRow = el("tr");
    headRow.appendChild(el("th", "", "轮次"));
    headRow.appendChild(el("th", "", "迭代"));
    for (const tag of tags) headRow.appendChild(el("th", "", tagLabel(tag)));
    head.appendChild(headRow);
    table.appendChild(head);
    const bodyRows = el("tbody");
    for (const run of state.overview.runs) {
      const row = el("tr");
      const name = el("td", "");
      const cell = el("div", "cell-run");
      cell.appendChild(el("span", "", run.label));
      cell.appendChild(el("small", "", run.running ? "训练中" : (run.stamp || run.experiment)));
      name.appendChild(cell);
      row.appendChild(name);
      const values = run.values || {};
      const anyValues = Object.values(values);
      const step = anyValues.length ? anyValues[0].last_step : null;
      row.appendChild(el("td", "num", step == null ? "—" : compactStep(step)));
      for (const tag of tags) {
        const value = values[tag];
        const cellNode = el("td", "num " + (value ? folder(value.change) : ""));
        cellNode.textContent = value ? fmt(value.last) : "—";
        cellNode.title = value ? `${tag}\n最后 ${fmt(value.last)} · 首值 ${fmt(value.first)} · ${fmtPct(value.pct)}` : tag;
        row.appendChild(cellNode);
      }
      row.addEventListener("click", () => { closeDrawers(); selectRun(run.id); });
      bodyRows.appendChild(row);
    }
    table.appendChild(bodyRows);
    body.appendChild(table);
    body.appendChild(el("div", "wm-note",
      "⚠️ 跨轮次比较务必取同区间：指令范围课程在 iteration 5000 附近放宽，之后误差类指标会台阶式变差。" +
      "另外课程升级判据要求 tracking_ratio ≥ 0.75 并且存活 ≥ 18 秒，早期 level_mean = 0 是正常的。"));
  }

  function renderHealthDrawer() {
    const body = $("healthBody");
    body.innerHTML = "";
    if (!state.health) { body.appendChild(el("div", "empty", "正在体检…")); return; }
    $("healthSub").textContent = state.run ? state.run.label + " · 已检查 " + state.health.checked + " 个指标" : "";
    if (!state.health.items.length) {
      body.appendChild(el("div", "empty", "没有发现异常：点足够多、没有 NaN、也没有异常台阶。"));
      return;
    }
    for (const item of state.health.items) {
      const card = el("div", "health-item");
      const head = el("div", "hi-head");
      head.appendChild(el("span", "hi-name", item.label));
      head.appendChild(el("span", "hi-tag", item.group));
      const level = item.findings.some((finding) => finding.level === "bad") ? "bad"
        : item.findings.some((finding) => finding.level === "warn") ? "warn" : "info";
      head.appendChild(el("span", "hi-pill lv-" + level,
        level === "bad" ? "异常" : level === "warn" ? "注意" : "提示"));
      card.appendChild(head);
      for (const finding of item.findings) {
        const line = el("div", "hi-find");
        line.appendChild(el("b", "", finding.title + "："));
        line.appendChild(document.createTextNode(finding.detail));
        card.appendChild(line);
      }
      card.addEventListener("click", () => {
        closeDrawers();
        state.tag = item.tag;
        saveSettings();
        loadSeries();
        loadOverview();
      });
      card.style.cursor = "pointer";
      body.appendChild(card);
    }
  }

  async function renderDictDrawer() {
    const body = $("dictBody");
    if (!body.dataset.loaded) {
      body.innerHTML = "";
      body.appendChild(el("div", "empty", "正在读取…"));
      try {
        const data = await getJSON("/api/dict");
        body.dataset.loaded = "1";
        body.dataset.entries = JSON.stringify(data.entries);
      } catch (error) {
        body.innerHTML = "";
        body.appendChild(el("div", "empty", "读不到指标词典：" + error.message));
        return;
      }
    }
    const entries = JSON.parse(body.dataset.entries || "[]");
    const filter = ($("dictFilter").value || "").trim().toLowerCase();
    body.innerHTML = "";
    let currentGroup = "";
    let shown = 0;
    for (const entry of entries) {
      const text = (entry.label + " " + entry.tag + " " + entry.desc).toLowerCase();
      if (filter && !text.includes(filter)) continue;
      if (entry.group_label !== currentGroup) {
        currentGroup = entry.group_label;
        body.appendChild(el("div", "dict-sep", currentGroup));
      }
      const card = el("div", "dict-item");
      const head = el("div", "di-head");
      head.appendChild(el("span", "di-name", entry.label));
      head.appendChild(el("span", "di-tag", entry.tag));
      if (entry.direction) {
        head.appendChild(el("span", "di-dir", entry.direction === "up" ? "越大越好" : "越小越好"));
      }
      card.appendChild(head);
      card.appendChild(el("div", "di-desc", entry.desc));
      card.addEventListener("click", () => { closeDrawers(); selectTag(entry.tag); });
      card.style.cursor = "pointer";
      body.appendChild(card);
      shown++;
    }
    if (!shown) body.appendChild(el("div", "empty", "没有匹配的指标"));
  }

  // ------------------------------------------------------------------ 动作
  async function selectRun(id) {
    if (!id) return;
    state.token++;
    const token = state.token;
    state.runId = id;
    state.run = null;
    state.series = null;
    state.subSeries = null;
    state.health = null;
    state.overlay = false;
    $("btnOverlay").classList.remove("active");
    const known = state.runs.find((run) => run.id === id);
    $("qName").textContent = known ? known.label : "…";
    $("qPrice").textContent = "—";
    $("qChange").textContent = "—";
    $("qPct").textContent = "—";
    mainChart.setData(null);
    subChart.setData(null);
    renderAll();
    const status = await loadRun(token);
    if (token !== state.token) return;
    if (status === "loading") {
      // 还没解析完：等一会儿再来（解析在后台线程做）
      setTimeout(() => { if (token === state.token) selectRun(id); }, 1600);
      return;
    }
    if (status !== "ready") return;
    saveSettings();
    renderAll();
    await Promise.all([loadSeries(), loadSub(), loadOverview(), loadHealth()]);
    if (state.wantOverlay) {
      state.wantOverlay = false;
      state.overlay = true;
      saveSettings();
      $("btnOverlay").classList.add("active");
      refreshOverlay();
    }
  }

  function selectTag(tag) {
    if (!tag || tag === state.tag) return;
    state.tag = tag;
    saveSettings();
    loadSeries();
    loadOverview();
  }

  function setPeriod(period) {
    state.period = period;
    if (state.run) state.bucket = state.run.plan[period] || state.run.plan.mid;
    saveSettings();
    renderAll();
    loadSeries();
    loadSub();
  }

  function bindUI() {
    $("qStar").addEventListener("click", () => { if (state.runId) toggleFavorite(state.runId); });

    $("search").addEventListener("input", () => renderSearch());
    $("search").addEventListener("focus", () => renderSearch());
    document.addEventListener("click", (event) => {
      if (!event.target.closest(".search-wrap")) $("searchResults").hidden = true;
    });

    $("refreshRate").addEventListener("change", (event) => {
      state.refreshMs = Number(event.target.value);
      saveSettings();
      scheduleRefresh();
      toast(state.refreshMs ? "每 " + state.refreshMs / 1000 + " 秒刷新一次" : "改成手动刷新", 1800);
    });
    $("colorMode").addEventListener("change", (event) => {
      state.colorMode = event.target.value;
      document.body.classList.toggle("us", state.colorMode === "us");
      saveSettings();
      mainChart.setPalette(palette());
      subChart.setPalette(palette());
      renderAll();
    });
    $("volumeMetric").addEventListener("change", (event) => {
      state.volumeKind = event.target.value;
      saveSettings();
      if (state.series && mainChart.data) {
        mainChart.setData(Object.assign({}, mainChart.data, {
          volumeKind: state.volumeKind,
          seriesKey: mainChart.data.seriesKey,
        }));
      }
    });

    document.querySelectorAll("#periodTabs .tab[data-period]").forEach((button) => {
      button.addEventListener("click", () => setPeriod(button.dataset.period));
    });
    document.querySelectorAll("#subTabs .tab[data-subgroup]").forEach((button) => {
      button.addEventListener("click", () => {
        state.subGroup = button.dataset.subgroup;
        saveSettings();
        const candidates = SUB_GROUPS[state.subGroup] || [];
        const tag = candidates.find((item) => state.run && state.run.tags.includes(item)) || candidates[0];
        if (tag) loadSub(tag);
        renderAll();
      });
    });
    $("subSelect").addEventListener("change", (event) => loadSub(event.target.value));
    $("btnOverlay").addEventListener("click", () => {
      if ($("btnOverlay").classList.contains("disabled")) return;
      state.overlay = !state.overlay;
      saveSettings();
      $("btnOverlay").classList.toggle("active", state.overlay);
      refreshOverlay();
    });

    $("btnZoomReset").addEventListener("click", () => mainChart.reset());
    $("btnFull").addEventListener("click", toggleFull);
    $("btnReplay").addEventListener("click", toggleReplay);
    $("btnRefresh").addEventListener("click", () => refreshAll(true));
    $("btnReloadRun").addEventListener("click", () => { if (state.runId) selectRun(state.runId); });
    $("btnAllRuns").addEventListener("click", () => openDrawer("runDrawer"));
    $("btnSortWatch").addEventListener("click", () => {
      state.sort = state.sort === "default" ? "value" : state.sort === "value" ? "name" : "default";
      saveSettings();
      renderWatchlist();
      toast("轮次排序：" + (state.sort === "default" ? "按时间" : state.sort === "value" ? "按当前指标" : "按名称"), 1500);
    });
    $("btnRescan").addEventListener("click", async () => {
      try {
        const data = await getJSON("/api/rescan");
        toast("已重扫：" + data.runs + " 个轮次", 2000);
        await loadState();
        renderAll();
        await loadOverview();
      } catch (error) { toast(error.message, 3000); }
    });
    $("btnExport").addEventListener("click", () => {
      if (!state.runId || !state.tag) return;
      const bucket = state.period === "trend" ? 1 : state.bucket;
      window.location.href = `/api/export?id=${encodeURIComponent(state.runId)}&tag=${encodeURIComponent(state.tag)}&bucket=${bucket}`;
      toast("正在导出 CSV…", 1800);
    });
    $("btnHelp").addEventListener("click", () => openDrawer("helpDrawer"));
    $("disclaimer").addEventListener("click", () => openDrawer("aboutDrawer"));
    document.querySelectorAll("[data-drawer]").forEach((button) => {
      button.addEventListener("click", () => openDrawer(button.dataset.drawer));
    });
    document.querySelectorAll("[data-close]").forEach((button) => {
      button.addEventListener("click", closeDrawers);
    });
    document.querySelectorAll(".drawer").forEach((drawer) => {
      drawer.addEventListener("click", (event) => { if (event.target === drawer) closeDrawers(); });
    });
    $("drawerFilter").addEventListener("input", renderRunDrawer);
    $("dictFilter").addEventListener("input", () => renderDictDrawer());
    $("warnTicker").addEventListener("click", () => openDrawer("healthDrawer"));
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        closeDrawers();
        if (document.body.classList.contains("full")) toggleFull();
        return;
      }
      const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement.tagName);
      if (typing || document.querySelector(".drawer:not([hidden])")) return;
      if (event.key === "ArrowUp" || event.key === "ArrowDown") {
        const runs = sortedRuns();
        const index = runs.findIndex((run) => run.id === state.runId);
        if (index < 0) return;
        const next = runs[index + (event.key === "ArrowUp" ? -1 : 1)];
        if (next) {
          event.preventDefault();
          selectRun(next.id);
        }
      }
    });
    document.addEventListener("visibilitychange", () => { if (!document.hidden) refreshAll(false); });
  }

  function palette() {
    const style = getComputedStyle(document.body);
    const read = (name, fallback) => (style.getPropertyValue(name) || fallback).trim();
    return {
      up: read("--up", "#ff4d4f"),
      down: read("--down", "#00b578"),
      accent: read("--accent", "#f0b90b"),
    };
  }

  function renderSearch() {
    const box = $("searchResults");
    const text = $("search").value.trim().toLowerCase();
    box.innerHTML = "";
    if (!text) { box.hidden = true; return; }
    const matches = [];
    for (const run of state.runs) {
      if ((run.label + " " + run.experiment).toLowerCase().includes(text)) {
        matches.push({ kind: "run", label: run.label, sub: run.experiment, id: run.id });
      }
    }
    const info = (state.run && state.run.tag_info) || [];
    for (const entry of info) {
      if ((entry.label + " " + entry.tag).toLowerCase().includes(text)) {
        matches.push({ kind: "tag", label: entry.label, sub: entry.tag, id: entry.tag });
      }
    }
    if (!matches.length) {
      box.appendChild(el("div", "sr-empty", "没有匹配的轮次或指标"));
      box.hidden = false;
      return;
    }
    for (const match of matches.slice(0, 26)) {
      const item = el("div", "sr-item");
      item.appendChild(el("span", "sr-name", match.label));
      item.appendChild(el("span", "sr-prov", match.sub));
      item.appendChild(el("span", "sr-code", match.kind === "run" ? "轮次" : "指标"));
      item.addEventListener("click", () => {
        box.hidden = true;
        $("search").value = "";
        if (match.kind === "run") selectRun(match.id);
        else { selectTag(match.id); }
      });
      box.appendChild(item);
    }
    box.hidden = false;
  }

  function toggleFull() {
    document.body.classList.toggle("full");
    $("btnFull").textContent = document.body.classList.contains("full") ? "🗗" : "🖥️";
    setTimeout(() => { mainChart.resize(); subChart.resize(); }, 60);
  }

  function toggleReplay() {
    const button = $("btnReplay");
    if (state.replayTimer) {
      clearInterval(state.replayTimer);
      state.replayTimer = null;
      button.classList.remove("on");
      mainChart.setReveal(0);
      mainChart.reset();
      return;
    }
    const data = state.series;
    if (!data) { toast("还没有数据可以回放", 1800); return; }
    const total = data.mode === "line" ? (data.points ? data.points.y.length : 0) : (data.candles || []).length;
    if (!total) return;
    button.classList.add("on");
    let shown = 1;
    state.replayTimer = setInterval(() => {
      shown = Math.min(total, shown + Math.max(1, Math.round(total / 120)));
      mainChart.setReveal(shown);
      if (shown >= total) {
        clearInterval(state.replayTimer);
        state.replayTimer = null;
        button.classList.remove("on");
        setTimeout(() => { mainChart.setReveal(0); mainChart.reset(); }, 900);
      }
    }, 60);
  }

  function fillSubSelect() {
    const select = $("subSelect");
    select.innerHTML = "";
    const info = (state.run && state.run.tag_info) || [];
    const groups = new Map();
    for (const entry of info) {
      if (!groups.has(entry.group_label)) groups.set(entry.group_label, []);
      groups.get(entry.group_label).push(entry);
    }
    for (const [group, entries] of groups) {
      const optgroup = document.createElement("optgroup");
      optgroup.label = group;
      for (const entry of entries) {
        const option = document.createElement("option");
        option.value = entry.tag;
        option.textContent = entry.label;
        optgroup.appendChild(option);
      }
      select.appendChild(optgroup);
    }
    if (state.subTag) select.value = state.subTag;
  }

  // ------------------------------------------------------------------ 刷新
  function scheduleRefresh() {
    if (scheduleRefresh.timer) clearTimeout(scheduleRefresh.timer);
    if (!state.refreshMs) {
      $("statusNext").textContent = "手动刷新";
      return;
    }
    state.nextRefresh = Date.now() + state.refreshMs;
    scheduleRefresh.timer = setTimeout(() => { refreshAll(false); }, state.refreshMs);
  }

  async function refreshAll(showToast) {
    if (state.loading) return;
    state.loading = true;
    try {
      await loadState();
      await Promise.all([loadSeries(), loadSub(), loadOverview(), loadHealth()]);
      renderAll();
      if (showToast) toast("已刷新", 1200);
    } catch (error) {
      setStatus(error.message, "bad");
    } finally {
      state.loading = false;
      scheduleRefresh();
    }
  }

  function tickClock() {
    const now = new Date();
    const pad2 = (n) => String(n).padStart(2, "0");
    $("clock").textContent = `${pad2(now.getHours())}:${pad2(now.getMinutes())}:${pad2(now.getSeconds())}`;
    $("clockDate").textContent = `${now.getFullYear()}-${pad2(now.getMonth() + 1)}-${pad2(now.getDate())}`;
    if (state.nextRefresh) {
      const left = Math.max(0, Math.round((state.nextRefresh - Date.now()) / 1000));
      $("statusNext").textContent = "下次刷新 " + left + " 秒后";
    }
  }

  // ------------------------------------------------------------------ 启动
  let mainChart = null;
  let subChart = null;
  let tooltip = null;

  function makeChart(container, isSub) {
    const canvas = document.createElement("canvas");
    container.appendChild(canvas);
    const chart = new TerminalChart(canvas, {
      defaultCount: 140,
      ticks: isSub ? 3 : 4,
      onHover: isSub ? null : (row, index, x, y) => showTooltip(row, x, y),
      onLeave: isSub ? null : hideTooltip,
    });
    if (isSub) {
      const tip = document.createElement("div");
      tip.className = "kline-tip";
      tip.hidden = true;
      container.appendChild(tip);
      chart.options.onHover = (row, index, x, y) => showSubTooltip(tip, row, x, y);
      chart.options.onLeave = () => { tip.hidden = true; };
    }
    return chart;
  }

  function showTooltip(row, x, y) {
    const box = $("mainChart");
    if (!tooltip) {
      tooltip = document.createElement("div");
      tooltip.className = "kline-tip";
      tooltip.hidden = true;
      box.appendChild(tooltip);
    }
    if (!row) { tooltip.hidden = true; return; }
    tooltip.innerHTML = "";
    const info = tagInfo(state.tag);
    tooltip.appendChild(el("div", "kt-title",
      state.period === "trend" ? "步 " + compactStep(row.step) : "窗口 " + compactStep(row.t) + "（" + row.s + "–" + row.e + "）"));
    const line = (name, value, cls) => {
      const rowNode = el("div", "kt-row");
      rowNode.appendChild(el("b", "", name));
      const span = el("span", cls || "", value);
      rowNode.appendChild(span);
      tooltip.appendChild(rowNode);
    };
    if (state.period === "trend") {
      line("值", fmt(row.v));
      line("指标", info ? info.label : "—");
    } else {
      line("开", fmt(row.o));
      line("高", fmt(row.h), "up");
      line("低", fmt(row.l), "down");
      line("收", fmt(row.c), folder(row.c - row.o));
      line("均值", fmt(row.mean));
      line("标准差", fmt(row.std));
      line("点数", String(row.n));
      tooltip.appendChild(el("div", "kt-sep"));
      line("窗口时间", clockTime(row.w0) + " → " + clockTime(row.w1));
    }
    tooltip.hidden = false;
    const maxX = box.clientWidth - tooltip.offsetWidth - 12;
    const maxY = box.clientHeight - tooltip.offsetHeight - 8;
    tooltip.style.left = Math.max(6, Math.min(maxX, x + 16)) + "px";
    tooltip.style.top = Math.max(4, Math.min(maxY, y - 10)) + "px";
  }

  function hideTooltip() {
    if (tooltip) tooltip.hidden = true;
  }

  function showSubTooltip(tip, row, x, y) {
    if (!row) { tip.hidden = true; return; }
    tip.innerHTML = "";
    const label = tagLabel(state.subTag);
    tip.appendChild(el("div", "kt-title", label + " · 步 " + compactStep(row.t)));
    const line = (name, value) => {
      const node = el("div", "kt-row");
      node.appendChild(el("b", "", name));
      node.appendChild(el("span", "", value));
      tip.appendChild(node);
    };
    if (state.period === "trend") line("值", fmt(row.v));
    else {
      line("均值", fmt(row.mean));
      line("开 / 收", fmt(row.o) + " / " + fmt(row.c));
    }
    tip.hidden = false;
    const maxX = tip.parentElement.clientWidth - tip.offsetWidth - 12;
    const maxY = tip.parentElement.clientHeight - tip.offsetHeight - 8;
    tip.style.left = Math.max(6, Math.min(maxX, x + 14)) + "px";
    tip.style.top = Math.max(4, Math.min(maxY, y - 10)) + "px";
  }

  // 深链接：?run=<id 或轮次名>&tag=<指标>&period=<trend|short|mid|long|full>&drawer=<run|tag|compare|health|dict|help|about>
  // 主要用途是把「我正在看的这张图」发给别人；顺手也方便截图核对。
  function applyDeepLink() {
    const params = new URLSearchParams(location.search);
    const tag = params.get("tag");
    if (tag) state.tag = tag;
    const sub = params.get("sub");
    if (sub) state.subTag = sub;
    if (params.get("overlay") === "1") state.wantOverlay = true;
    const period = params.get("period");
    if (["trend", "short", "mid", "long", "full"].includes(period)) state.period = period;
    const wanted = params.get("run");
    if (wanted) {
      const match = state.runs.find((run) => run.id === wanted || run.label === wanted || run.path.endsWith(wanted));
      if (match) state.preferredRun = match.id;
    }
    const drawer = params.get("drawer");
    if (drawer) setTimeout(() => openDrawer(drawer + "Drawer"), 700);
  }

  async function boot() {
    bindUI();
    applySaved();
    mainChart = makeChart($("mainChart"), false);
    subChart = makeChart($("subChart"), true);
    mainChart.setPalette(palette());
    subChart.setPalette(palette());
    tickClock();
    setInterval(tickClock, 1000);
    try {
      await loadState();
    } catch (error) {
      setStatus("连不上本地服务：" + error.message, "bad");
      return;
    }
    if (!state.runs.length) {
      setStatus("没有找到任何 event 文件，检查 logs/rsl_rl 目录", "bad");
      $("watchlist").appendChild(el("div", "empty", "没有找到训练轮次。<br>确认 logs/rsl_rl 下有 events.out.tfevents* 文件，<br>或者用 --logdir 指定别的目录后重启。"));
      return;
    }
    applyDeepLink();
    const preferred = state.preferredRun || state.favorites[0] || null;
    const initial = state.runs.find((run) => run.id === preferred) || state.runs[0];
    await selectRun(initial.id);
    scheduleRefresh();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
