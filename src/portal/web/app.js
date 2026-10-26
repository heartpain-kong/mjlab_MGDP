/* mjlab 总览 —— 只做三件事：把 /api/state 画出来、按钮调 /api/service、日志尾巴轮询 /api/log。 */
(function () {
  "use strict";

  const KEY = "mjlab.portal.refresh";
  const state = {
    token: null,
    data: null,
    refreshMs: 5000,
    logSource: "service:dashboard",
    follow: true,
    timer: null,
    logTimer: null,
    busy: {},
  };

  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  };

  function bytes(value) {
    if (!value) return "0 B";
    if (value > 1024 ** 3) return (value / 1024 ** 3).toFixed(1) + " GB";
    if (value > 1024 ** 2) return (value / 1024 ** 2).toFixed(0) + " MB";
    return (value / 1024).toFixed(0) + " KB";
  }

  function num(value, digits) {
    if (value == null || !isFinite(value)) return "—";
    if (digits != null) return Number(value).toFixed(digits);
    if (value === 0) return "0";
    const abs = Math.abs(value);
    if (abs >= 1e6) return (value / 1e6).toFixed(2) + "M";
    if (abs >= 1e4) return (value / 1e3).toFixed(1) + "k";
    if (abs >= 100) return value.toFixed(1);
    if (abs >= 1) return value.toFixed(2);
    if (abs >= 0.001) return value.toFixed(4);
    return value.toExponential(2);
  }

  function intNum(value) {
    if (value == null || !isFinite(value)) return "—";
    return Math.round(value).toLocaleString("zh-CN");
  }

  function ago(seconds) {
    if (seconds == null || !isFinite(seconds)) return "—";
    if (seconds < 60) return Math.max(0, Math.round(seconds)) + " 秒前";
    if (seconds < 3600) return Math.round(seconds / 60) + " 分钟前";
    if (seconds < 86400) return (seconds / 3600).toFixed(1) + " 小时前";
    return (seconds / 86400).toFixed(1) + " 天前";
  }

  function duration(seconds) {
    if (seconds == null || !isFinite(seconds)) return "—";
    if (seconds < 60) return Math.round(seconds) + " 秒";
    if (seconds < 3600) return Math.round(seconds / 60) + " 分钟";
    return Math.floor(seconds / 3600) + " 小时 " + Math.round((seconds % 3600) / 60) + " 分";
  }

  function clock(seconds) {
    if (!seconds) return "—";
    const date = new Date(seconds * 1000);
    const pad = (n) => String(n).padStart(2, "0");
    return `${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
  }

  function toast(message) {
    const node = $("toast");
    node.textContent = message;
    node.classList.remove("hidden");
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => node.classList.add("hidden"), 2800);
  }

  async function getJSON(url) {
    const response = await fetch(url, { cache: "no-store" });
    const data = await response.json().catch(() => ({ error: "返回的不是 JSON" }));
    if (!response.ok) throw new Error(data.error || ("请求失败 " + response.status));
    return data;
  }

  // ---------------------------------------------------------------- 数据
  async function load() {
    try {
      const data = await getJSON("/api/state");
      state.data = data;
      state.token = data.token;
      render();
    } catch (error) {
      toast("读取失败：" + error.message);
    }
    schedule();
  }

  function schedule() {
    clearTimeout(state.timer);
    state.timer = setTimeout(load, state.refreshMs);
  }

  // ---------------------------------------------------------------- 渲染
  function render() {
    const data = state.data;
    if (!data) return;
    $("host").textContent = `${data.environment.hostname} · ${data.environment.platform}`;
    $("root").textContent = data.root;
    $("footer-dirs").textContent = data.runs.logdirs && data.runs.logdirs.length
      ? "数据源：" + data.runs.logdirs.join("  ·  ")
      : "没有找到数据源（logs/rsl_rl 或 数据/ 目录）";
    renderSummary();
    renderServices();
    renderRuns();
    renderEnvironment();
    renderGpu();
    renderProcesses();
    renderHistory();
    renderLogOptions();
  }

  function trainingProcess() {
    const items = (state.data.processes && state.data.processes.items) || [];
    return items.find((item) => item.role_key === "train") || null;
  }

  function renderSummary() {
    const data = state.data;
    const train = trainingProcess();
    const liveRuns = (data.runs.items || []).filter((run) => run.running);
    if (train) {
      $("stat-training").textContent = "训练中";
      $("stat-training").style.color = "var(--teal)";
      $("stat-training-detail").textContent = `PID ${train.pid} · 已跑 ${duration(train.uptime)}` +
        (train.cpu_percent != null ? ` · CPU ${train.cpu_percent}%` : "");
    } else if (liveRuns.length) {
      $("stat-training").textContent = "有 run 在写";
      $("stat-training").style.color = "var(--teal)";
      $("stat-training-detail").textContent = liveRuns.map((run) => run.label).slice(0, 3).join(" / ");
    } else {
      $("stat-training").textContent = "空闲";
      $("stat-training").style.color = "";
      $("stat-training-detail").textContent = "没有发现 train.py 进程";
    }

    const gpu = data.gpu;
    if (gpu.available && gpu.gpus.length) {
      const first = gpu.gpus[0];
      $("stat-gpu").textContent = first.name.replace(/NVIDIA\s*/i, "").slice(0, 20);
      $("stat-gpu").style.color = "";
      const used = first.memory_used_mib;
      const total = first.memory_total_mib;
      $("stat-gpu-detail").textContent = `${used}/${total} MiB · 利用率 ${first.utilization_gpu}%` +
        (first.temperature_c != null ? ` · ${first.temperature_c}℃` : "");
    } else {
      $("stat-gpu").textContent = "无 NVIDIA GPU";
      $("stat-gpu").style.color = "var(--warn)";
      const adapters = (gpu.adapters || []).map((item) => item.name).filter((name) => !/IddDesk|Virtual|Oray/i.test(name));
      $("stat-gpu-detail").textContent = adapters.length ? adapters.join(" / ") : (gpu.reason || "");
    }

    const disk = data.resources.disk;
    if (disk) {
      $("stat-disk").textContent = disk.percent + "%";
      $("stat-disk-detail").textContent = `剩余 ${bytes(disk.free)} / 共 ${bytes(disk.total)}`;
    }
    const items = data.runs.items || [];
    const totalSize = items.reduce((sum, run) => sum + (run.size || 0), 0);
    $("stat-runs").textContent = items.length + " 轮";
    $("stat-runs-detail").textContent = items.length
      ? `${bytes(totalSize)} · 最新 ${items[0].label}（${ago((data.now - items[0].mtime))}）`
      : "归档目录是空的";
  }

  function renderServices() {
    const box = $("services");
    box.innerHTML = "";
    for (const service of state.data.services) {
      const card = el("div", "service");
      const top = el("div", "service-top");
      top.appendChild(el("span", "dot " + service.status));
      top.appendChild(el("span", "service-name", service.label));
      top.appendChild(el("span", "service-port", ":" + service.port));
      top.appendChild(el("span", "service-status", statusText(service)));
      card.appendChild(top);
      card.appendChild(el("div", "service-desc", service.description));
      const actions = el("div", "service-actions");
      const open = el("a", "open", "打开 ↗");
      open.href = service.url;
      open.target = "_blank";
      open.rel = "noopener";
      actions.appendChild(open);
      if (service.status === "running" || service.status === "external") {
        const stop = el("button", "stop", service.managed ? "停止" : "外部进程");
        stop.disabled = !service.managed;
        stop.title = service.managed ? "停止本页启动的进程" : "不是本页启动的，请在它自己的窗口停止";
        stop.addEventListener("click", () => act(service.name, "stop", stop));
        actions.appendChild(stop);
      } else {
        const start = el("button", "start", service.status === "starting" ? "启动中…" : "启动");
        start.disabled = service.status === "starting" || Boolean(state.busy[service.name]);
        start.addEventListener("click", () => act(service.name, "start", start));
        actions.appendChild(start);
      }
      card.appendChild(actions);
      box.appendChild(card);
    }
  }

  function statusText(service) {
    if (service.status === "running") return service.managed ? "运行中" : "运行中（外部）";
    if (service.status === "external") return "已在别处运行";
    if (service.status === "starting") return "启动中…";
    if (service.status === "failed") return "启动失败（返回码 " + service.returncode + "）";
    return "未启动";
  }

  async function act(name, action, button) {
    state.busy[name] = true;
    if (button) { button.disabled = true; button.textContent = action === "start" ? "启动中…" : "停止中…"; }
    try {
      const response = await fetch("/api/service", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Mjlab-Token": state.token },
        body: JSON.stringify({ name, action }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "操作失败");
      toast(`${data.service.label}：${statusText(data.service)}`);
    } catch (error) {
      toast(error.message);
    } finally {
      state.busy[name] = false;
      await load();
    }
  }

  function openKline(runId) {
    const kline = state.data.services.find((service) => service.name === "kline");
    if (!kline || (kline.status !== "running" && kline.status !== "external")) {
      toast("曲线终端没在运行：先在上面那张卡上点「启动」");
      return;
    }
    window.open(`http://127.0.0.1:${kline.port}/?run=${encodeURIComponent(runId)}`, "_blank", "noopener");
  }

  function renderRuns() {
    const table = $("runs");
    table.innerHTML = "";
    const items = state.data.runs.items || [];
    const source = state.data.runs.source;
    $("runs-hint").textContent = source === "kline"
      ? "指标来自曲线终端（已解析 event 文件）"
      : items.length ? "只读文件信息；启动曲线终端后可看到指标" : "没有找到 event 文件";
    const head = el("thead");
    const row = el("tr");
    for (const label of ["轮次", "平均回报", "迭代", "地形等级", "跟踪比", "失败率", "训练至", "大小", "曲线"]) {
      row.appendChild(el("th", "", label));
    }
    head.appendChild(row);
    table.appendChild(head);
    const body = el("tbody");
    if (!items.length) {
      const empty = el("tr");
      const cell = el("td", "empty", "没有找到训练轮次：logs/rsl_rl 和 数据/ 都是空的");
      cell.colSpan = 9;
      empty.appendChild(cell);
      body.appendChild(empty);
    }
    const value = (run, tag) => (run.values && run.values[tag] ? run.values[tag] : null);
    for (const run of items) {
      const line = el("tr", "run");
      const parsing = run.state === "loading" && state.data.runs.source === "kline";
      const nameCell = el("td");
      const name = el("div", "run-name");
      name.appendChild(el("b", "", run.label));
      const sub = run.stamp || run.experiment;
      if (sub) name.appendChild(el("small", "", sub));
      nameCell.appendChild(name);
      if (run.running) nameCell.appendChild(el("span", "badge live", "在写"));
      else if (parsing) nameCell.appendChild(el("span", "badge", "解析中"));
      line.appendChild(nameCell);
      const reward = value(run, "Train/mean_reward");
      const cell = (text) => el("td", "num", parsing && !reward ? "…" : text);
      line.appendChild(cell(reward ? num(reward.last) : "—"));
      line.appendChild(cell(reward ? intNum(reward.last_step) : "—"));
      const level = value(run, "Curriculum/terrain_levels/level_mean");
      line.appendChild(cell(level ? num(level.last) : "—"));
      const tracking = value(run, "Curriculum/terrain_levels/tracking_ratio");
      line.appendChild(cell(tracking ? num(tracking.last) : "—"));
      const failure = value(run, "Curriculum/terrain_levels/reset_batch_failure_fraction");
      line.appendChild(cell(failure ? num(failure.last) : "—"));
      const trained = reward && reward.updated ? reward.updated : run.mtime;
      line.appendChild(el("td", "num", clock(trained)));
      line.appendChild(el("td", "num", bytes(run.size)));
      const action = el("td");
      const kline = el("button", "link", "K线");
      kline.title = "在曲线终端里打开这个轮次";
      kline.addEventListener("click", (event) => {
        event.stopPropagation();
        openKline(run.id);
      });
      action.appendChild(kline);
      line.appendChild(action);
      line.addEventListener("click", () => openKline(run.id));
      line.style.cursor = "pointer";
      body.appendChild(line);
    }
    table.appendChild(body);
  }

  function renderEnvironment() {
    const env = state.data.environment;
    const box = $("env");
    box.innerHTML = "";
    $("env-hint").textContent = env.hostname;
    const rows = [
      ["Python", env.python],
      ["解释器", env.executable],
      ["torch", (env.packages.torch || "未安装") + (env.torch_cpu_only ? "（CPU 版）" : "")],
      ["mjlab", env.packages.mjlab || "未安装"],
      ["mujoco", env.packages.mujoco || "未安装"],
      ["mujoco-warp", env.packages["mujoco-warp"] || "未安装"],
      ["rsl-rl", env.packages["rsl-rl-lib"] || "未安装"],
      ["tensorboard", env.packages.tensorboard || "未安装"],
      ["CPU", env.cpu_count + " 核" + (state.data.resources.memory
        ? ` · 内存 ${Math.round(state.data.resources.memory.used / 1024 ** 3)}/${Math.round(state.data.resources.memory.total / 1024 ** 3)} GB`
        : "")],
    ];
    for (const [key, value] of rows) {
      box.appendChild(el("dt", "", key));
      const dd = el("dd", "", value);
      dd.title = value;
      box.appendChild(dd);
    }
  }

  function renderGpu() {
    const gpu = state.data.gpu;
    const box = $("gpu");
    box.innerHTML = "";
    if (gpu.available) {
      $("gpu-hint").textContent = gpu.source || "";
      for (const item of gpu.gpus) {
        const wrap = el("div", "kv");
        wrap.appendChild(el("dt", "", "型号"));
        wrap.appendChild(el("dd", "", item.name));
        wrap.appendChild(el("dt", "", "显存"));
        wrap.appendChild(el("dd", "", `${item.memory_used_mib} / ${item.memory_total_mib} MiB（剩 ${item.memory_free_mib}）`));
        wrap.appendChild(el("dt", "", "利用率"));
        wrap.appendChild(el("dd", "", (item.utilization_gpu != null ? item.utilization_gpu : "—") + " %"));
        wrap.appendChild(el("dt", "", "温度 / 功耗"));
        wrap.appendChild(el("dd", "", `${item.temperature_c ?? "—"} ℃ / ${item.power_w ?? "—"} W`));
        box.appendChild(wrap);
      }
      const users = state.data.gpu_processes || [];
      if (users.length) {
        box.appendChild(el("div", "hint", "占用显存的进程：" + users.map((p) => `${p.pid} ${p.name} ${p.memory_mib}MiB`).join(" · ")));
      }
    } else {
      $("gpu-hint").textContent = "nvidia-smi 不可用";
      const note = el("div", "hint", gpu.reason || "没有找到 GPU");
      box.appendChild(note);
      const adapters = (gpu.adapters || []).filter((item) => !/IddDesk|Virtual|Oray/i.test(item.name));
      if (adapters.length) {
        const kv = el("div", "kv");
        for (const adapter of adapters) {
          kv.appendChild(el("dt", "", "显卡"));
          kv.appendChild(el("dd", "", adapter.name + (adapter.driver ? `（驱动 ${adapter.driver}）` : "")));
        }
        box.appendChild(kv);
      }
      box.appendChild(el("div", "hint", "mjlab 训练需要 CUDA；这台机器上更适合「看」而不是「训」。"));
    }
  }

  function renderProcesses() {
    const box = $("procs");
    box.innerHTML = "";
    const data = state.data.processes;
    if (!data.available) {
      $("proc-hint").textContent = "缺 psutil";
      box.appendChild(el("div", "empty", data.reason || "读不到进程列表"));
      return;
    }
    $("proc-hint").textContent = data.items.length ? data.items.length + " 个" : "无";
    if (!data.items.length) {
      box.appendChild(el("div", "empty", "没有发现训练 / 回放 / Sim2Sim / 面板进程"));
      return;
    }
    for (const item of data.items) {
      const card = el("div", "proc");
      const top = el("div", "proc-top");
      top.appendChild(el("span", "proc-role", item.role));
      top.appendChild(el("span", "proc-meta",
        `PID ${item.pid} · ${duration(item.uptime)}` +
        (item.cpu_percent != null ? ` · CPU ${item.cpu_percent}%` : "") +
        (item.memory_mb != null ? ` · ${item.memory_mb} MB` : "")));
      card.appendChild(top);
      const cmd = el("div", "proc-cmd", item.command || item.name || "");
      cmd.title = item.command || "";
      card.appendChild(cmd);
      box.appendChild(card);
    }
  }

  function renderHistory() {
    const box = $("history");
    box.innerHTML = "";
    const runs = (state.data.history && state.data.history.runs) || [];
    if (!runs.length) {
      box.appendChild(el("div", "empty", "还没有面板启动过的记录"));
      return;
    }
    for (const run of runs.slice(0, 8)) {
      const item = el("div", "hist");
      const top = el("div", "hist-top");
      top.appendChild(el("span", "hist-mode", run.mode || "?"));
      const chip = el("span", "status-chip " + (run.status || ""), run.status || "未知");
      top.appendChild(chip);
      top.appendChild(el("span", "hist-time", (run.started || "").replace("T", " ").slice(5, 16)));
      item.appendChild(top);
      const detail = [run.task, run.run_name, run.envs ? run.envs + " env" : "", run.iterations ? run.iterations + " 轮" : "",
        run.duration_s ? duration(run.duration_s) : ""].filter(Boolean).join(" · ");
      item.appendChild(el("div", "hist-detail", detail || run.command || ""));
      box.appendChild(item);
    }
  }

  function renderLogOptions() {
    const select = $("log-source");
    const signature = JSON.stringify(state.data.services.map((s) => s.name + ":" + s.status)) +
      JSON.stringify((state.data.logs || []).map((log) => log.path + ":" + log.size));
    if (select.dataset.signature === signature) return;
    select.dataset.signature = signature;
    const current = select.value || state.logSource;
    select.innerHTML = "";
    addServiceOptions(select);
    addFileOptions(select);
    const values = Array.from(select.options).map((option) => option.value);
    select.value = values.includes(current) ? current : (values[0] || "");
    state.logSource = select.value;
  }

  function addServiceOptions(select) {
    const group = document.createElement("optgroup");
    group.label = "服务日志";
    for (const service of state.data.services) {
      const option = document.createElement("option");
      option.value = "service:" + service.name;
      option.textContent = service.label + "（: " + service.port + "）";
      group.appendChild(option);
    }
    select.appendChild(group);
  }

  function addFileOptions(select) {
    const logs = state.data.logs || [];
    if (!logs.length) return;
    const group = document.createElement("optgroup");
    group.label = "logs/ 下的文件";
    for (const log of logs) {
      const option = document.createElement("option");
      option.value = "file:" + log.path;
      option.textContent = log.name + "（" + bytes(log.size) + "）";
      group.appendChild(option);
    }
    select.appendChild(group);
  }

  async function loadLog() {
    if (document.hidden) return;
    const source = state.logSource;
    const pre = $("log");
    if (!source) return;
    try {
      let data;
      if (source.startsWith("service:")) {
        data = await getJSON("/api/log?name=" + encodeURIComponent(source.slice(8)) + "&lines=120");
      } else {
        data = await getJSON("/api/log?path=" + encodeURIComponent(source.slice(5)) + "&lines=160");
      }
      const text = data.text || "";
      $("log-hint").textContent = data.exists === false ? "文件还没生成（服务没启动过）" : bytes(data.size) + " · 只显示尾部";
      if (pre.textContent !== text) {
        pre.textContent = text || "（空）";
        if (state.follow) pre.scrollTop = pre.scrollHeight;
      }
    } catch (error) {
      $("log-hint").textContent = error.message;
    }
  }

  // ---------------------------------------------------------------- 启动
  function bind() {
    const saved = Number(localStorage.getItem(KEY));
    if ([2000, 5000, 15000, 30000].includes(saved)) state.refreshMs = saved;
    $("refresh").value = String(state.refreshMs);
    $("refresh").addEventListener("change", (event) => {
      state.refreshMs = Number(event.target.value);
      localStorage.setItem(KEY, String(state.refreshMs));
      schedule();
    });
    $("reload").addEventListener("click", () => load());
    $("log-follow").addEventListener("change", (event) => { state.follow = event.target.checked; });
    $("log-source").addEventListener("change", (event) => {
      state.logSource = event.target.value;
      loadLog();
    });
    const tick = () => {
      const now = new Date();
      const pad = (n) => String(n).padStart(2, "0");
      $("clock").textContent = `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
    };
    tick();
    setInterval(tick, 1000);
    setInterval(loadLog, 2000);
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) { load(); loadLog(); }
    });
  }

  bind();
  load().then(loadLog);
})();
