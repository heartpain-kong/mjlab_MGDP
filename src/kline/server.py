"""只监听 127.0.0.1 的本地行情服务：静态页面 + 几个只读 JSON 接口。

安全边界和现有的训练工作台一致：只绑本机地址、校验 Host 头、静态文件白名单、
CSP 只允许自身资源。这里没有任何写文件 / 起进程的接口，最坏情况就是读一堆
event 文件 —— 所以不需要 token。
"""

from __future__ import annotations

import json
import math
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import aggregate
from .labels import GROUPS, KEY_TAGS, TAGS, describe, describe_all
from .store import Store

WEB = Path(__file__).parent / "web"
ASSETS = {
  "/": ("index.html", "text/html; charset=utf-8"),
  "/index.html": ("index.html", "text/html; charset=utf-8"),
  "/assets/app.css": ("app.css", "text/css; charset=utf-8"),
  "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
  "/assets/chart.js": ("chart.js", "text/javascript; charset=utf-8"),
  "/favicon.svg": ("favicon.svg", "image/svg+xml"),
}
CACHE_CONTROL = "no-store"
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
       "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")


def create_server(root, logdirs=None, port=8766, host="127.0.0.1", cache_dir=None):
  if host not in {"127.0.0.1", "localhost"}:
    raise ValueError("本地面板只允许绑定 127.0.0.1 或 localhost")
  root = Path(root).resolve()
  # 工程自己的 logs/rsl_rl 永远在列表里；--logdir 是「再加几个」而不是「换成这些」，
  # 否则把归档目录加进来之后，新开的训练反而看不见了。
  default_logdir = root / "logs" / "rsl_rl"
  paths = [default_logdir] if default_logdir.is_dir() else []
  for path in logdirs or []:
    path = Path(path).expanduser().resolve()
    if path not in paths:
      paths.append(path)
  if not paths:
    paths = [default_logdir]
  logdirs = paths
  store = Store(logdirs, Path(cache_dir).resolve() if cache_dir else root / "logs" / "kline_cache")
  store.scan()

  def flag_tensorboard():
    try:
      import tensorboard  # noqa: F401
      return True
    except ImportError:
      return False

  def state_payload():
    if time.time() - store.last_scan > 5:
      store.scan()
    runs = []
    for run in store.ordered():
      runs.append({
        "id": run.id,
        "label": run.label,
        "stamp": run.stamp,
        "experiment": run.experiment,
        "path": str(run.path),
        "state": run.state,
        "error": run.error,
        "size": run.size,
        "mtime": run.mtime,
        "running": run.running,
        "points": run.points,
        "first_step": run.bounds()[0],
        "last_step": run.bounds()[1],
        "load_seconds": run.load_seconds,
      })
    return {
      "runs": runs,
      "roots": [{"path": str(path), "exists": path.is_dir()} for path in store.roots],
      "cache": str(store.cache_dir),
      "has_tensorboard": flag_tensorboard(),
      "key_tags": list(KEY_TAGS),
      "groups": [{"key": key, "label": label} for key, label, _ in GROUPS],
      "generated": time.time(),
    }

  def series_payload(run, tag, mode, bucket, max_points):
    series = run.series.get(tag)
    if series is None:
      raise KeyError(f"这个轮次没有指标 {tag}")
    if mode == "line":
      points = aggregate.trend(series, max_points=max_points)
      # 基准用**原始**首值：分时曲线是抽稀后的均值，第一点会和真实起点差一点点，
      # 以它为分母算涨跌幅很容易因为接近 0 而炸成几万倍。
      first = next((float(value) for value in series.value if math.isfinite(value)), None)
      data = {"mode": "line", "points": points, "base": first}
    else:
      bucket = max(1, int(bucket))
      rows = aggregate.candles(series, bucket)
      data = {
        "mode": "k",
        "candles": [
          {
            "t": row["t"], "s": row["s"], "e": row["e"],
            "o": row["o"], "h": row["h"], "l": row["l"], "c": row["c"],
            "n": row["n"], "mean": row["mean"], "std": row["std"],
            "vol": row["vol"], "w0": row["w0"], "w1": row["w1"],
          }
          for row in rows
        ],
        "base": rows[0]["o"] if rows else None,
        "bucket": bucket,
      }
    data["tag"] = tag
    data["summary"] = aggregate.summary(series)
    data["tail"] = [
      [int(series.step[index]), float(series.value[index])]
      for index in range(max(0, len(series) - 240), len(series))
      if math.isfinite(series.value[index])
    ]
    data["run"] = run.id
    return data

  class Handler(BaseHTTPRequestHandler):
    server_version = "LgDogKline/1.0"

    def log_message(self, *args):  # 别把每个请求都打到终端
      pass

    def valid_host(self):
      try:
        address = urlparse("http://" + self.headers.get("Host", ""))
        return address.hostname in {"127.0.0.1", "localhost"} and address.port == self.server.server_port
      except ValueError:
        return False

    def send(self, value, status=200, content_type="application/json; charset=utf-8", headers=None):
      if isinstance(value, (dict, list)):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
      elif isinstance(value, str):
        body = value.encode("utf-8")
      else:
        body = value
      self.send_response(status)
      self.send_header("Content-Type", content_type)
      self.send_header("Content-Length", str(len(body)))
      self.send_header("Cache-Control", CACHE_CONTROL)
      self.send_header("X-Content-Type-Options", "nosniff")
      self.send_header("Content-Security-Policy", CSP)
      for key, item in (headers or {}).items():
        self.send_header(key, item)
      self.end_headers()
      if self.command != "HEAD":
        self.wfile.write(body)

    def fail(self, message, status=400):
      return self.send({"error": message}, status)

    def do_HEAD(self):
      self.do_GET()

    def do_GET(self):
      if not self.valid_host():
        return self.fail("只接受本机地址访问", 403)
      parsed = urlparse(self.path)
      query = parse_qs(parsed.query)
      try:
        if parsed.path in ASSETS:
          filename, mime = ASSETS[parsed.path]
          target = (WEB / filename).resolve()
          if WEB.resolve() not in target.parents and target.parent != WEB.resolve():
            return self.fail("页面不存在", 404)
          return self.send(target.read_bytes(), content_type=mime)
        if parsed.path == "/api/state":
          return self.send(state_payload())
        if parsed.path == "/api/rescan":
          store.scan()
          return self.send({"runs": len(store.runs), "roots": [str(p) for p in store.roots]})
        if parsed.path == "/api/dict":
          return self.send({
            "entries": describe_all(sorted(TAGS)),
            "groups": [{"key": key, "label": label} for key, label, _ in GROUPS],
          })
        if parsed.path == "/api/run":
          return self.send(self.run_payload(_one(query, "id")))
        if parsed.path == "/api/series":
          return self.send(self.series_endpoint(query))
        if parsed.path == "/api/overview":
          return self.send(self.overview_payload(_csv_query(query, "tags")))
        if parsed.path == "/api/health":
          return self.send(self.health_payload(_one(query, "id")))
        if parsed.path == "/api/export":
          return self.export_endpoint(query)
        return self.fail("页面不存在", 404)
      except KeyError as error:
        return self.fail(str(error).strip("'\""), 404)
      except (ValueError, OSError) as error:
        return self.fail(str(error), 400)
      except Exception as error:  # 兜底：别让一个奇怪的 event 文件把连接直接掐掉
        return self.fail(f"服务出错：{type(error).__name__}: {error}", 500)

    # ---- 各接口 -----------------------------------------------------------
    def resolve(self, run_id):
      run = store.get(run_id)
      if run is None:
        raise KeyError("找不到这个轮次，可能目录已经被移走了")
      return run

    def loaded(self, run_id, tag=None):
      run = store.ensure(run_id)
      if run.state != "ready":
        return None, run
      return run, run

    def run_payload(self, run_id):
      run = self.resolve(run_id)
      store.ensure(run.id)
      meta = run.meta()
      tags = describe_all(meta["tags"])
      first, last, _wall0, _wall1 = run.bounds()
      per_tag = max(1, run.points // max(1, len(meta["tags"])))
      plan = aggregate.window_plan(first or 0, last or 0, per_tag)
      default_tag = "Train/mean_reward" if "Train/mean_reward" in meta["tags"] else (meta["tags"][0] if meta["tags"] else "")
      default_sub = "Loss/value" if "Loss/value" in meta["tags"] else (meta["tags"][1] if len(meta["tags"]) > 1 else "")
      meta["tag_info"] = tags
      meta["plan"] = plan
      meta["default_tag"] = default_tag
      meta["default_sub"] = default_sub
      return meta

    def series_endpoint(self, query):
      run_id = _one(query, "id")
      tag = _one(query, "tag")
      mode = _one(query, "mode", "k")
      bucket = int(_one(query, "bucket", "100"))
      max_points = int(_one(query, "max_points", "2400"))
      run = self.resolve(run_id)
      store.ensure(run.id)
      if run.state != "ready":
        return {"state": run.state, "error": run.error, "run": run.id, "tag": tag, "mode": mode}
      data = series_payload(run, tag, mode, bucket, max_points)
      data["state"] = "ready"
      data["running"] = run.running
      return data

    def overview_payload(self, tags):
      tags = tags or list(KEY_TAGS)
      rows = []
      for run in store.ordered():
        store.ensure(run.id)
        values = {}
        if run.state == "ready":
          for tag in tags:
            if tag not in run.series:
              continue
            if tag not in run.summary:
              run.summary[tag] = aggregate.summary(run.series[tag])
            values[tag] = run.summary[tag]
        rows.append({
          "id": run.id,
          "label": run.label,
          "stamp": run.stamp,
          "experiment": run.experiment,
          "state": run.state,
          "running": run.running,
          "mtime": run.mtime,
          "values": values,
        })
      return {"tags": tags, "runs": rows}

    def health_payload(self, run_id):
      run = self.resolve(run_id)
      store.ensure(run.id)
      if run.state != "ready":
        return {"state": run.state, "error": run.error, "run": run.id, "items": []}
      items = []
      for tag in run.meta()["tags"]:
        if tag not in run.health:
          run.health[tag] = aggregate.diagnose(run.series[tag], tag)
        findings = run.health[tag]
        if findings and findings[0]["level"] != "ok":
          info = describe(tag)
          items.append({"tag": tag, "label": info["label"], "group": info["group_label"], "findings": findings})
      return {"state": "ready", "run": run.id, "checked": len(run.meta()["tags"]), "items": items}

    def export_endpoint(self, query):
      run_id = _one(query, "id")
      tag = _one(query, "tag")
      bucket = int(_one(query, "bucket", "100"))
      run = self.resolve(run_id)
      store.ensure(run.id, wait=True)
      if run.state != "ready":
        raise ValueError(run.error or "这个轮次还没读出来")
      rows = aggregate.candles(run.series[tag], bucket)
      csv = aggregate.to_csv(run.label, tag, rows)
      name = f"kline_{_safe(run.label)}_{_safe(tag)}_b{bucket}.csv"
      return self.send(csv, content_type="text/csv; charset=utf-8",
                       headers={"Content-Disposition": f'attachment; filename="{name}"'})

  server = ThreadingHTTPServer((host, port), Handler)
  server.daemon_threads = True
  server.store = store
  return server


def serve(root, logdirs=None, port=8766, host="127.0.0.1", cache_dir=None, open_browser=True):
  server = create_server(root, logdirs, port, host, cache_dir)
  url = f"http://127.0.0.1:{server.server_port}"
  print(f"LgDog 曲线终端: {url}", flush=True)
  print(f"工程: {Path(root).resolve()}", flush=True)
  for path in server.store.roots:
    print(f"  数据源: {path}{'' if path.is_dir() else '（目录不存在）'}", flush=True)
  print(f"  轮次: {len(server.store.runs)}  缓存: {server.store.cache_dir}", flush=True)
  if not _has_tensorboard():
    print("⚠ 当前 Python 没有 tensorboard，页面能打开但读不了 event 文件。", flush=True)
  if open_browser:
    import webbrowser
    webbrowser.open(url)
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    print("已停止", flush=True)
  finally:
    server.server_close()


def _has_tensorboard():
  try:
    import tensorboard  # noqa: F401
    return True
  except ImportError:
    return False


def _one(query, key, default=None):
  values = query.get(key)
  if not values:
    if default is None:
      raise ValueError(f"缺少参数 {key}")
    return default
  return values[0]


def _csv_query(query, key):
  raw = query.get(key)
  if not raw:
    return []
  return [item for item in (part.strip() for part in raw[0].split(",")) if item]


def _safe(text):
  return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text))[:60]
