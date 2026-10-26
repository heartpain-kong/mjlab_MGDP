"""把 event 文件读成标量序列，并缓存到磁盘，避免每次开页面重扫几十 MB。

设计要点（都是为了「打开就快」）：
  * event 文件一次解析、磁盘缓存一次；缓存头里记着源文件的
    (路径, mtime_ns, 大小)，任何一个变了就自动重解析，不需要手动清缓存。
  * 解析放到后台线程做，页面先拿到 ``state="loading"`` 并显示进度，
    不阻塞其它请求；大的 run（60 MB 左右）第一次要十来秒。
  * 同一个 run 只允许一个线程解析（``_lock``），解析中的请求返回同一状态。
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
import time
from array import array
from pathlib import Path

CACHE_VERSION = 3
CACHE_MAGIC = b"LGKLINE2\n"
EVENT_PREFIX = "events.out.tfevents"
# 解析中的 run：mtime 在这么久以内就算「还在训练」，页面上按未收盘的最后一根处理。
LIVE_SECONDS = 180.0
SKIP_DIRS = {"params", "git", "videos", "images", "__pycache__", ".git", "tensorboard"}


class Series:
  """一条标量曲线：step / value / wall_time 三条等长数组，元素是原生 number。"""

  __slots__ = ("step", "value", "wall")

  def __init__(self, step, value, wall):
    self.step = step
    self.value = value
    self.wall = wall

  def __len__(self):
    return len(self.value)


class Run:
  """一个训练轮次（run 目录）的元信息 + 解析结果。"""

  def __init__(self, path, root, experiment, label, stamp):
    self.path = path
    self.root = root
    self.experiment = experiment
    self.label = label
    self.stamp = stamp
    self.id = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    self.state = "idle"  # idle | loading | ready | error
    self.error = None
    self.error_at = 0.0
    self.series = {}
    self.summary = {}   # tag -> 统计（页面顶部那些数）
    self.health = {}    # tag -> 体检结论
    self.size = 0
    self.mtime = 0.0
    self.event_files = []
    self.loaded_at = None
    self.loaded_signature = None
    self.load_seconds = None

  # ---- 元信息 -------------------------------------------------------------
  @property
  def running(self):
    return (time.time() - self.mtime) < LIVE_SECONDS

  def signature(self):
    return [[str(p), int(p.stat().st_mtime_ns), int(p.stat().st_size)] for p in self.event_files]

  @property
  def points(self):
    return sum(len(s) for s in self.series.values())

  def bounds(self):
    """(首步, 末步, 开始时间, 最后时间)。

    ⚠️ ``*/time`` 这类 tag 的 step 存的是**秒**，不是迭代数。把它们算进步数范围
    会让「全程」变成几万，窗口宽度跟着算错 —— 所以步数只取普通 tag，
    wall time 还是取全部（它们本来就是同一段时间）。
    """
    first = last = None
    wall0 = wall1 = None
    plain = [(tag, series) for tag, series in self.series.items() if not tag.endswith("/time")]
    for _tag, series in (plain or list(self.series.items())):
      if not len(series):
        continue
      if first is None or series.step[0] < first:
        first = series.step[0]
      if last is None or series.step[-1] > last:
        last = series.step[-1]
    for series in self.series.values():
      if not len(series):
        continue
      if wall0 is None or series.wall[0] < wall0:
        wall0 = series.wall[0]
      if wall1 is None or series.wall[-1] > wall1:
        wall1 = series.wall[-1]
    return first, last, wall0, wall1

  def meta(self):
    first, last, wall0, wall1 = self.bounds()
    return {
      "id": self.id,
      "label": self.label,
      "stamp": self.stamp,
      "experiment": self.experiment,
      "path": str(self.path),
      "state": self.state,
      "error": self.error,
      "tags": sorted(self.series),
      "points": self.points,
      "size": self.size,
      "mtime": self.mtime,
      "first_step": first,
      "last_step": last,
      "started": wall0,
      "updated": wall1,
      "duration": (wall1 - wall0) if (wall0 and wall1) else None,
      "running": self.running,
      "load_seconds": self.load_seconds,
    }


class Store:
  """扫描若干 logdir、按需解析、带磁盘缓存。"""

  def __init__(self, roots, cache_dir):
    self.roots = [Path(r) for r in roots]
    self.cache_dir = Path(cache_dir)
    self.runs = {}
    self._lock = threading.Lock()
    self._run_locks = {}
    self._threads = {}
    self._slots = threading.Semaphore(2)
    self.last_scan = 0.0

  # ---- 扫描 ---------------------------------------------------------------
  def scan(self):
    """重新扫目录；已经解析过的 run 保留内存里的数据。

    同一个 run 可能同时出现在工程 logs/ 和归档目录里（复制过去的那份）。
    按「标签 + event 文件总大小」去重，靠前的根目录优先 —— 否则自选列表里
    会出现两行一模一样的 V8，看不出哪个是新的。
    """
    found = {}
    seen = {}
    for root in self.roots:
      if not root.is_dir():
        continue
      for path, files in _walk_runs(root):
        rel = path.relative_to(root).parts
        experiment = rel[0] if len(rel) > 1 else root.name
        label, stamp = _split_name(path.name)
        run = Run(path, root, experiment, label, stamp)
        if run.id in self.runs:
          run = self.runs[run.id]
        run.event_files = [path / name for name in files]
        run.size = sum(p.stat().st_size for p in run.event_files)
        run.mtime = max(p.stat().st_mtime for p in run.event_files)
        key = (run.label, run.size)
        if key in seen and seen[key] != run.id:
          continue
        seen[key] = run.id
        found[run.id] = run
    self.runs = found
    self.last_scan = time.time()
    return self.runs

  def ordered(self):
    """新的在前。"""
    return sorted(self.runs.values(), key=lambda r: (r.mtime, r.label), reverse=True)

  def get(self, run_id):
    run = self.runs.get(run_id)
    if run is None:
      self.scan()
      run = self.runs.get(run_id)
    return run

  # ---- 解析 ---------------------------------------------------------------
  def ensure(self, run_id, wait=False, max_age=15.0):
    """确保 run 已经解析；wait=True 时阻塞到出结果（导出/测试用）。

    训练还在写时，同一份文件最多每 ``max_age`` 秒重读一次 —— 否则每个请求
    都重扫几十 MB，页面反而卡死。
    """
    run = self.get(run_id)
    if run is None:
      return None
    if run.state == "ready":
      if not self._dirty(run, max_age):
        return run
      run.state = "idle"
    if run.state == "error":
      # 还在训练的 run 偶尔会读到半截记录，过一会儿再试；归档的就不折腾了。
      if time.time() - run.mtime > LIVE_SECONDS or (time.time() - run.error_at) < 30:
        return run
      run.state = "idle"
    if wait:
      self._load(run)
      return run
    self._load_async(run)
    return run

  def _dirty(self, run, max_age):
    """解析完之后源文件又长了？（只对还在训练的 run 检查，而且限频）"""
    if time.time() - run.mtime > LIVE_SECONDS:
      return False
    if time.time() - (run.loaded_at or 0) < max_age:
      return False
    try:
      return run.signature() != run.loaded_signature
    except OSError:
      return False

  def _run_lock(self, run_id):
    with self._lock:
      return self._run_locks.setdefault(run_id, threading.Lock())

  def _load_async(self, run):
    """后台线程解析：请求线程只负责把活派出去，绝不在这里读几十 MB。

    （这一条以前是直接在请求线程里跑的，导致 `/api/overview` 在冷缓存下
    串行解析十几个 run、几十秒不返回。）
    """
    if run.state == "loading":
      return
    with self._lock:
      thread = self._threads.get(run.id)
      if thread is not None and thread.is_alive():
        return
      thread = threading.Thread(target=self._load, args=(run, True), daemon=True,
                                name=f"kline-load-{run.id}")
      self._threads[run.id] = thread
    thread.start()

  def _load(self, run, background=False):
    lock = self._run_lock(run.id)
    if background:
      if not lock.acquire(blocking=False):
        return
    else:
      lock.acquire()
    try:
      if run.state == "ready" and not self._dirty(run, 0):
        return
      run.state = "loading"
      started = time.time()
      signature = run.signature()
      try:
        series = self._read_cache(run, signature)
        if series is None:
          with self._slots:
            series = _parse_event_dir(run.path)
          self._write_cache(run, signature, series)
        run.series = series
        run.summary.clear()
        run.health.clear()
        run.loaded_signature = signature
        run.load_seconds = time.time() - started
        run.loaded_at = time.time()
        run.state = "ready"
        run.error = None
      except _MissingTensorboard:
        run.state = "error"
        run.error_at = time.time()
        run.error = "当前 Python 环境没有 tensorboard，读不了 event 文件"
      except Exception as error:  # 半截写入的 event 文件不该让整页挂掉
        run.error_at = time.time()
        message = f"解析失败：{type(error).__name__}: {error}"
        if run.series:
          # 训练正写着文件时偶发读失败：保留上一次的数据继续画，只在状态栏提示。
          run.state = "ready"
          run.error = message
        else:
          run.state = "error"
          run.error = message
    finally:
      lock.release()

  # ---- 磁盘缓存 -----------------------------------------------------------
  def _cache_path(self, run):
    return self.cache_dir / f"{run.id}.lgk"

  def _read_cache(self, run, signature):
    path = self._cache_path(run)
    if not path.is_file():
      return None
    try:
      with gzip.open(path, "rb") as handle:
        if handle.read(len(CACHE_MAGIC)) != CACHE_MAGIC:
          return None
        head = _read_u32_blob(handle)
        header = json.loads(head.decode("utf-8"))
        if header.get("version") != CACHE_VERSION or header.get("signature") != signature:
          return None
        body = handle.read()
      return _decode_body(header, body)
    except (OSError, ValueError, KeyError, EOFError):
      return None

  def _write_cache(self, run, signature, series):
    try:
      self.cache_dir.mkdir(parents=True, exist_ok=True)
      path = self._cache_path(run)
      tmp = path.with_suffix(".tmp")
      with gzip.open(tmp, "wb", compresslevel=3) as handle:
        handle.write(_encode_cache(signature, series))
      os.replace(tmp, path)
    except OSError:
      pass  # 缓存写不进去不影响使用，只是下次还得再解析一遍


class _MissingTensorboard(Exception):
  pass


def _walk_runs(root):
  """在 root 下最多三层找到「直接包含 event 文件」的目录。"""
  found = []
  for dirpath, dirnames, filenames in os.walk(root):
    directory = Path(dirpath)
    depth = len(directory.relative_to(root).parts)
    events = sorted(n for n in filenames if n.startswith(EVENT_PREFIX))
    if events:
      found.append((directory, events))
      dirnames[:] = []  # run 目录下面不再往下走（params/ git/ 之类）
      continue
    if depth >= 3:
      dirnames[:] = []
    else:
      dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS)
  return found


def _split_name(name):
  """``2026-10-18_18-30-44_V8_him_standard_seed42`` → 标签 + 时间戳。"""
  parts = name.split("_")
  if len(parts) >= 3 and len(parts[0]) == 10 and len(parts[1]) == 8 and parts[0][4] == "-":
    return "_".join(parts[2:]), f"{parts[0]} {parts[1].replace('-', ':')}"
  return name, ""


def _parse_event_dir(directory):
  try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  except ImportError as error:  # pragma: no cover - 取决于环境
    raise _MissingTensorboard() from error
  accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0})
  accumulator.Reload()
  tags = accumulator.Tags().get("scalars", [])
  series = {}
  for tag in tags:
    events = accumulator.Scalars(tag)
    if not events:
      continue
    series[tag] = Series(
      array("q", (int(event.step) for event in events)),
      array("d", (float(event.value) for event in events)),
      array("d", (float(event.wall_time) for event in events)),
    )
  return series


# ---- 缓存文件：magic + u32 长度 + JSON 头 + 按 tag 排列的裸数组 -------------
def _encode_cache(signature, series):
  tags = sorted(series)
  layout = []
  offset = 0
  for tag in tags:
    item = series[tag]
    n = len(item)
    layout.append({"tag": tag, "n": n, "offset": offset})
    offset += n * 8  # 每段都是 8 字节定长：int64 / float64 / float64
  header = {"version": CACHE_VERSION, "signature": signature, "tags": layout, "bytes": offset}
  head = json.dumps(header, ensure_ascii=False).encode("utf-8")
  chunks = [CACHE_MAGIC, len(head).to_bytes(4, "little"), head]
  for tag in tags:
    item = series[tag]
    chunks.append(item.step.tobytes())
  for tag in tags:
    chunks.append(series[tag].value.tobytes())
  for tag in tags:
    chunks.append(series[tag].wall.tobytes())
  return b"".join(chunks)


def _decode_body(header, body):
  # 三种数组是分段存的（所有 step 放完再放 value），所以要先算出每段的起点。
  total = sum(item["n"] for item in header["tags"]) * 8
  if len(body) != total * 3:
    raise ValueError("缓存长度对不上，重新解析")
  series = {}
  step_base = 0
  value_base = total
  wall_base = total * 2
  for item in header["tags"]:
    n = item["n"]
    step = array("q")
    step.frombytes(body[step_base + item["offset"]:step_base + item["offset"] + n * 8])
    value = array("d")
    value.frombytes(body[value_base + item["offset"]:value_base + item["offset"] + n * 8])
    wall = array("d")
    wall.frombytes(body[wall_base + item["offset"]:wall_base + item["offset"] + n * 8])
    if len(step) != n or len(value) != n or len(wall) != n:
      raise ValueError("缓存内数组长度对不上，重新解析")
    series[item["tag"]] = Series(step, value, wall)
  return series


def _read_u32_blob(handle):
  raw = handle.read(4)
  if len(raw) != 4:
    raise EOFError("缓存头不完整")
  return handle.read(int.from_bytes(raw, "little"))
