"""把一条标量曲线切成蜡烛 / 分时点，并做数据体检。

一个窗口（bucket）= 一根蜡烛：
    开盘 = 窗口里第一个值，收盘 = 最后一个值，
    最高 / 最低 = 窗口里的极大 / 极小，
    成交量 = 窗口内的波动口径（振幅 / 标准差 / 点数 / 相对变化）。

窗口按 step 对齐到 bucket 的整数倍，所以新数据进来时老蜡烛不会变形 ——
和股票软件里「今天这根还没收盘」是同一个道理：最后一个窗口不满也算一根。
"""

from __future__ import annotations

import math
import statistics

# 成交量（波动柱）的四种口径，和页面顶栏的下拉一一对应
VOLUME_KINDS = ("range", "std", "count", "rel")

# 这几类 tag 本来就是「每轮固定值」的计数器，别把它们报成「全程不变」。
COUNTER_HINTS = ("updates", "_count", "/count")

NICE_BUCKETS = (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 25000, 50000, 100000)


def nice_bucket(target):
  """把「大概切多少步一根」取整成 1/2/5×10^n，刻度才是人读的数字。"""
  target = max(1, float(target))
  for value in NICE_BUCKETS:
    if value >= target:
      return value
  return NICE_BUCKETS[-1]


def window_plan(first_step, last_step, count=0):
  """给页面算一组「短 / 中 / 长」窗口宽度，让每档大致是 120 / 40 / 12 根。

    ``count`` 只是兜底：有些 tag 的 step 不递增（比如 ``*/time``），
    这时用记录条数当跨度，免得所有点挤进同一个窗口。
  """
  span = max(1, int(last_step) - int(first_step))
  span = max(span, int(count))
  return {
    "short": nice_bucket(span / 120.0),
    "mid": nice_bucket(span / 40.0),
    "long": nice_bucket(span / 12.0),
    "full": nice_bucket(span),
  }


def candles(series, bucket, align=True):
  """把一条序列合成蜡烛。``align=True`` 时窗口按 step 对齐。"""
  bucket = max(1, int(bucket))
  out = []
  step = series.step
  value = series.value
  wall = series.wall
  current = None
  seen = 0
  for index in range(len(step)):
    point = value[index]
    if not math.isfinite(point):
      continue
    key = (step[index] // bucket) * bucket if align else seen // bucket
    seen += 1
    if current is None or key != current["t"]:
      if current is not None:
        out.append(current)
      current = {
        "t": key,
        "s": int(step[index]),
        "e": int(step[index]),
        "o": point,
        "h": point,
        "l": point,
        "c": point,
        "n": 0,
        "w0": wall[index],
        "w1": wall[index],
        "_sum": 0.0,
        "_sum2": 0.0,
      }
    current["e"] = int(step[index])
    current["c"] = point
    if point > current["h"]:
      current["h"] = point
    if point < current["l"]:
      current["l"] = point
    current["n"] += 1
    current["w1"] = wall[index]
    current["_sum"] += point
    current["_sum2"] += point * point
  if current is not None:
    out.append(current)
  for item in out:
    n = item["n"]
    mean = item.pop("_sum") / n
    variance = max(0.0, item.pop("_sum2") / n - mean * mean)
    item["mean"] = mean
    item["std"] = math.sqrt(variance)
    item["vol"] = {
      "range": item["h"] - item["l"],
      "std": item["std"],
      "count": float(n),
      "rel": abs(item["c"] - item["o"]) / abs(item["o"]) if item["o"] else 0.0,
    }
  return out


def trend(series, max_points=2400):
  """分时图：原始点太多时按顺序等分取均值，形状不变、传输量降下来。"""
  n = len(series)
  if n == 0:
    return {"x": [], "y": [], "step": [], "n": 0}
  chunk = max(1, math.ceil(n / float(max_points)))
  xs, ys, steps = [], [], []
  for start in range(0, n, chunk):
    stop = min(n, start + chunk)
    total = 0.0
    usable = 0
    for index in range(start, stop):
      point = series.value[index]
      if math.isfinite(point):
        total += point
        usable += 1
    if not usable:
      continue
    ys.append(total / usable)
    steps.append(int(series.step[start]))
    xs.append(len(xs))
  return {"x": xs, "y": ys, "step": steps, "n": n}


def summary(series):
  """页面上那些「现价 / 首值 / 最高 / 最低 / 涨跌」用的统计。"""
  if series is None or not len(series):
    return None
  values = [v for v in series.value if math.isfinite(v)]
  if not values:
    return None
  first = values[0]
  last = values[-1]
  change = last - first
  return {
    "first": first,
    "last": last,
    "min": min(values),
    "max": max(values),
    "mean": statistics.fmean(values),
    "change": change,
    "pct": (change / abs(first) * 100.0) if first else None,
    "n": len(values),
    "last_step": int(series.step[len(series) - 1]),
    "updated": float(series.wall[len(series) - 1]),
  }


def diagnose(series, tag=""):
  """数据体检：把「这条曲线靠不靠谱」翻译成几条人话。"""
  findings = []
  if series is None or not len(series):
    return [{"level": "warn", "title": "没有数据", "detail": "这条指标在 event 文件里没有记录点。"}]
  raw = list(series.value)
  bad = sum(1 for v in raw if not math.isfinite(v))
  good = [v for v in raw if math.isfinite(v)]
  steps = list(series.step)
  walls = list(series.wall)

  if bad:
    findings.append({
      "level": "warn" if bad < len(raw) * 0.02 else "bad",
      "title": f"{bad} 个非有限值",
      "detail": f"占 {bad / len(raw) * 100:.2f}%。图上这些点被跳过，NaN 通常会连带影响整段训练。",
    })
  if len(good) < 20:
    findings.append({"level": "warn", "title": "采样太少", "detail": f"只有 {len(good)} 个点，K 线的影线没有统计意义。"})
  if len(good) > 20:
    unique = len(set(good))
    spread = max(good) - min(good)
    level = statistics.fmean(good)
    counter = any(word in tag for word in COUNTER_HINTS)
    if counter:
      pass  # 计数器类指标「每轮固定」是正常的，不报
    elif unique == 1:
      findings.append({
        "level": "warn",
        "title": "全程不变",
        "detail": f"{len(good)} 个点全都是 {good[0]:.6g} —— 这一项在本次训练里可能从未被触发，"
                  "调奖励时值得确认一下是不是真的接上了。",
      })
    elif spread <= max(1e-12, abs(level) * 0.001):
      # 只有「整段几乎是一条直线」才算，别把「离散取值的比例」误判成常量：
      # 摔倒比例、非法接触这类指标天生只有几十个不同取值，但波动是真的。
      findings.append({
        "level": "warn",
        "title": "几乎不变",
        "detail": f"整段只在 {min(good):.4g} ~ {max(good):.4g} 之间，相对波动不到 0.1% —— "
                  "先确认这个指标真的接上了，而不是常量占位。",
      })
  # 突变：相邻变化的中位数做尺度，十倍以上就算「爆点」
  deltas = [abs(good[i] - good[i - 1]) for i in range(1, len(good))]
  if len(deltas) > 10:
    scale = statistics.median(deltas)
    if scale > 0:
      jumps = [(i + 1, deltas[i]) for i in range(len(deltas)) if deltas[i] > scale * 20]
      if jumps:
        worst = max(jumps, key=lambda item: item[1])
        findings.append({
          "level": "info",
          "title": f"{len(jumps)} 处台阶",
          "detail": f"最大一次在 step {steps[worst[0]]}，变化 {worst[1]:.4g}，是中位变化的 {worst[1] / scale:.0f} 倍；"
                    "课程放宽、指令范围切换常出现这种台阶。",
        })
  # 时间断档
  if len(walls) > 10:
    gaps = [walls[i] - walls[i - 1] for i in range(1, len(walls))]
    typical = statistics.median(gaps)
    if typical > 0:
      holes = [(i, gaps[i]) for i in range(len(gaps)) if gaps[i] > typical * 30]
      if holes:
        index, gap = max(holes, key=lambda item: item[1])
        findings.append({
          "level": "info",
          "title": "时间断档",
          "detail": f"step {steps[index + 1]} 前面断了 {gap / 60:.0f} 分钟（常见于中断后续训或换机器）。",
        })
  if not findings:
    findings.append({"level": "ok", "title": "没有明显问题", "detail": "点足够多、没有 NaN、也没有异常台阶。"})
  return findings


def to_csv(run_label, tag, rows):
  """导出蜡烛，方便丢进 Excel / pandas 再算。"""
  head = "run,tag,window_step,first_step,last_step,open,high,low,close,mean,std,points,start_time,end_time\n"
  lines = [head]
  for row in rows:
    lines.append(
      f'{_csv(run_label)},{_csv(tag)},{row["t"]},{row["s"]},{row["e"]},'
      f'{row["o"]:.10g},{row["h"]:.10g},{row["l"]:.10g},{row["c"]:.10g},'
      f'{row["mean"]:.10g},{row["std"]:.10g},{row["n"]},'
      f'{row["w0"]:.3f},{row["w1"]:.3f}\n'
    )
  return "".join(lines)


def _csv(text):
  text = str(text)
  return '"' + text.replace('"', '""') + '"' if any(ch in text for ch in ',"\n') else text
