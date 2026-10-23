"""曲线终端的回归测试：窗口聚合、缓存、HTTP 接口。

不需要 GPU / MuJoCo；只有「真的写一个 event 文件再读回来」那几条需要
当前 Python 装了 tensorboard，没装就自动跳过。
"""

import http.client
import json
import math
import tempfile
import threading
import time
import unittest
from array import array
from pathlib import Path

from src.kline import aggregate
from src.kline.labels import describe, describe_all
from src.kline.server import create_server
from src.kline.store import Series, Store

try:
  from tensorboard.compat.proto.event_pb2 import Event
  from tensorboard.compat.proto.summary_pb2 import Summary
  from tensorboard.summary.writer.event_file_writer import EventFileWriter
  HAS_TENSORBOARD = True
except ImportError:  # pragma: no cover - 取决于环境
  HAS_TENSORBOARD = False


def make_series(steps, values):
  return Series(array("q", steps), array("d", values), array("d", [float(s) for s in steps]))


def write_event_dir(directory, series, wall_start=1_700_000_000.0):
  """用 tensorboard 自带的 writer 造一个真正的 event 文件。"""
  directory = Path(directory)
  directory.mkdir(parents=True, exist_ok=True)
  writer = EventFileWriter(str(directory))
  for tag, values in series.items():
    for step, value in enumerate(values):
      summary = Summary(value=[Summary.Value(tag=tag, simple_value=float(value))])
      writer.add_event(Event(wall_time=wall_start + step, step=step, summary=summary))
  writer.flush()
  writer.close()
  return directory


class CandleTests(unittest.TestCase):
  def test_window_is_aligned_and_last_window_is_partial(self):
    series = make_series(range(12), [1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2])
    rows = aggregate.candles(series, 5)
    self.assertEqual([row["t"] for row in rows], [0, 5, 10])
    first = rows[0]
    self.assertEqual((first["o"], first["c"], first["h"], first["l"], first["n"]), (1, 5, 5, 1, 5))
    self.assertEqual((first["s"], first["e"]), (0, 4))
    self.assertEqual(rows[-1]["n"], 2)  # 最后一个窗口没满，照股票里「还没收盘」处理

  def test_unaligned_buckets_keep_requested_point_count(self):
    series = make_series(range(7), [3, 1, 4, 1, 5, 9, 2])
    rows = aggregate.candles(series, 3, align=False)
    self.assertEqual([row["n"] for row in rows], [3, 3, 1])
    self.assertEqual(rows[0]["h"], 4)
    self.assertEqual(rows[0]["l"], 1)

  def test_non_finite_points_are_skipped(self):
    series = make_series(range(4), [1.0, math.nan, 3.0, math.inf])
    rows = aggregate.candles(series, 10)
    self.assertEqual(rows[0]["n"], 2)
    self.assertEqual(rows[0]["c"], 3.0)

  def test_volume_kinds_cover_range_std_count_and_relative(self):
    series = make_series([0, 1, 2], [10.0, 20.0, 30.0])
    row = aggregate.candles(series, 10)[0]
    self.assertEqual(row["vol"]["range"], 20.0)
    self.assertEqual(row["vol"]["count"], 3.0)
    self.assertAlmostEqual(row["vol"]["rel"], 2.0)
    self.assertAlmostEqual(row["vol"]["std"], math.sqrt(200.0 / 3.0), places=9)

  def test_nice_bucket_and_window_plan_are_rounded(self):
    self.assertEqual(aggregate.nice_bucket(1), 1)
    self.assertEqual(aggregate.nice_bucket(37), 50)
    self.assertEqual(aggregate.nice_bucket(12345), 25000)
    plan = aggregate.window_plan(0, 1000, 1001)
    for key in ("short", "mid", "long", "full"):
      self.assertIn(plan[key], aggregate.NICE_BUCKETS)
    self.assertLess(plan["short"], plan["mid"])
    self.assertLess(plan["mid"], plan["long"])
    self.assertLessEqual(plan["long"], plan["full"])

  def test_trend_downsamples_without_changing_shape(self):
    series = make_series(range(10000), [float(step % 7) for step in range(10000)])
    points = aggregate.trend(series, max_points=1000)
    self.assertEqual(points["n"], 10000)
    self.assertLessEqual(len(points["y"]), 1000)
    self.assertGreater(len(points["y"]), 500)
    self.assertEqual(points["step"][0], 0)

  def test_summary_reports_first_last_and_spread(self):
    series = make_series(range(5), [2.0, 4.0, 1.0, 5.0, 3.0])
    summary = aggregate.summary(series)
    self.assertEqual((summary["first"], summary["last"]), (2.0, 3.0))
    self.assertEqual((summary["min"], summary["max"]), (1.0, 5.0))
    self.assertAlmostEqual(summary["pct"], 50.0)

  def test_diagnose_flags_nan_constant_and_steps(self):
    nan_series = make_series(range(30), [1.0] * 15 + [math.nan] * 15)
    titles = [item["title"] for item in aggregate.diagnose(nan_series)]
    self.assertTrue(any("非有限值" in title for title in titles))
    flat = aggregate.diagnose(make_series(range(40), [2.0] * 40))
    self.assertTrue(any("全程不变" in item["title"] for item in flat))
    # 离散取值的比例（摔倒比例这种）波动是真的，不能被判成常量
    noisy = aggregate.diagnose(make_series(range(40), [0.2 + 0.05 * (step % 5) for step in range(40)]))
    self.assertFalse(any("不变" in item["title"] for item in noisy))
    jump = [0.01 * step for step in range(30)]
    jump[20] = 100.0
    steps = [item["title"] for item in aggregate.diagnose(make_series(range(30), jump))]
    self.assertTrue(any("台阶" in title for title in steps))
    clean = aggregate.diagnose(make_series(range(60), [math.sin(step / 4) for step in range(60)]))
    self.assertEqual(clean[0]["level"], "ok")

  def test_diagnose_skips_counters(self):
    counter = aggregate.diagnose(make_series(range(40), [20.0] * 40), "Loss/him_updates")
    self.assertFalse(any("不变" in item["title"] for item in counter))

  def test_csv_export_has_header_and_rows(self):
    series = make_series(range(4), [1.0, 2.0, 3.0, 4.0])
    csv = aggregate.to_csv("run, with comma", "Train/mean_reward", aggregate.candles(series, 2))
    lines = csv.strip().splitlines()
    self.assertEqual(len(lines), 3)
    self.assertIn("window_step", lines[0])
    self.assertIn('"run, with comma"', lines[1])


class LabelTests(unittest.TestCase):
  def test_known_tag_gets_chinese_name_and_direction(self):
    info = describe("Train/mean_reward")
    self.assertEqual(info["label"], "平均回报")
    self.assertEqual(info["direction"], "up")
    self.assertTrue(info["known"])

  def test_unknown_tag_falls_back_to_its_group(self):
    info = describe("Episode_Metrics/brand_new_metric")
    self.assertEqual(info["group_label"], "回合指标")
    self.assertFalse(info["known"])
    self.assertEqual(info["label"], "brand new metric")

  def test_describe_all_is_grouped(self):
    entries = describe_all(["Episode_Reward/foot_gait", "Train/mean_reward"])
    self.assertEqual([entry["group_label"] for entry in entries], ["训练", "奖励项"])


@unittest.skipUnless(HAS_TENSORBOARD, "当前 Python 没装 tensorboard")
class StoreTests(unittest.TestCase):
  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)
    self.logdir = self.root / "logs/rsl_rl/lgdog1_velocity_blind_him_standard_v1/2026-10-18_18-30-44_V8_x"
    # 三个长度不同的 tag：缓存按「所有 step / 所有 value / 所有 wall」分段存，
    # 长度不一样才验证得到偏移是不是算对了。
    write_event_dir(self.logdir, {
      "Train/mean_reward": [1.0, 2.0, 3.0, 4.0, 5.0],
      "Loss/value": [0.5, 0.4, 0.3],
      "Policy/mean_std": [1.0, 1.1],
    })
    (self.logdir / "params").mkdir()
    self.cache = self.root / "cache"

  def test_scan_parses_and_caches_then_reuses_cache(self):
    store = Store([self.root / "logs/rsl_rl"], self.cache)
    store.scan()
    run = next(iter(store.runs.values()))
    self.assertEqual(run.label, "V8_x")
    self.assertEqual(run.stamp, "2026-10-18 18:30:44")
    self.assertEqual(run.experiment, "lgdog1_velocity_blind_him_standard_v1")

    store.ensure(run.id, wait=True)
    self.assertEqual(run.state, "ready", run.error)
    self.assertEqual(len(run.series["Train/mean_reward"]), 5)
    cached = list(self.cache.glob("*.lgk"))
    self.assertEqual(len(cached), 1)

    warm = Store([self.root / "logs/rsl_rl"], self.cache)
    warm.scan()
    again = warm.runs[run.id]
    warm.ensure(again.id, wait=True)
    self.assertEqual(again.state, "ready", again.error)
    # 从缓存读回来的每一条都必须完整，不能串位
    self.assertEqual({tag: len(item) for tag, item in again.series.items()},
                     {"Train/mean_reward": 5, "Loss/value": 3, "Policy/mean_std": 2})
    self.assertEqual(list(again.series["Train/mean_reward"].value), [1.0, 2.0, 3.0, 4.0, 5.0])
    self.assertEqual(list(again.series["Loss/value"].step), [0, 1, 2])
    self.assertGreaterEqual(again.load_seconds, 0.0)

  def test_stale_cache_is_ignored(self):
    store = Store([self.root / "logs/rsl_rl"], self.cache)
    store.scan()
    run = next(iter(store.runs.values()))
    store.ensure(run.id, wait=True)
    # 把缓存弄坏：签名对不上时必须重新解析而不是报错
    cached = next(self.cache.glob("*.lgk"))
    cached.write_bytes(b"not a cache")
    fresh = Store([self.root / "logs/rsl_rl"], self.cache)
    fresh.scan()
    run2 = fresh.runs[run.id]
    fresh.ensure(run2.id, wait=True)
    self.assertEqual(run2.state, "ready", run2.error)
    self.assertEqual(len(run2.series["Train/mean_reward"]), 5)

  def test_missing_directory_yields_no_runs(self):
    store = Store([self.root / "nowhere"], self.cache)
    self.assertEqual(store.scan(), {})


class HTTPTests(unittest.TestCase):
  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)
    self.logdir = self.root / "logs/rsl_rl/exp/2026-10-18_18-30-44_V8_x"
    if HAS_TENSORBOARD:
      write_event_dir(self.logdir, {"Train/mean_reward": [float(step) for step in range(12)]})
    else:  # pragma: no cover
      self.logdir.mkdir(parents=True)
    self.server = create_server(self.root, [self.root / "logs/rsl_rl"], port=0,
                                cache_dir=self.root / "cache")
    self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
    self.thread.start()
    self.addCleanup(self.cleanup)

  def cleanup(self):
    self.server.shutdown()
    self.server.server_close()
    self.thread.join(timeout=3)

  def request(self, path, method="GET", headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
    connection.request(method, path, headers=headers or {})
    response = connection.getresponse()
    body = response.read()
    connection.close()
    return response.status, body

  def json(self, path):
    status, body = self.request(path)
    self.assertEqual(status, 200, body[:200])
    return json.loads(body)

  def test_page_and_assets_are_served(self):
    status, body = self.request("/")
    self.assertEqual(status, 200)
    self.assertIn("曲线战士", body.decode())
    for asset in ("/assets/app.css", "/assets/app.js", "/assets/chart.js", "/favicon.svg"):
      self.assertEqual(self.request(asset)[0], 200, asset)
    self.assertEqual(self.request("/assets/../../setup.py")[0], 404)

  def test_host_header_must_match_loopback(self):
    self.assertEqual(self.request("/api/state", headers={"Host": "unrelated.example"})[0], 403)

  def test_state_lists_the_run(self):
    data = self.json("/api/state")
    self.assertEqual(len(data["runs"]), 1)
    self.assertEqual(data["runs"][0]["label"], "V8_x")
    self.assertEqual(data["key_tags"][0], "Train/mean_reward")

  @unittest.skipUnless(HAS_TENSORBOARD, "当前 Python 没装 tensorboard")
  def test_run_series_health_export_and_dict(self):
    run_id = self.json("/api/state")["runs"][0]["id"]
    meta = self.json(f"/api/run?id={run_id}")
    self.assertEqual(meta["state"], "ready", meta.get("error"))
    self.assertEqual(meta["default_tag"], "Train/mean_reward")
    self.assertIn("mid", meta["plan"])

    candle = self.json(f"/api/series?id={run_id}&tag=Train/mean_reward&mode=k&bucket=4")
    self.assertEqual(candle["state"], "ready")
    self.assertEqual(len(candle["candles"]), 3)
    self.assertEqual(candle["candles"][0]["o"], 0.0)
    self.assertEqual(candle["candles"][0]["c"], 3.0)
    self.assertTrue(candle["tail"])

    line = self.json(f"/api/series?id={run_id}&tag=Train/mean_reward&mode=line")
    self.assertEqual(line["mode"], "line")
    self.assertEqual(line["points"]["n"], 12)

    health = self.json(f"/api/health?id={run_id}")
    self.assertEqual(health["checked"], 1)

    overview = self.json("/api/overview?tags=Train/mean_reward")
    self.assertEqual(overview["runs"][0]["values"]["Train/mean_reward"]["last"], 11.0)

    status, body = self.request(f"/api/export?id={run_id}&tag=Train/mean_reward&bucket=4")
    self.assertEqual(status, 200)
    self.assertIn("window_step", body.decode())

    dictionary = self.json("/api/dict")
    self.assertTrue(any(entry["tag"] == "Train/mean_reward" for entry in dictionary["entries"]))

  def test_unknown_run_returns_404(self):
    self.assertEqual(self.request("/api/run?id=deadbeef00")[0], 404)
    self.assertEqual(self.request("/api/series?id=deadbeef00&tag=x")[0], 404)


if __name__ == "__main__":
  unittest.main()
