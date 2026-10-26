"""mjlab 总览的回归测试：数据源发现、接口形状、只读边界。

不需要 GPU / 训练依赖；子服务一律不启动（autostart=False），只测页面和 JSON。
"""

import http.client
import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path

from src.portal.server import create_server, discover_logdirs
from src.portal.services import Services


def free_port():
  with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    return sock.getsockname()[1]


class DiscoveryTests(unittest.TestCase):
  def test_logdirs_include_project_logs_and_sibling_archive(self):
    with tempfile.TemporaryDirectory() as directory:
      base = Path(directory)
      project = base / "unitree_rl_mjlab_lgdog"
      (project / "logs" / "rsl_rl").mkdir(parents=True)
      (base / "数据").mkdir()
      dirs = discover_logdirs(project)
      self.assertIn((project / "logs" / "rsl_rl").resolve(), dirs)
      self.assertIn((base / "数据").resolve(), dirs)

  def test_missing_roots_are_skipped(self):
    with tempfile.TemporaryDirectory() as directory:
      project = Path(directory) / "proj"
      project.mkdir()
      self.assertEqual(discover_logdirs(project), [])

  def test_services_expose_argv_and_stopped_status(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      services = Services(root, "python", [root / "logs" / "rsl_rl"],
                          ports={"dashboard": free_port(), "kline": free_port(), "tensorboard": free_port()})
      self.assertEqual(services.argv("kline")[:3], ["python", "-m", "src.kline"])
      self.assertIn("--logdir", services.argv("kline"))
      self.assertEqual(services.argv("tensorboard")[1:3], ["-m", "tensorboard.main"])
      meta = services.describe("dashboard")
      self.assertEqual(meta["status"], "stopped")
      self.assertFalse(meta["managed"])
      with self.assertRaises(KeyError):
        services.describe("nope")


class PortalHTTPTests(unittest.TestCase):
  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)
    self.logdir = self.root / "logs/rsl_rl/lgdog1_velocity_blind_him_standard_v1/2026-10-25_10-00-00_V1"
    self.logdir.mkdir(parents=True)
    (self.logdir / "events.out.tfevents.1790000000.host.1.0").write_bytes(b"not really an event file")
    self.server = create_server(self.root, "python", [self.root / "logs" / "rsl_rl"], port=0, autostart=False,
                                ports={"dashboard": free_port(), "kline": free_port(), "tensorboard": free_port()})
    self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
    self.thread.start()
    self.addCleanup(self.cleanup)

  def cleanup(self):
    self.server.shutdown()
    self.server.server_close()
    self.thread.join(timeout=3)

  def request(self, path, method="GET", body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=15)
    connection.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers or {})
    response = connection.getresponse()
    data = response.read()
    connection.close()
    return response.status, data

  def test_page_and_assets(self):
    status, body = self.request("/")
    self.assertEqual(status, 200)
    self.assertIn("mjlab 总览", body.decode())
    for asset in ("/assets/style.css", "/assets/app.js", "/favicon.svg"):
      self.assertEqual(self.request(asset)[0], 200, asset)
    self.assertEqual(self.request("/assets/../../lgdog.py")[0], 404)

  def test_state_has_every_section(self):
    status, body = self.request("/api/state")
    self.assertEqual(status, 200)
    data = json.loads(body)
    for key in ("token", "root", "environment", "resources", "gpu", "services", "runs", "processes", "logs"):
      self.assertIn(key, data)
    self.assertEqual([item["name"] for item in data["services"]], ["dashboard", "kline", "tensorboard"])
    self.assertEqual(data["runs"]["items"][0]["label"], "V1")
    self.assertEqual(data["runs"]["source"], "files")

  def test_host_header_must_be_loopback(self):
    self.assertEqual(self.request("/api/state", headers={"Host": "unrelated.example"})[0], 403)

  def test_start_requires_token_and_same_origin(self):
    status, _ = self.request("/api/service", "POST", {"name": "kline", "action": "start"})
    self.assertEqual(status, 403)
    token = json.loads(self.request("/api/state")[1])["token"]
    headers = {"X-Mjlab-Token": token, "Content-Type": "application/json"}
    # 名字非法：要报 400 而不是把子进程拉起来
    status, body = self.request("/api/service", "POST", {"name": "nope", "action": "start"}, headers)
    self.assertEqual(status, 400)
    self.assertIn("没有这个服务", json.loads(body)["error"])
    headers["Origin"] = "https://unrelated.example"
    self.assertEqual(self.request("/api/service", "POST", {"name": "kline", "action": "start"}, headers)[0], 403)

  def test_log_endpoint_refuses_paths_outside_logs(self):
    status, body = self.request("/api/log?path=" + str(self.root / "setup.py"))
    self.assertEqual(status, 400)
    self.assertIn("logs", json.loads(body)["error"])

  def test_service_log_endpoint_reports_missing_file(self):
    status, body = self.request("/api/log?name=dashboard")
    self.assertEqual(status, 200)
    data = json.loads(body)
    self.assertFalse(data["exists"])
    self.assertEqual(data["text"], "")


if __name__ == "__main__":
  unittest.main()
