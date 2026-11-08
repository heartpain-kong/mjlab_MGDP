"""Conda 环境配置脚本的静态回归测试（不需要真实 conda / GPU）。

只检查：
- configure_lgdog.sh 语法正确、--help 可用；
- 脚本默认环境名/前缀是 lgdog；
- 激活钩子会加载 server_env.sh（CUDA 驱动修复）；
- environment_lgdog.yml 能解析且 name 为 lgdog。
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "configure_lgdog.sh"
ENV_FILE = ROOT / "environment_lgdog.yml"


def find_bash() -> str | None:
  candidates = [
    shutil.which("bash"),
    r"E:\git\Git\bin\bash.exe",
    "/bin/bash",
    "/usr/bin/bash",
  ]
  for candidate in candidates:
    if candidate and Path(candidate).is_file():
      return candidate
  return None


class ConfigureScriptTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.bash = find_bash()

  def test_script_exists_and_is_lf(self):
    self.assertTrue(SCRIPT.is_file(), SCRIPT)
    data = SCRIPT.read_bytes()
    self.assertNotIn(b"\r\n", data, "configure_lgdog.sh 必须是 LF 换行")
    self.assertTrue(data.startswith(b"#!/usr/bin/env bash"))

  def test_default_environment_is_lgdog(self):
    text = SCRIPT.read_text(encoding="utf-8")
    self.assertIn("conda/envs/lgdog", text)
    self.assertIn("lgdog_server_env", text)
    self.assertIn("server_env.sh", text)
    self.assertIn("verify_install.py", text)
    self.assertIn("--skip-system-deps", text)

  def test_shell_syntax(self):
    if not self.bash:
      self.skipTest("bash 不可用")
    result = subprocess.run([self.bash, "-n", str(SCRIPT)], capture_output=True, text=True)
    self.assertEqual(result.returncode, 0, result.stderr)

  def test_help(self):
    if not self.bash:
      self.skipTest("bash 不可用")
    result = subprocess.run([self.bash, str(SCRIPT), "--help"], capture_output=True, text=True)
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("--env-dir", result.stdout)
    self.assertIn("--start", result.stdout)

  def test_env_file_parses(self):
    data = yaml.safe_load(ENV_FILE.read_text(encoding="utf-8"))
    self.assertEqual(data["name"], "lgdog")
    self.assertIn("python=3.11", data["dependencies"])


if __name__ == "__main__":
  unittest.main()
