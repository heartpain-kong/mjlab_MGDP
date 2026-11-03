"""Task model, router table and frontend contract tests (no server needed)."""

from __future__ import annotations

import re
import shutil
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.dashboard import routes
from src.dashboard.catalog import DEFAULT_TASK, TASKS, capabilities, resolve_task, task_payload

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "src/dashboard/web"


class TaskModelTests(unittest.TestCase):
  def test_default_task_exists_and_is_first(self):
    self.assertIn(DEFAULT_TASK, TASKS)
    self.assertEqual(next(iter(TASKS)), DEFAULT_TASK)

  def test_every_task_resolves_by_alias_and_id(self):
    for alias, spec in TASKS.items():
      self.assertEqual(resolve_task(alias)[0], alias)
      self.assertEqual(resolve_task(spec.id)[0], alias)

  def test_unknown_task_lists_choices(self):
    with self.assertRaisesRegex(ValueError, "可选"):
      resolve_task("nope")

  def test_mgdp_declares_pretrain_capability(self):
    spec = TASKS["mgdp"]
    self.assertTrue(spec.requires_pretrain)
    self.assertIn("pretrain_mgdp.py", spec.pretrain_command)
    self.assertTrue(spec.pretrain_checkpoint.endswith(".pt"))
    self.assertIn("预训练", spec.pretrain_hint())
    self.assertTrue(capabilities(spec)["needs_pretrain"])

  def test_task_payload_keeps_flat_fields_and_capabilities(self):
    payload = task_payload("standard")
    self.assertEqual(payload["alias"], "standard")
    self.assertIn("id", payload)
    self.assertIn("experiment", payload)
    self.assertIn("capabilities", payload)
    self.assertFalse(payload["capabilities"]["needs_pretrain"])

  def test_flat_tasks_do_not_support_curriculum(self):
    for alias in ("flat", "him_flat"):
      self.assertFalse(TASKS[alias].supports_curriculum)


class RouteTableTests(unittest.TestCase):
  def setUp(self):
    self.ctx = SimpleNamespace(
      root=ROOT, python=Path("python"), token="t",
      jobs=SimpleNamespace(list=lambda: [], counts=lambda: {}, lock=None, jobs={}),
      gpu=lambda: {"available": False, "gpus": []},
      metrics=lambda task: {"task": task, "available": False},
    )

  def test_unknown_get_and_post_return_404(self):
    self.assertEqual(routes.dispatch_get(self.ctx, "/api/nope", {})[0], 404)
    self.assertEqual(routes.dispatch_post(self.ctx, "/api/nope", {})[0], 404)

  def test_state_route_exposes_tasks_and_capabilities(self):
    status, payload = routes.dispatch_get(self.ctx, "/api/state", {})
    self.assertEqual(status, 200)
    self.assertIn("tasks", payload)
    self.assertTrue(all("capabilities" in item for item in payload["tasks"]))
    self.assertEqual(len(payload["tasks"]), len(TASKS))

  def test_metrics_route_uses_injected_reader(self):
    status, payload = routes.dispatch_get(self.ctx, "/api/metrics", {"task": ["standard"]})
    self.assertEqual((status, payload["task"]), (200, "standard"))

  def test_pretrain_readiness_follows_checkpoint_on_disk(self):
    import tempfile
    from types import SimpleNamespace as NS

    with tempfile.TemporaryDirectory() as tmp:
      ctx = NS(root=Path(tmp), python=Path("python"), token="t",
               jobs=NS(list=lambda: [], counts=lambda: {}, lock=None, jobs={}),
               gpu=lambda: {"available": False, "gpus": []},
               metrics=lambda task: {"task": task, "available": False})
      _, payload = routes.dispatch_get(ctx, "/api/state", {})
      mgdp = next(item for item in payload["tasks"] if item["alias"] == "mgdp")
      self.assertFalse(mgdp["capabilities"]["pretrain_ready"])

      checkpoint = Path(tmp) / mgdp["capabilities"]["pretrain_checkpoint"]
      checkpoint.parent.mkdir(parents=True, exist_ok=True)
      checkpoint.write_text("fixture", encoding="utf-8")
      _, payload = routes.dispatch_get(ctx, "/api/state", {})
      mgdp = next(item for item in payload["tasks"] if item["alias"] == "mgdp")
      self.assertTrue(mgdp["capabilities"]["pretrain_ready"])


class FrontendContractTests(unittest.TestCase):
  """Static wiring checks: JS ids exist, JS API paths are served, JS parses."""

  def test_every_dom_id_used_by_js_exists_in_html(self):
    js = (WEB / "app.js").read_text(encoding="utf-8")
    html = (WEB / "index.html").read_text(encoding="utf-8")
    used = set(re.findall(r'\$\("([A-Za-z0-9_-]+)"\)(?!\?)', js))
    # Ids created by JS templates (not present in the static page).
    dynamic = set(re.findall(r'\.id\s*=\s*"([A-Za-z0-9_-]+)"', js))
    declared = set(re.findall(r'id="([A-Za-z0-9_-]+)"', html))
    missing = sorted(used - declared - dynamic)
    self.assertEqual(missing, [], f"app.js references ids missing from index.html: {missing}")

  def test_every_api_path_used_by_js_is_routed(self):
    js = (WEB / "app.js").read_text(encoding="utf-8")
    paths = {
      match.split("?")[0]
      for match in re.findall(r'api\("(/api/[^"`]+)"', js)
      for match in [match.replace("${", "{")]
    }
    known = set(routes.GET_ROUTES) | set(routes.POST_ROUTES)
    # Template literals are reported with the raw query suffix; normalise.
    normalised = {path.split("?")[0].split("&")[0] for path in paths}
    missing = sorted(path for path in normalised if path not in known and "{" not in path)
    self.assertEqual(missing, [], f"app.js calls unserved API paths: {missing}")

  def test_static_assets_are_whitelisted(self):
    html = (WEB / "index.html").read_text(encoding="utf-8")
    srcs = re.findall(r'(?:src|href)="(/assets/[^"]+)"', html)
    from src.dashboard.server import STATIC_FILES
    missing = [src for src in srcs if src not in STATIC_FILES]
    self.assertEqual(missing, [], f"HTML requests assets not served: {missing}")

  @unittest.skipUnless(shutil.which("node"), "node is not installed")
  def test_format_module_is_pure_and_usable_from_node(self):
    path = WEB / "js" / "format.js"
    script = (
      "const fs=require('fs'),vm=require('vm');"
      "const ctx={};ctx.globalThis=ctx;vm.createContext(ctx);"
      f"vm.runInContext(fs.readFileSync({str(path)!r},'utf8'),ctx);"
      "const f=ctx.LgDogFormat;"
      "if(!f)process.exit(2);"
      "if(f.metricValue({metrics:{a:{value:7}}},'a')!==7)process.exit(3);"
      "if(f.metricValue(undefined,'a')!==undefined)process.exit(4);"
      "if(f.healthText('warning')!=='需要检查')process.exit(5);"
      "if(f.healthText('other')!=='等待数据')process.exit(6);"
      "console.log('ok');"
    )
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    self.assertEqual(result.returncode, 0, result.stderr or result.stdout)


if __name__ == "__main__":
  unittest.main()
