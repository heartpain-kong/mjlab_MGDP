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

  def test_race_pretraining_and_viewer_use_selected_task_and_separate_encoder(self):
    from unittest.mock import Mock
    self.ctx.jobs=Mock()
    self.ctx.jobs.launch.return_value={"id":"test"}
    routes.dispatch_post(self.ctx,"/api/pretrain",{"task":"race_mgdp"})
    argv,mode,alias=self.ctx.jobs.launch.call_args.args
    self.assertEqual((mode,alias),("pretrain","race_mgdp"))
    self.assertIn("LgDog1-Race-MGDP",argv)
    for option,value in (("--num-envs","64"),("--terrain-rows","4"),("--terrain-cols","10"),("--batch-size","64")):
      self.assertEqual(argv[argv.index(option)+1],value)
    self.assertTrue(argv[-1].endswith("mgdp_race_v3.pt"))
    routes.dispatch_post(self.ctx,"/api/terrain/view",{"task":"race_mgdp"})
    argv,mode,alias=self.ctx.jobs.launch.call_args.args
    self.assertEqual(argv[-2:],["--task","LgDog1-Race-MGDP"])
    with self.assertRaisesRegex(ValueError,"不使用"):
      routes.dispatch_post(self.ctx,"/api/pretrain",{"task":"standard"})

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

  def test_pretrain_route_launches_mgdp_pretraining(self):
    launched = {}

    class Jobs:
      def launch(self, argv, mode, task):
        launched.update(argv=argv, mode=mode, task=task)
        return {"id": "pretrain01"}

    ctx = SimpleNamespace(
      root=ROOT, python=Path("python"), token="t", jobs=Jobs(),
      gpu=lambda: {"available": False, "gpus": []},
      metrics=lambda task: {"task": task, "available": False},
    )
    status, payload = routes.dispatch_post(ctx, "/api/pretrain", {})
    self.assertEqual(status, 200)
    self.assertEqual(payload["job"]["id"], "pretrain01")
    self.assertEqual(launched["mode"], "pretrain")
    joined = " ".join(launched["argv"])
    self.assertIn("pretrain_mgdp.py", joined)
    self.assertIn("--num-envs", joined)
    self.assertIn("mgdp_perception.pt", joined)


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
  def test_pretrain_remains_available_and_old_weights_do_not_finish_new_job(self):
    script = r"""
const fs=require('fs'), vm=require('vm'), assert=require('node:assert/strict');
const nodes={};
for(const id of ['task','task-prereq','task-prereq-actions','pretrain','pretrain-state']){
  const classes=new Set(['hidden']);
  nodes[id]={value:'race_mgdp',disabled:false,textContent:'',className:'',
    classList:{add:c=>classes.add(c),remove:c=>classes.delete(c),contains:c=>classes.has(c)}};
}
const ctx={document:{getElementById:id=>nodes[id]},window:{LgDogFormat:{}},
  setInterval:fn=>{ctx.poll=fn;return 1;},clearInterval:()=>{ctx.cleared=true;}};
vm.createContext(ctx);
const source=fs.readFileSync(process.argv[1],'utf8').replace(/\nstart\(\);\s*$/,'');
vm.runInContext(source,ctx);
vm.runInContext(`
  state={tasks:[{alias:'race_mgdp',capabilities:{needs_pretrain:true,pretrain_ready:true,
    pretrain_checkpoint:'logs/mgdp/perception/mgdp_race_v3.pt'}},
    {alias:'standard',capabilities:{needs_pretrain:false}}],jobs:[]};
  updateTask=()=>renderTaskPrerequisite(task());
  refresh=async()=>{}; preview=async()=>{};
  api=async()=>state; toast=text=>globalThis.message=text;
  updateTask();
`,ctx);
assert.equal(nodes['task-prereq-actions'].classList.contains('hidden'),false);
assert.equal(nodes.pretrain.disabled,false);
assert.equal(nodes.pretrain.textContent,'重新预训练感知模型');
vm.runInContext(`state.tasks[0].capabilities.pretrain_ready=false;updateTask();`,ctx);
assert.equal(nodes['task-prereq-actions'].classList.contains('hidden'),false);
assert.equal(nodes.pretrain.textContent,'预训练感知模型');
vm.runInContext(`state.tasks[0].capabilities.pretrain_ready=true;
  state.jobs=[{id:'new-run',mode:'pretrain',task:'race_mgdp',status:'running'}];
  updateTask();pollPretrain('new-run','race_mgdp');`,ctx);
assert.equal(nodes.pretrain.disabled,true);
(async()=>{
  // The old checkpoint remains on disk throughout this new running job.
  await ctx.poll();
  assert.equal(ctx.cleared,undefined);
  assert.equal(ctx.message,undefined);
  assert.equal(nodes.pretrain.disabled,true);
  vm.runInContext(`state.jobs[0].status='stopping';updateTask();`,ctx);
  assert.equal(nodes.pretrain.disabled,true);
  vm.runInContext(`state.jobs[0].status='failed';`,ctx);
  await ctx.poll();
  assert.equal(ctx.cleared,true);
  assert.match(ctx.message,/失败/);
  assert.doesNotMatch(ctx.message,/完成/);
  assert.equal(nodes.pretrain.disabled,false);
  assert.match(nodes['pretrain-state'].textContent,/现有感知模型仍可用/);
  vm.runInContext(`state.jobs[0].status='completed';pollPretrain('new-run','race_mgdp');`,ctx);
  await ctx.poll();
  assert.match(ctx.message,/预训练完成/);
  assert.equal(nodes.pretrain.disabled,false);
  vm.runInContext(`state.jobs[0].status='finished';state.tasks[0].capabilities.pretrain_ready=false;updateTask();`,ctx);
  assert.match(nodes['pretrain-state'].textContent,/没有生成感知权重/);
  // A task switch must not expose pretraining on tasks without an encoder.
  nodes.task.value='standard';vm.runInContext('updateTask();',ctx);
  assert.equal(nodes['task-prereq-actions'].classList.contains('hidden'),true);
  nodes.task.value='race_mgdp';
  vm.runInContext(`state.jobs[0].task='mgdp';state.jobs[0].status='running';updateTask();`,ctx);
  assert.equal(nodes.pretrain.disabled,true);
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    result = subprocess.run(["node", "-e", script, str(WEB / "app.js")],
                            capture_output=True, text=True, encoding="utf-8", timeout=15)
    self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

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
