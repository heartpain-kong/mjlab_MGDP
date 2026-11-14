from types import SimpleNamespace
from src.runtime_profile import StageProfiler


def test_cpu_stage_measurement_restores_method_and_does_not_repeat_calls():
  calls = []
  obj = SimpleNamespace(step=lambda value: calls.append(value) or value * 2)
  original = obj.step
  with StageProfiler("cpu") as profile:
    profile.watch(obj, "step", "physics")
    assert obj.step(3) == 6
    assert obj.step(4) == 8
  assert obj.step is original
  assert calls == [3,4]
  report = profile.report()["physics"]
  assert report["calls"] == 2
  assert report["seconds"] >= 0
