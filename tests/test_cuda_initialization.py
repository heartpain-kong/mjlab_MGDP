"""Keep the first phase error, propagate CUDA sync failures, restore factories."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.cuda_initialization import InitializationTrace, trace_cuda_initialization
from src.scene_diagnostics import report_device_memory, scene_memory_estimates


class CUDAInitializationTests(unittest.TestCase):
  def test_nested_error_record_keeps_specific_stage_and_original_exception(self):
    with tempfile.TemporaryDirectory() as directory, \
         patch("src.cuda_initialization.report_device_memory"):
      trace=InitializationTrace("cuda:0",directory)
      error=RuntimeError("device allocation failed")
      with self.assertRaises(RuntimeError) as caught:
        with trace.phase("environment-construction"):
          with trace.phase("physics-data"):pass
          with trace.phase("sensor-context"):raise error
      self.assertIs(caught.exception,error)
      record=json.loads((Path(directory)/"init_error.json").read_text(encoding="utf-8"))
      self.assertEqual(record["phase"],"sensor-context")
      self.assertEqual(record["last_completed"],"physics-data")
      self.assertIn("device allocation failed",record["traceback"])
      progress=json.loads((Path(directory)/"init_stage.json").read_text(encoding="utf-8"))
      self.assertEqual(progress["failed_phase"],"sensor-context")

  def test_post_phase_cuda_sync_error_is_not_swallowed(self):
    with tempfile.TemporaryDirectory() as directory, \
         patch("src.cuda_initialization.report_device_memory",side_effect=[None,RuntimeError("CUDA error 700")]):
      trace=InitializationTrace("cuda:0",directory)
      with self.assertRaisesRegex(RuntimeError,"CUDA error 700"):
        trace.wrap("simulation-graphs",lambda:"returned before async error")()
      error=json.loads((Path(directory)/"init_error.json").read_text(encoding="utf-8"))
      self.assertEqual(error["phase"],"simulation-graphs")
    with patch("torch.cuda.synchronize",side_effect=RuntimeError("CUDA error 700")):
      with self.assertRaisesRegex(RuntimeError,"CUDA error 700"):
        report_device_memory("cuda:0","graphs",strict=True)

  def test_factory_restoration_on_error_and_cpu_noop(self):
    import mujoco_warp as mw
    from mjlab.sim import Simulation
    original=(mw.put_model,mw.put_data,mw.create_render_context,Simulation.create_graph)
    with tempfile.TemporaryDirectory() as directory, \
         patch("src.cuda_initialization.report_device_memory"):
      with self.assertRaisesRegex(RuntimeError,"context allocation"):
        with trace_cuda_initialization("cuda:0",directory):
          raise RuntimeError("context allocation")
      self.assertEqual(original,(mw.put_model,mw.put_data,mw.create_render_context,Simulation.create_graph))
    with tempfile.TemporaryDirectory() as directory:
      with trace_cuda_initialization("cpu",directory):pass
      self.assertEqual(list(Path(directory).iterdir()),[])

  def test_fixed_map_lookup_does_not_shrink_with_environment_count(self):
    values={n:scene_memory_estimates(27838,n) for n in (512,1024,2048,3072)}
    self.assertEqual(values[512]["pair_lookup_bytes"],values[3072]["pair_lookup_bytes"])
    self.assertGreater(values[2048]["device_subtotal_bytes"],6.9*1024**3)
    self.assertGreater(values[3072]["device_subtotal_bytes"],8*1024**3)
    self.assertLess(values[512]["device_subtotal_bytes"],4*1024**3)

  def test_trace_wraps_compact_model_factory_without_bypassing_it(self):
    import mujoco_warp as mw
    from types import SimpleNamespace as NS
    from src.sim_memory import compact_model_allocations
    model=NS(ngeom=4)
    with tempfile.TemporaryDirectory() as directory, \
         patch("src.cuda_initialization.report_device_memory"), \
         patch("src.sim_memory._unused_pair_table_safe",return_value=True), \
         patch("src.sim_memory._shared_bvh_factory",return_value=None), \
         patch("src.sim_memory._nonvisual_recreate_factory",return_value=None), \
         patch("src.sim_memory._streaming_model_factory",return_value=lambda *_:model), \
         patch.object(mw,"put_model",side_effect=AssertionError("native path must not run")) as native:
      with compact_model_allocations() as stats,trace_cuda_initialization("cuda:0",directory):
        self.assertIs(mw.put_model(None),model)
      native.assert_not_called()
      self.assertTrue(stats["streamed_pair_construction"])
      self.assertIs(mw.put_model,native)


if __name__=="__main__":unittest.main()
