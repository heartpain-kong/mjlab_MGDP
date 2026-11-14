from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from scripts.check_mgdp_inference import validate
from src.tasks.velocity.mdp.mgdp_perception import MGPDPerception, MGPDPerceptionCfg
from src.tasks.velocity.mdp.mgdp_inference import configure_perception_runtime, FrozenDepthRuntime


def test_validation_entry_moves_loaded_module_to_requested_device(tmp_path):
  from scripts import check_mgdp_inference as script
  # A meta device exercises migration on a CPU-only host. The loader creates
  # a CPU module even if checkpoint tensors were mapped to another device.
  model = MGPDPerception(MGPDPerceptionCfg())
  def check(module, device, *args):
    assert device == "meta"
    assert all(value.device.type == "meta" for value in module.parameters())
    assert all(value.device.type == "meta" for value in module.buffers())
    assert not module.training
    return {"passed":True}
  with patch.object(script.MGPDPerception,"from_checkpoint",return_value=model) as load, \
       patch.object(script,"validate",side_effect=check), \
       patch("sys.argv",["check", "--perception", "unused.pt", "--device", "meta",
                         "--output", str(tmp_path/"report.json")]):
    script.main()
  load.assert_called_once_with("unused.pt",map_location="cpu")


@pytest.mark.parametrize("execution,precision", [("eager","fp32"), ("cuda_graph","fp32"), ("eager","bf16")])
def test_recurrent_features_partial_resets_match_legacy_and_do_not_modify_weights(execution, precision):
  with torch.random.fork_rng(devices=[]):
    torch.manual_seed(42)
    model = MGPDPerception(MGPDPerceptionCfg()).eval()
    report = validate(model,"cpu",execution,precision,4,9,5)
    assert report["passed"]
    assert report["graph_count"] == 0


def test_gru_runs_once_for_multiple_cnn_chunks_and_outputs_are_not_aliased():
  model = MGPDPerception(MGPDPerceptionCfg()).eval()
  depth = torch.rand(9,2,16,16)
  with torch.no_grad(), patch.object(model.memory, "forward", wraps=model.memory.forward) as memory:
    out, hidden = model.feature_batched(depth, None, 4)
    assert memory.call_count == 1
    snapshot = out.clone()
    model.feature_batched(depth*.5, hidden, 4)
    torch.testing.assert_close(out, snapshot)


def test_runtime_options_validate_and_preserve_unspecified_values():
  cfg = NS(observations={"perception":NS(terms={"depth_feature":NS(params={"inference_precision":"bf16"})})})
  configure_perception_runtime(cfg,execution="cuda_graph",batch_size=2048)
  assert cfg.observations["perception"].terms["depth_feature"].params == {
    "inference_precision":"bf16", "inference_execution":"cuda_graph", "inference_batch_size":2048}
  with pytest.raises(ValueError):
    configure_perception_runtime(cfg,batch_size=-1)
  with pytest.raises(ValueError):
    configure_perception_runtime(NS(observations={}),execution="cuda_graph")
  with pytest.raises(ValueError):
    FrozenDepthRuntime(None,"bad")
  model = MGPDPerception(MGPDPerceptionCfg())
  with pytest.raises(RuntimeError,match="eval"):
    model.feature_batched(torch.rand(2,2,16,16),None,1)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_cuda_graph_replays_across_frames_partial_resets_and_tail_batch(precision):
  if not torch.cuda.is_available():
    pytest.skip("requires real CUDA CNN graph capture/replay")
  if precision == "bf16" and not torch.cuda.is_bf16_supported():
    pytest.skip("device lacks BF16 support")
  model = MGPDPerception(MGPDPerceptionCfg()).cuda().eval()
  report = validate(model,"cuda:0","cuda_graph",precision,4,9,8)
  assert report["graph_count"] == 2
  assert report["graph_replays"] == 24
