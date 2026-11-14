"""Old PPO weights must never silently consume a newly pretrained encoder."""

from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
import warnings
from unittest.mock import patch

import torch

from src.mgdp_checkpoint import (
  bind_checkpoint_encoder, encoder_fingerprint, file_digest,
  save_encoder_contract, validate_runtime_encoder,
)
from src.tasks.velocity.mdp.mgdp_perception import MGPDPerception, MGPDPerceptionCfg


class MGDPCheckpointTests(unittest.TestCase):
  def env(self, module, path):
    cfg=NS(observations={
      "actor":NS(history_length=4),
      "perception":NS(terms={"depth_feature":NS(params={"checkpoint":str(path)})})},
      decimation=4,sim=NS(mujoco=NS(timestep=.005)))
    return NS(cfg=cfg,step_dt=.02,_mgdp_perception_state={"module":module})

  def test_play_restores_original_features_after_global_encoder_is_retrained(self):
    with tempfile.TemporaryDirectory() as tmp,torch.random.fork_rng(devices=[]):
      root=Path(tmp);torch.manual_seed(42)
      original=MGPDPerception(MGPDPerceptionCfg()).eval()
      public=root/"current.pt";original.save(public)
      env=self.env(original,public)
      contract=save_encoder_contract(env,root)
      policy=root/"model_100.pt";torch.save({"infos":{"mgdp_perception":contract}},policy)
      # Re-pretraining updates the public file but must not change this policy's inputs.
      torch.manual_seed(43);replacement=MGPDPerception(MGPDPerceptionCfg()).eval()
      replacement.save(public)
      restored_path=bind_checkpoint_encoder(env.cfg,policy)
      self.assertNotEqual(restored_path,public)
      restored=MGPDPerception.from_checkpoint(restored_path).eval()
      depth=torch.rand(2,2,16,16)
      with torch.no_grad():
        expected,_=original.feature(depth,None)
        actual,_=restored.feature(depth,None)
        wrong,_=replacement.feature(depth,None)
      torch.testing.assert_close(expected,actual)
      self.assertFalse(torch.allclose(expected,wrong))
      validate_runtime_encoder(env,{"infos":{"mgdp_perception":contract}})
      env._mgdp_perception_state["module"]=replacement
      with self.assertRaisesRegex(ValueError,"不匹配"):
        validate_runtime_encoder(env,{"infos":{"mgdp_perception":contract}})

  def test_contract_accepts_same_weights_reserialized_but_rejects_corrupt_artifact(self):
    with tempfile.TemporaryDirectory() as tmp:
      root=Path(tmp);module=MGPDPerception(MGPDPerceptionCfg()).eval()
      env=self.env(module,root/"public.pt")
      contract=save_encoder_contract(env,root)
      policy=root/"model.pt";torch.save({"infos":{"mgdp_perception":contract}},policy)
      copy=root/"original-copy.pt";module.save(copy)
      self.assertEqual(encoder_fingerprint(module),encoder_fingerprint(MGPDPerception.from_checkpoint(copy)))
      self.assertEqual(bind_checkpoint_encoder(env.cfg,policy,copy),copy.resolve())
      (root/contract["file"]).write_bytes(b"corrupt")
      with self.assertRaisesRegex(ValueError,"校验失败"):
        bind_checkpoint_encoder(env.cfg,policy)

  def test_legacy_policy_uses_run_encoder_and_checks_onnx_checksum(self):
    import onnx
    with tempfile.TemporaryDirectory() as tmp:
      root=Path(tmp);module=MGPDPerception(MGPDPerceptionCfg()).eval()
      env=self.env(module,root/"new-public.pt")
      policy=root/"model.pt";torch.save({"infos":{}},policy)
      original=root/"policy.perception.pt";module.save(original)
      graph=onnx.helper.make_graph([],"stub",[],[])
      model=onnx.helper.make_model(graph)
      onnx.helper.set_model_props(model,{"policy_architecture":"lgdog_mgdp_v1",
        "perception_checkpoint":original.name,"perception_sha256":file_digest(original)})
      onnx.save(model,root/"policy.onnx")
      with warnings.catch_warnings(record=True) as caught:
        self.assertEqual(bind_checkpoint_encoder(env.cfg,policy),original.resolve())
      self.assertTrue(caught)  # Old checkpoint/run pairing cannot be proven per snapshot.
      original.write_bytes(b"wrong")
      with self.assertRaisesRegex(ValueError,"不匹配"):
        bind_checkpoint_encoder(env.cfg,policy)

  def test_legacy_policy_without_original_requires_explicit_original_file(self):
    with tempfile.TemporaryDirectory() as tmp:
      root=Path(tmp);module=MGPDPerception(MGPDPerceptionCfg()).eval()
      public=root/"new-public.pt";module.save(public)
      env=self.env(module,public)
      policy=root/"model.pt";torch.save({"infos":{}},policy)
      with self.assertRaisesRegex(ValueError,"旧MGDP策略没有保存配套感知文件"):
        bind_checkpoint_encoder(env.cfg,policy)
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        self.assertEqual(bind_checkpoint_encoder(env.cfg,policy,public),public.resolve())

  def test_non_mgdp_policy_is_unchanged_and_does_not_load_checkpoint(self):
    cfg=NS(observations={"actor":NS(history_length=3)})
    self.assertIsNone(bind_checkpoint_encoder(cfg,"nonexistent.pt"))
    self.assertIsNone(save_encoder_contract(NS(cfg=cfg),"nonexistent"))

  def test_precision_is_saved_restored_and_mismatches_are_rejected(self):
    with tempfile.TemporaryDirectory() as tmp:
      root=Path(tmp);module=MGPDPerception(MGPDPerceptionCfg()).eval()
      env=self.env(module,root/"public.pt")
      params=env.cfg.observations["perception"].terms["depth_feature"].params
      params["inference_precision"]="bf16"
      contract=save_encoder_contract(env,root)
      self.assertEqual(contract["inference_precision"],"bf16")
      policy=root/"model.pt";torch.save({"infos":{"mgdp_perception":contract}},policy)
      params["inference_precision"]="fp32"
      bind_checkpoint_encoder(env.cfg,policy)
      self.assertEqual(params["inference_precision"],"bf16")
      validate_runtime_encoder(env,{"infos":{"mgdp_perception":contract}})
      params["inference_precision"]="fp32"
      with self.assertRaisesRegex(ValueError,"precision"):
        validate_runtime_encoder(env,{"infos":{"mgdp_perception":contract}})

  def test_sensing_contract_restores_new_and_legacy_filters(self):
    from src.mgdp_checkpoint import sensing_contract, restore_sensing_contract
    from src.tasks.velocity.config.lgdog1.mgdp_env_cfg import lgdog1_race_mgdp_env_cfg
    cfg = lgdog1_race_mgdp_env_cfg()
    new = sensing_contract(cfg)
    self.assertEqual(new, {"terrain_scan": [0], "depth_rays": [0]})
    restore_sensing_contract(cfg, None)
    self.assertEqual(sensing_contract(cfg), {"terrain_scan": [0,1,2], "depth_rays": None})
    restore_sensing_contract(cfg, {"sensing": new})
    self.assertEqual(sensing_contract(cfg), new)

  def test_encoder_preserves_sensing_metadata_and_rejects_mismatched_environment(self):
    from src.tasks.velocity.config.lgdog1.mgdp_env_cfg import lgdog1_race_mgdp_env_cfg
    from src.tasks.velocity.mdp.mgdp_observations import get_perception
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp)/"encoder.pt"
      module = MGPDPerception(MGPDPerceptionCfg())
      filters = {"terrain_scan": [0], "depth_rays": [0]}
      module.save(path, extra={"sensing": filters})
      restored = MGPDPerception.from_checkpoint(path)
      self.assertNotEqual(encoder_fingerprint(module), encoder_fingerprint(restored))
      copy = Path(tmp)/"sidecar.pt"
      restored.save(copy)
      self.assertEqual(MGPDPerception.from_checkpoint(copy).sensing, filters)
      self.assertEqual(encoder_fingerprint(restored),
                       encoder_fingerprint(MGPDPerception.from_checkpoint(copy)))
      env = NS(device="cpu", cfg=lgdog1_race_mgdp_env_cfg(terrain_only_sensing=False))
      with self.assertRaisesRegex(ValueError, "过滤设置"):
        get_perception(env, MGPDPerceptionCfg(), str(copy))

  def test_play_entry_binds_original_encoder_before_allocating_physics(self):
    from scripts import play
    class StopBeforePhysics(Exception):pass
    with tempfile.TemporaryDirectory() as tmp:
      root=Path(tmp);module=MGPDPerception(MGPDPerceptionCfg()).eval()
      contract=save_encoder_contract(self.env(module,root/"public.pt"),root)
      policy=root/"model_100.pt";torch.save({"infos":{"mgdp_perception":contract}},policy)
      with patch.object(play,"configure_torch_backends"), \
           patch.object(play,"ManagerBasedRlEnv",side_effect=StopBeforePhysics) as create:
        with self.assertRaises(StopBeforePhysics):
          play.run_play("LgDog1-Race-MGDP",play.PlayConfig(checkpoint_file=str(policy),device="cpu"))
      configured=create.call_args.kwargs["cfg"].observations["perception"].terms["depth_feature"].params["checkpoint"]
      self.assertEqual(Path(configured),(root/contract["file"]).resolve())


if __name__=="__main__":unittest.main()
