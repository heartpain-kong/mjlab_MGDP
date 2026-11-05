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
