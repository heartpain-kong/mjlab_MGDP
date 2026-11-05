"""Memory optimizations must preserve encoder outputs and backend inputs."""

import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from src.sim_memory import compact_model_allocations
from src.tasks.velocity.mdp.mgdp_perception import MGPDPerception, MGPDPerceptionCfg
from src.tasks.velocity.mdp.mgdp_observations import mgdp_depth_feature
from src.mgdp_checkpoint import encoder_fingerprint


class MGDPVRAMTests(unittest.TestCase):
  def test_chunked_features_match_full_batch_across_time_and_partial_resets(self):
    with torch.random.fork_rng(devices=[]),tempfile.TemporaryDirectory() as tmp:
      torch.manual_seed(42)
      model=MGPDPerception(MGPDPerceptionCfg()).eval()
      path=Path(tmp)/"encoder.pt";model.save(path)
      def env():
        return NS(scene={"depth_rays":NS(data=NS(distances=torch.ones(9,256)))},
          device="cpu",episode_length_buf=torch.zeros(9,dtype=torch.long))
      whole,chunked=env(),env()
      fingerprint=encoder_fingerprint(model)
      for step in range(4):
        distance=torch.rand(9,256)*5
        for item in (whole,chunked):
          item.scene["depth_rays"].data.distances=distance.clone()
          item.episode_length_buf.fill_(step)
          if step==2:item.episode_length_buf[[0,4,8]]=0
        full=mgdp_depth_feature(whole,checkpoint=str(path),inference_batch_size=0)
        split=mgdp_depth_feature(chunked,checkpoint=str(path),inference_batch_size=4)
        torch.testing.assert_close(full,split,rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(whole._mgdp_perception_state["hidden"],chunked._mgdp_perception_state["hidden"],rtol=2e-5,atol=2e-6)
      state=chunked._mgdp_perception_state
      self.assertEqual(state["module"].depth_encoder.skip_connections,[])
      self.assertEqual(encoder_fingerprint(state["module"]),fingerprint)

  def test_pretraining_rebuilds_skip_maps_and_backpropagates_after_feature_inference(self):
    with torch.random.fork_rng(devices=[]):
      torch.manual_seed(42)
      model=MGPDPerception(MGPDPerceptionCfg()).eval()
      with torch.no_grad():model.feature_batched(torch.rand(5,2,16,16),None,2)
      self.assertEqual(model.depth_encoder.skip_connections,[])
      model.train()
      loss,_=model.losses(torch.rand(2,3,2,16,16),torch.rand(2,3,1,17,11))
      loss.backward()
      self.assertEqual(len(model.depth_encoder.skip_connections),3)
      for module in (model.depth_encoder,model.depth_decoder,model.memory):
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()))

  def test_factory_reclaims_only_unused_table_and_restores_on_failure(self):
    import mujoco_warp as mjwarp
    import warp as wp
    table=wp.zeros(6,dtype=wp.vec2i,device="cpu")
    filtered=wp.zeros(3,dtype=wp.vec2i,device="cpu")
    lookup=wp.zeros(6,dtype=wp.vec2i,device="cpu")
    model=NS(nxn_geom_pair=table,nxn_geom_pair_filtered=filtered,nxn_pairid=lookup)
    factory=lambda *_args,**_kwargs:model
    with patch.object(mjwarp,"put_model",factory),patch("src.sim_memory._unused_pair_table_safe",return_value=True):
      with self.assertRaisesRegex(RuntimeError,"data allocation"):
        with compact_model_allocations() as stats:
          result=mjwarp.put_model(None)
          self.assertEqual(result.nxn_geom_pair.size,0)
          self.assertIs(result.nxn_geom_pair_filtered,filtered)
          self.assertIs(result.nxn_pairid,lookup)
          self.assertEqual(stats["unused_pair_bytes_released"],table.capacity)
          raise RuntimeError("data allocation")
      self.assertIs(mjwarp.put_model,factory)

  def test_unverified_backend_and_explicit_disable_keep_original_factory(self):
    import mujoco_warp as mjwarp
    factory=mjwarp.put_model
    with patch("src.sim_memory._unused_pair_table_safe",return_value=False):
      with compact_model_allocations() as stats:
        self.assertIs(mjwarp.put_model,factory)
        self.assertEqual(stats["unused_pair_bytes_released"],0)
    with compact_model_allocations(False):self.assertIs(mjwarp.put_model,factory)


if __name__=="__main__":unittest.main()
