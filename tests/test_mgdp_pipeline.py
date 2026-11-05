"""Regression checks for recurrent training and CPU deployment observations."""

from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from mjlab.sensor import PinholeCameraPatternCfg

from src.mgdp_deployment import MGDPPolicyInput
from src.tasks.velocity.mdp.mgdp_observations import mgdp_depth_feature
from src.tasks.velocity.mdp.mgdp_perception import MGPDPerception, MGPDPerceptionCfg


class MGDPPipelineTests(unittest.TestCase):
  def test_sim2sim_rejects_wrong_encoder_and_requires_legacy_encoder(self):
    from scripts import sim2sim
    with tempfile.TemporaryDirectory() as tmp:
      onnx = Path(tmp) / "policy.onnx"
      onnx.write_bytes(b"stub")
      encoder = Path(tmp) / "encoder.pt"
      encoder.write_bytes(b"wrong weights")
      args = NS(onnx=str(onnx), task="LgDog1-Race-MGDP", map="race", map_file=None,
                perception=None)
      metadata = {"policy_architecture": "lgdog_mgdp_v1", "history_length": "4",
        "history_order": "term_major_oldest_to_newest", "frame_dim": "47",
        "control_dt": "0.02", "perception_checkpoint": encoder.name,
        "perception_sha256": "incorrect"}
      session = NS(get_modelmeta=lambda: NS(custom_metadata_map=metadata))
      config = NS(observations={"perception": object()})
      with patch.object(sim2sim, "build_model", return_value=(None, None, {"env_cfg": config})), \
           patch.object(sim2sim, "load_onnx", return_value=(session, "obs", [1, 252], [1, 12])):
        with self.assertRaisesRegex(ValueError, "does not match"):
          sim2sim.run(args)
        metadata.clear()
        metadata["policy_architecture"] = "lgdog_ppo_v1"
        with self.assertRaisesRegex(ValueError, "requires --perception"):
          sim2sim.run(args)

  def test_collector_drops_sequences_crossing_episode_reset(self):
    from scripts.pretrain_mgdp import collect
    # Env 0 resets halfway through the first block; env 1 never resets.
    episodes = [(1, 1), (2, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7)]
    env = NS(num_envs=2, device="cpu", episode_length_buf=None,
      scene={"depth_rays": NS(data=NS(distances=torch.zeros(2, 256)))})
    index = iter(enumerate(episodes))
    env.reset = lambda: None
    def step(actions):
      n, episode = next(index)
      env.episode_length_buf = torch.tensor(episode)
      env.scene["depth_rays"].data.distances.fill_(n / 10)
    env.step = step
    args = NS(sample_envs=2, num_envs=2, steps=7, action_hold=25, sequence_length=3)
    with patch("scripts.pretrain_mgdp.height_scan", return_value=torch.zeros(2, 187)):
      stacks, _ = collect(args, env)
    # First block retains only env 1, second retains both: 3 x 3 steps.
    self.assertEqual(tuple(stacks.shape), (3, 3, 2, 16, 16))

  def test_sequence_loss_trains_recurrent_weights(self):
    torch.manual_seed(42)
    model = MGPDPerception(MGPDPerceptionCfg()).train()
    before = model.memory.weight_hh_l0.detach().clone()
    loss, metrics = model.losses(torch.rand(2, 4, 2, 16, 16), torch.rand(2, 4, 1, 17, 11))
    loss.backward()
    for name, parameter in model.memory.named_parameters():
      self.assertIsNotNone(parameter.grad, name)
      self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
    torch.optim.Adam(model.parameters(), lr=1e-3).step()
    self.assertFalse(torch.equal(before, model.memory.weight_hh_l0))
    self.assertGreater(metrics["temporal"], 0)

  def test_collector_samples_across_terrain_columns_instead_of_an_env_prefix(self):
    from scripts.pretrain_mgdp import collect
    env = NS(num_envs=5, device="cpu", episode_length_buf=torch.zeros(5,dtype=torch.long),
      scene={"depth_rays":NS(data=NS(distances=torch.arange(5.)[:,None].expand(5,256)))},
      reset=lambda:None)
    env.step=lambda actions:env.episode_length_buf.add_(1)
    args=NS(sample_envs=3,num_envs=5,steps=3,action_hold=25,sequence_length=2)
    with patch("scripts.pretrain_mgdp.height_scan",return_value=torch.arange(5.)[:,None].expand(5,187)):
      stacks,heights=collect(args,env)
    torch.testing.assert_close(heights[:,0,0],torch.tensor([0.,.4,.8]))
    torch.testing.assert_close(stacks[:,0,0,0,0],torch.tensor([.02,.4,.8]))

  def test_validation_batches_preserve_whole_sequences_and_weight_tail_samples(self):
    from scripts.pretrain_mgdp import evaluate
    class Model(torch.nn.Module):
      def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1));self.seen=[]
      def losses(self,s,h):
        self.seen.append(tuple(s.shape))
        if h.shape != (len(s),4,1,17,11):raise AssertionError(h.shape)
        return self.weight.sum(),{"total":float(s[:,0,0,0,0].mean())}
    model=Model().train()
    stacks=torch.arange(5.)[:,None,None,None,None].expand(5,4,2,16,16)
    metrics=evaluate(model,stacks,torch.zeros(5,4,187),NS(height_hw=(17,11)),batch_size=2)
    self.assertEqual([shape[:2] for shape in model.seen],[(2,4),(2,4),(1,4)])
    self.assertAlmostEqual(metrics["total"],2.)
    self.assertTrue(model.training)

  def test_simulation_is_closed_and_collected_before_network_training(self):
    from scripts.pretrain_mgdp import collect_and_release
    import gc, weakref
    closed=[]
    class Env:
      def __init__(self):
        generator=NS(num_rows=4,num_cols=10,sub_terrains={"flat":NS(layout_version=7)})
        self.cfg=NS(events={},scene=NS(terrain=NS(terrain_generator=generator)))
        self.cycle=self
      def close(self):closed.append(True)
    references=[]
    def build(args):
      env=Env();references.append(weakref.ref(env));return env
    with patch("scripts.pretrain_mgdp.build_env",side_effect=build), \
         patch("scripts.pretrain_mgdp.collect",return_value=(torch.zeros(2,4,2,16,16),torch.zeros(2,4,187))):
      stacks,heights,metadata=collect_and_release(NS(device="cpu",steps=16))
    self.assertEqual(closed,[True])
    gc.collect()
    self.assertIsNone(references[0]())
    self.assertEqual(metadata["terrain_grid"],(4,10))
    self.assertEqual(stacks.device.type,"cpu")

  def test_cpu_depth_features_and_term_major_history_match_training(self):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom type="box" size="0.1 10 10" pos="2.1 0 0"/>
      <body name="robot"><site name="depth" quat="0.7071067812 0 -0.7071067812 0"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sensor = NS(frame=NS(name="depth"), ray_alignment="base", max_distance=5.,
      include_geom_groups=(0, 1, 2), exclude_parent_body=True,
      pattern=PinholeCameraPatternCfg(width=16, height=16, fovy=87))
    with tempfile.TemporaryDirectory() as tmp:
      checkpoint = Path(tmp) / "encoder.pt"
      MGPDPerception(MGPDPerceptionCfg()).save(checkpoint)
      deploy = MGDPPolicyInput(model, sensor, checkpoint)
      depth = deploy.depth(model, data)
      directions = deploy.directions @ data.site_xmat[deploy.site].reshape(3, 3).T
      expected = np.clip(2 / directions[:, 0], .1, 5) / 5
      np.testing.assert_allclose(depth.numpy().reshape(-1), expected, atol=1e-6)
      env = NS(device="cpu", episode_length_buf=torch.zeros(1, dtype=torch.long),
        scene={"depth_rays": NS(data=NS(distances=depth.reshape(1, -1) * 5))})
      frames = []
      for step in range(6):
        env.episode_length_buf[:] = step
        frame = np.arange(47, dtype=np.float32) + step * 100
        frames.append(frame)
        result = deploy.push(frame, model, data)
        feature = mgdp_depth_feature(env, checkpoint=str(checkpoint))
        np.testing.assert_allclose(result[-64:], feature[0].numpy(), atol=1e-6)
        history = [frames[max(0, step - i)] for i in range(3, -1, -1)]
        np.testing.assert_array_equal(result[:12], np.array(history)[:, :3].reshape(-1))
        self.assertEqual(result.shape, (252,))
      deploy.reset()
      restarted = deploy.push(frames[0], model, data)
      fresh = MGDPPolicyInput(model, sensor, checkpoint).push(frames[0], model, data)
      np.testing.assert_allclose(restarted, fresh, atol=1e-6)


if __name__ == "__main__":
  unittest.main()
