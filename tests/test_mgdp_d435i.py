from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import math, tempfile, unittest
import mujoco
import numpy as np
import torch
from rsl_rl.modules.distribution import GaussianDistribution
from src.mgdp_depth import (PROFILES, pattern_for, pattern_contract, apply_depth_profile,
                            camera_contract, ray_depth, reduce_z_depth)
from src.mgdp_checkpoint import restore_sensing_contract, encoder_fingerprint
from src.mgdp_deployment import MGDPPolicyInput
from src.mgdp_warmstart import reset_exploration_std
from src.tasks.velocity.mdp.mgdp_perception import MGPDPerception, MGPDPerceptionCfg
from src.tasks.velocity.mdp.mgdp_observations import _depth_image, get_perception


class D435iTests(unittest.TestCase):
  def test_upright_axes_and_independent_fovs(self):
    p=pattern_for('d435i_native32');_,r=p.generate_rays(None,'cpu')
    # Existing site rotation: local X -> up, Y -> left, -Z -> forward.
    world=r.numpy()@np.array([[0,0,-1],[0,1,0],[1,0,0]]).T
    world=world.reshape(32,48,3)
    self.assertGreater(world[0,0,1],0);self.assertGreater(world[0,0,2],0)
    self.assertLess(world[-1,-1,1],0);self.assertLess(world[-1,-1,2],0)
    self.assertAlmostEqual(float(world[0,0,1]/world[0,0,0]),(47/48)*math.tan(math.radians(43.5)),places=6)

  def test_invalid_depth_does_not_become_close_obstacle(self):
    z=torch.full((1,1,32,32),2.)
    z[0,0,:2,:2]=torch.tensor([[0.,float('nan')],[float('inf'),-1.]])
    z[0,0,2:4,2:4]=torch.tensor([[0.,1.2],[2.,3.]])
    depth,valid=reduce_z_depth(z,'d435i_edge16')
    self.assertEqual(depth[0,0,0,0],1.);self.assertFalse(valid[0,0,0,0])
    self.assertAlmostEqual(float(depth[0,0,1,1]),.4,places=6)
    self.assertTrue(valid[0,0,1,1])

  def test_optical_depth_and_cpu_deployment_agree_on_real_plane(self):
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom type="box" size="0.1 10 10" pos="2.1 0 0"/>
      <body name="robot"><site name="depth" quat="0.7071067812 0 -0.7071067812 0"/></body>
    </worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    for profile in ('d435i_edge16','d435i_native32'):
      p=pattern_for(profile);cfg=MGPDPerceptionCfg(depth_hw=PROFILES[profile]['output'],depth_min=.2,depth_max=3.)
      module=MGPDPerception(cfg);module.camera=pattern_contract(p)
      with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'encoder.pt';module.save(path)
        sensor=NS(frame=NS(name='depth'),ray_alignment='base',max_distance=5.,
                  include_geom_groups=(0,),exclude_parent_body=True,pattern=p)
        deploy=MGDPPolicyInput(model,sensor,path)
        depth=deploy.depth(model,data)
        torch.testing.assert_close(depth,torch.full_like(depth,2/3),atol=1e-6,rtol=1e-6)
        raw=torch.tensor((2/deploy.directions[:,2].__abs__())[None],dtype=torch.float32)
        env=NS(scene={'depth_rays':NS(cfg=NS(pattern=p),data=NS(distances=raw))})
        torch.testing.assert_close(_depth_image(env,'depth_rays',cfg),depth)
        self.assertEqual(MGPDPerception.from_checkpoint(path).camera,module.camera)

  def test_camera_contract_restores_and_rejects_mixing(self):
    from src.tasks.velocity.config.lgdog1.mgdp_env_cfg import lgdog1_wall_training_env_cfg
    cfg=lgdog1_wall_training_env_cfg();self.assertIsNone(camera_contract(cfg))
    contract=apply_depth_profile(cfg,'d435i_edge16')
    self.assertEqual(cfg.observations['perception'].terms['depth_feature'].params['cfg'].depth_max,3.)
    with self.assertRaisesRegex(ValueError,'differs'):restore_sensing_contract(cfg,{'camera':None})
    fresh=lgdog1_wall_training_env_cfg()
    restore_sensing_contract(fresh,{'camera':contract,'sensing':{'depth_rays':[0],'terrain_scan':[0]}})
    self.assertEqual(camera_contract(fresh),contract)

  def test_matching_shape_cannot_hide_changed_camera(self):
    module=MGPDPerception(MGPDPerceptionCfg());old=encoder_fingerprint(module)
    module.camera=pattern_contract(pattern_for('d435i_edge16'))
    self.assertNotEqual(old,encoder_fingerprint(module))
    env=NS(device='cpu',cfg=NS(scene=NS(sensors=())))
    with patch.object(MGPDPerception,'from_checkpoint',return_value=module):
      with self.assertRaisesRegex(ValueError,'camera profile'):get_perception(env,module.cfg,'stub')

  def test_explicit_std_override_handles_both_parameterizations(self):
    for kind in ('scalar','log'):
      dist=GaussianDistribution(12,init_std=.12,std_type=kind)
      runner=NS(alg=NS(actor=NS(distribution=dist)),_mgdp_warm_start={})
      reset_exploration_std(runner,.3);dist.update(torch.zeros(2,12))
      torch.testing.assert_close(dist.std,torch.full((2,12),.3))
      self.assertEqual(runner._mgdp_warm_start['exploration_std_override'],.3)

  def test_new_pretrain_profile_does_not_overwrite_old_encoder(self):
    from scripts.pretrain_mgdp import parse_args
    with patch('sys.argv',['pretrain','--task','LgDog1-Race-MGDP-WallTraining','--depth-profile','d435i_edge16']):
      args=parse_args()
    self.assertEqual(args.num_envs,64);self.assertEqual(args.terrain_cols,10)
    self.assertTrue(args.out.endswith('d435i_edge16.pt'))


if __name__=='__main__':unittest.main()
