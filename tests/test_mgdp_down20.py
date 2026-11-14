"""The confirmed mount must survive pretraining, restoration and CPU deployment."""
from functools import partial
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import math, tempfile, unittest
import mujoco
import numpy as np
import torch
from src.config.mgdp_camera import *
from src.mgdp_depth import camera_contract, pattern_contract, pattern_for
from src.mgdp_checkpoint import restore_sensing_contract
from src.mgdp_deployment import MGDPPolicyInput
from src.tasks.velocity.config.lgdog1.mgdp_env_cfg import (
    _mgdp_robot_spec, lgdog1_wall_training_env_cfg, lgdog1_wall_d435i20_env_cfg)
from src.tasks.velocity.mdp.mgdp_perception import MGPDPerception, MGPDPerceptionCfg


class Down20Tests(unittest.TestCase):
  def test_actual_depth_and_display_axes_point_down_twenty(self):
    from src.config.lgdog1 import FRONT_CAMERA_NAME
    model=_mgdp_robot_spec(20).compile()
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    site=model.site('mgdp_depth').id;cam=model.camera(FRONT_CAMERA_NAME).id
    expected=np.array([math.cos(math.radians(20)),0,-math.sin(math.radians(20))])
    np.testing.assert_allclose(-data.site_xmat[site].reshape(3,3)[:,2],expected,atol=1e-9)
    np.testing.assert_allclose(-data.cam_xmat[cam].reshape(3,3)[:,2],expected,atol=1e-9)
    np.testing.assert_allclose(model.site_pos[site],DEPTH_MOUNT_POS)
    self.assertEqual(model.cam_fovy[cam],58)
    np.testing.assert_array_equal(model.cam_resolution[cam],[848,480])
    old=_mgdp_robot_spec().compile();od=mujoco.MjData(old);mujoco.mj_forward(old,od)
    np.testing.assert_allclose(-od.site_xmat[old.site('mgdp_depth').id].reshape(3,3)[:,2],[1,0,0],atol=1e-9)

  def test_contract_records_actual_mount_and_rejects_other_angles(self):
    cfg=lgdog1_wall_d435i20_env_cfg();contract=camera_contract(cfg)
    self.assertEqual(contract,pattern_contract(pattern_for(D435I20_PROFILE)))
    self.assertEqual(contract['extrinsics'],mount_pose(20))
    cfg.scene.entities['robot'].spec_fn=partial(_mgdp_robot_spec,pitch_down_deg=15)
    self.assertNotEqual(camera_contract(cfg),contract)
    with self.assertRaisesRegex(ValueError,'differs'):
      restore_sensing_contract(cfg,{'camera':contract})
    with self.assertRaisesRegex(ValueError,'differs'):
      restore_sensing_contract(lgdog1_wall_d435i20_env_cfg(),{'camera':None})

  def test_contract_restore_rebuilds_actual_mount(self):
    cfg=lgdog1_wall_training_env_cfg()
    self.assertIsNone(camera_contract(cfg))
    contract=pattern_contract(pattern_for(D435I20_PROFILE))
    restore_sensing_contract(cfg,{'camera':contract})
    self.assertEqual(camera_contract(cfg),contract)
    self.assertEqual(cfg.scene.entities['robot'].spec_fn().site('mgdp_depth').quat.tolist(),contract['extrinsics']['quaternion_wxyz'])

  def test_cpu_deployment_restores_mount_before_first_raycast(self):
    # A real plane perpendicular to the down-20 optical axis, two metres away.
    axis=np.array([math.cos(math.radians(20)),0,-math.sin(math.radians(20))])
    pos=np.array(DEPTH_MOUNT_POS)+2.1*axis
    model=mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <geom type="box" size="10 10 .1" pos="{' '.join(map(str,pos))}"
        quat=".819152044289 0 -.573576436351 0"/>
      <body name="robot"><site name="mgdp_depth"/></body>
    </worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    cfg=MGPDPerceptionCfg(depth_hw=(16,16),depth_min=.2,depth_max=3.)
    module=MGPDPerception(cfg);module.camera=pattern_contract(pattern_for(D435I20_PROFILE))
    sensor=NS(frame=NS(name='mgdp_depth'),ray_alignment='base',include_geom_groups=(0,),exclude_parent_body=True,pattern=pattern_for('d435i_edge16'))
    with tempfile.TemporaryDirectory() as tmp:
      path=Path(tmp)/'encoder.pt';module.save(path)
      deploy=MGDPPolicyInput(model,sensor,path)
      depth=deploy.depth(model,data)
      torch.testing.assert_close(depth,torch.full_like(depth,2/3),atol=1e-6,rtol=1e-6)
      self.assertFalse(deploy._camera_pose_dirty)
      np.testing.assert_allclose(-data.site_xmat[deploy.site].reshape(3,3)[:,2],axis,atol=1e-9)

  def test_workbench_and_cli_select_same_isolated_camera(self):
    from src.dashboard.catalog import TASKS
    from src.dashboard.routes import pretrain
    from scripts.pretrain_mgdp import parse_args
    root=Path(__file__).resolve().parents[1]
    ctx=NS(root=root,python=Path('python'),jobs=Mock())
    status,_=pretrain(ctx,{'task':D435I20_ALIAS})
    argv=ctx.jobs.launch.call_args.args[0]
    with patch('sys.argv',argv[1:]):args=parse_args()
    self.assertEqual(status,200)
    self.assertEqual((args.task,args.depth_profile),(D435I20_TASK,D435I20_PROFILE))
    self.assertEqual((args.num_envs,args.terrain_rows,args.terrain_cols,args.batch_size),(64,8,10,64))
    self.assertEqual(Path(args.out),root/D435I20_CHECKPOINT)
    with patch('sys.argv',['pretrain','--task',D435I20_TASK]):args=parse_args()
    self.assertEqual((args.depth_profile,args.out),(D435I20_PROFILE,D435I20_CHECKPOINT))
    self.assertEqual(TASKS[D435I20_ALIAS].experiment,D435I20_EXPERIMENT)
    self.assertNotEqual(TASKS['race_mgdp_wall'].pretrain_checkpoint,D435I20_CHECKPOINT)
    with self.assertRaisesRegex(ValueError,'8行'):pretrain(ctx,{'task':D435I20_ALIAS,'terrain_rows':12})


if __name__=='__main__':unittest.main()
