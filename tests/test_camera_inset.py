"""Real camera geometry, offscreen rendering and overlay lifecycle checks."""

import unittest
import tempfile
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import mujoco
from src.camera_inset import CameraInset, find_camera
from src.assets.robots.lgdog1.lgdog1_constants import get_spec, LGDOG1_XML
from src.tasks.velocity.config.lgdog1.mgdp_env_cfg import _mgdp_robot_spec, DEPTH_SITE


class CameraInsetTests(unittest.TestCase):
  def test_launcher_camera_options_reach_both_cli_parsers(self):
    import mjlab
    import tyro
    from scripts.play import PlayConfig
    from scripts import sim2sim
    from src.dashboard.commands import build_command

    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      options = dict(task="flat", camera_inset=False, camera_inset_size=128,
                     camera_inset_fps=15., camera_inset_corner="top-right")
      checkpoint = root / "model.pt"; checkpoint.touch()
      onnx = root / "policy.onnx"; onnx.touch()
      argv = build_command(root, sys.executable, dict(options, mode="play", viewer="native",
                                                    checkpoint=str(checkpoint)))
      cfg = tyro.cli(PlayConfig, args=argv[3:], config=mjlab.TYRO_FLAGS)
      self.assertEqual((cfg.camera_inset, cfg.camera_inset_size, cfg.camera_inset_fps,
                        cfg.camera_inset_corner), (False, 128, 15., "top-right"))
      argv = build_command(root, sys.executable, dict(options, mode="sim", onnx=str(onnx)))
      with patch.object(sys, "argv", argv[1:]), patch.object(sim2sim, "run", return_value=0) as run:
        self.assertEqual(sim2sim.main(), 0)
      cfg = run.call_args.args[0]
      self.assertEqual((cfg.camera_inset, cfg.camera_inset_size, cfg.camera_inset_fps,
                        cfg.camera_inset_corner), (False, 128, 15., "top-right"))

  def test_camera_matches_depth_site_and_does_not_change_physics(self):
    model = _mgdp_robot_spec().compile()
    data = mujoco.MjData(model)
    data.qpos[3] = 1
    mujoco.mj_forward(model,data)
    camera = find_camera(model); site = model.site(DEPTH_SITE).id
    np.testing.assert_allclose(data.cam_xpos[camera], data.site_xpos[site], atol=1e-9)
    axes = data.cam_xmat[camera].reshape(3,3)
    np.testing.assert_allclose(-axes[:,2], [1,0,0], atol=1e-9)
    np.testing.assert_allclose(axes[:,1], [0,0,1], atol=1e-9)
    np.testing.assert_allclose(axes[:,0], [0,-1,0], atol=1e-9)
    np.testing.assert_allclose(-axes[:,2], -data.site_xmat[site].reshape(3,3)[:,2], atol=1e-9)
    self.assertEqual(model.cam_fovy[camera],87.)
    baseline = mujoco.MjModel.from_xml_path(str(LGDOG1_XML))
    camera_model = get_spec().compile()
    for name in ("body_mass","body_inertia","jnt_range","dof_armature","geom_size"):
      np.testing.assert_array_equal(getattr(baseline,name),getattr(camera_model,name))

  def test_real_render_corner_resize_toggle_and_close(self):
    spec = mujoco.MjSpec.from_string('''<mujoco><worldbody>
      <camera name="robot_front" quat="0.7071067812 0 -0.7071067812 0" fovy="87"/>
      <geom type="box" pos="2 0 0" size="0.1 10 10" rgba="1 0.1 0.1 1"/>
    </worldbody></mujoco>''')
    model=spec.compile();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    images=[];clears=[]
    viewer=NS(is_running=lambda:True, viewport=mujoco.MjrRect(20,30,900,600),
      set_images=lambda image:images.append(image),clear_images=lambda:clears.append(True))
    overlay=CameraInset(model,size=128,fps=10,corner="top-right")
    try:
      overlay.update(viewer,data,force=True)
      rect,image=images[-1]
      self.assertEqual((rect.left,rect.bottom,rect.width,rect.height),(780,490,128,128))
      self.assertEqual(image.shape,(128,128,3))
      self.assertGreater(int(image[64,64,0]),int(image[64,64,1]))
      overlay.toggle();overlay.update(viewer,data)
      self.assertEqual(len(clears),1)
      overlay.toggle();overlay.size=512
      viewer.viewport=mujoco.MjrRect(0,0,1800,1800)
      overlay.update(viewer,data,force=True)
      self.assertEqual(images[-1][1].shape,(512,512,3))
      viewer.viewport=mujoco.MjrRect(0,0,240,180)
      overlay.update(viewer,data,force=True)
      self.assertEqual(images[-1][1].shape,(60,60,3))
    finally:
      overlay.close()
    self.assertIsNone(overlay.renderer)


if __name__ == "__main__":
  unittest.main()
