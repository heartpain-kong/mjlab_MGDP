"""CPU MuJoCo depth observation and MGDP policy-input adapter."""

from pathlib import Path
import hashlib

import mujoco
import numpy as np
import torch

from src.him_contract import LGDOG_HIM_LAYOUT
from src.policy_history import PolicyHistory
from src.tasks.velocity.mdp.mgdp_perception import MGPDPerception


def file_digest(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MGDPPolicyInput:
  def __init__(self, model, sensor_cfg, checkpoint, history_length=4, inference_precision="fp32"):
    if inference_precision not in ("fp32", "bf16"):
      raise ValueError("Invalid MGDP deployment inference precision")
    self.inference_precision = inference_precision
    self.module = MGPDPerception.from_checkpoint(checkpoint).cpu().eval()
    self.cfg = self.module.cfg
    self.site = model.site(sensor_cfg.frame.name).id
    if sensor_cfg.ray_alignment != "base":
      raise ValueError("MGDP deployment requires base-aligned depth rays")
    self.pattern = sensor_cfg.pattern
    self._camera_pose_dirty = False
    if self.module.camera is not None:
      from src.mgdp_depth import pattern_for, pattern_contract
      self.pattern = pattern_for(self.module.camera['profile'])
      if pattern_contract(self.pattern) != self.module.camera:
        raise ValueError('Unsupported D435i camera calibration in encoder')
      if self.module.camera.get('version',1)>=2:
        pose=self.module.camera['extrinsics']
        model.site_pos[self.site]=pose['position_m']
        model.site_quat[self.site]=pose['quaternion_wxyz']
        # Compiled identity sites may bypass both position and rotation.
        # Clear that optimization after replacing the physical mount.
        model.site_sameframe[self.site]=0
        from src.config.lgdog1 import FRONT_CAMERA_NAME,FRONT_CAMERA_QUAT
        from src.config.mgdp_camera import tilt_quaternion
        for i in range(model.ncam):
          if (model.cam_bodyid[i]==model.site_bodyid[self.site] and
              model.camera(i).name.split('/')[-1]==FRONT_CAMERA_NAME):
            model.cam_pos[i]=pose['position_m']
            model.cam_quat[i]=tilt_quaternion(FRONT_CAMERA_QUAT,self.module.camera['pitch_down_deg'])
            model.cam_fovy[i]=self.module.camera['fovy'];model.cam_resolution[i]=(848,480)
        self._camera_pose_dirty = True
    offsets, directions = self.pattern.generate_rays(model, "cpu")
    self.offsets = offsets.numpy().astype(np.float64)
    self.directions = directions.numpy().astype(np.float64)
    if self.module.camera is None and len(self.directions) != self.cfg.depth_pixels:
      raise ValueError("MGDP camera resolution differs from perception checkpoint")
    if self.module.camera is None and not np.isclose(sensor_cfg.max_distance, self.cfg.depth_max):
      raise ValueError("MGDP camera range differs from perception checkpoint")
    self.groups = None if sensor_cfg.include_geom_groups is None else np.array(
      [i in sensor_cfg.include_geom_groups for i in range(6)], dtype=np.uint8)
    self.exclude = int(model.site_bodyid[self.site]) if sensor_cfg.exclude_parent_body else -1
    self.history = PolicyHistory(history_length)
    self.z_scale = directions[:, 2].abs()
    self.reset()

  def reset(self):
    self.history.reset()
    self.previous = None
    self.hidden = None

  def depth(self, model, data):
    if self._camera_pose_dirty:
      mujoco.mj_forward(model,data)
      self._camera_pose_dirty = False
    rotation = data.site_xmat[self.site].reshape(3, 3)
    origins = self.offsets @ rotation.T + data.site_xpos[self.site]
    directions = self.directions @ rotation.T
    distances = np.array([mujoco.mj_ray(model, data, origin, direction,
      self.groups, 1, self.exclude, None) for origin, direction in zip(origins, directions)])
    if self.module.camera is not None:
      from src.mgdp_depth import ray_depth
      return ray_depth(torch.from_numpy(distances.astype(np.float32))[None],
                       self.pattern, self.cfg, self.z_scale)
    distances = np.where(distances < 0, self.cfg.depth_max, distances)
    distances = np.clip(distances, self.cfg.depth_min, self.cfg.depth_max) / self.cfg.depth_max
    return torch.from_numpy(distances.astype(np.float32)).reshape(1, 1, *self.cfg.depth_hw)

  @torch.no_grad()
  def push(self, frame, model, data):
    history = self.history.push(frame).reshape(self.history.length, self.history.frame_dim)
    # mjlab flattens history within each term before concatenating terms.
    terms, start = [], 0
    for _, width in LGDOG_HIM_LAYOUT.terms:
      terms.append(history[:, start:start + width].reshape(-1))
      start += width
    depth = self.depth(model, data)
    previous = depth if self.previous is None else self.previous
    feature, self.hidden = self.module.feature_batched(
      torch.cat([previous, depth], dim=1), self.hidden, 0,
      precision=self.inference_precision)
    self.previous = depth
    return np.concatenate([*terms, feature[0].numpy()]).astype(np.float32)
