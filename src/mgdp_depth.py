"""Opt-in D435i camera conventions; legacy encoders keep their original rays."""
from dataclasses import dataclass, replace
import math
import torch
import torch.nn.functional as F
from mjlab.sensor import PinholeCameraPatternCfg
from src.config.mgdp_camera import D435I20_PROFILE, mount_pose


PROFILES = {
    # Four times the rays, same encoder input size. Nearest-valid reduction
    # deliberately preserves small foreground obstacles, at the cost of dilation.
    'd435i_edge16': {'rays': (32, 32), 'output': (16, 16), 'pool': 2},
    'd435i_native32': {'rays': (32, 48), 'output': (32, 48), 'pool': 1},
    # ★ 2026-11-19：与 env 的 32x32 编码输入对齐（射线 64x64，池化到 32x32）。
   # ★ 2026-11-19：俯角取 WMP 随机区间 15~25° 的中值 22°；识别距离对齐 WMP 的 2 m。
   D435I20_PROFILE: {'rays': (64, 64), 'output': (32, 32), 'pool': 2, 'pitch_down_deg': 22.},
}


@dataclass
class D435iPatternCfg(PinholeCameraPatternCfg):
    profile: str = 'd435i_edge16'
    fovx: float = 87.
    fovy: float = 58.

    def generate_rays(self, mj_model, device):
        # Pixel centres, in upright RealSense row-major order. The existing
        # mgdp_depth site maps local X to robot +Z and local Y to robot +Y.
        u = (torch.arange(self.width, device=device) + .5) / self.width * 2 - 1
        v = (torch.arange(self.height, device=device) + .5) / self.height * 2 - 1
        vv, uu = torch.meshgrid(v, u, indexing='ij')
        directions = torch.stack((-vv.flatten() * math.tan(math.radians(self.fovy / 2)),
                                  -uu.flatten() * math.tan(math.radians(self.fovx / 2)),
                                  -torch.ones(self.width * self.height, device=device)), dim=1)
        directions = F.normalize(directions, dim=1)
        return torch.zeros_like(directions), directions


def pattern_for(profile):
    spec = PROFILES[profile]
    h, w = spec['rays']
    return D435iPatternCfg(width=w, height=h, profile=profile)


def pattern_contract(pattern):
    if not isinstance(pattern, D435iPatternCfg):
        return None
    spec = PROFILES[pattern.profile]
    contract = {'version': 1, 'profile': pattern.profile, 'rays': [pattern.height, pattern.width],
            'output': list(spec['output']), 'pool': spec['pool'], 'fovx': pattern.fovx,
            'fovy': pattern.fovy, 'depth_min': .2, 'depth_max': 3.,
            'projection': 'optical_z', 'pixels': 'upright_row_major_centres',
            'reduction': 'nearest_valid', 'mount': 'mgdp_depth_v1'}
    if pattern.profile == D435I20_PROFILE:
        contract.update(version=2, mount='mgdp_depth_down20_v1',
                        pitch_down_deg=20., extrinsics=mount_pose(20.))
    return contract


def camera_contract(env_cfg):
    for sensor in getattr(getattr(env_cfg, 'scene', None), 'sensors', ()) or ():
        if sensor.name == 'depth_rays':
            contract = pattern_contract(sensor.pattern)
            if contract is not None and contract['version'] >= 2:
                # Inspect the actual entity spec, not just the profile name.
                spec = env_cfg.scene.entities[sensor.frame.entity].spec_fn()
                site = spec.site(sensor.frame.name)
                if site is None:
                    raise ValueError('D435i depth site is missing from robot spec')
                quat = list(site.quat);norm = math.sqrt(sum(v*v for v in quat))
                sign = 1. if quat[0] >= 0 else -1.
                contract['extrinsics'] = {
                    'position_m':[round(float(v),12) for v in site.pos],
                    'quaternion_wxyz':[round(float(v)*sign/norm,12) for v in quat]}
            return contract
    return None


def apply_depth_profile(env_cfg, profile):
    """Call before allocating an environment; no legacy defaults are modified."""
    from src.tasks.velocity.mdp.mgdp_perception import MGPDPerceptionCfg
    pattern = pattern_for(profile)
    from functools import partial
    from src.tasks.velocity.config.lgdog1.mgdp_env_cfg import _mgdp_robot_spec
    pitch = PROFILES[profile].get('pitch_down_deg',0.)
    robot = env_cfg.scene.entities['robot']
    env_cfg.scene.entities['robot'] = replace(robot, spec_fn=partial(_mgdp_robot_spec,pitch_down_deg=pitch))
    sensor = next(s for s in env_cfg.scene.sensors if s.name == 'depth_rays')
    sensor.pattern = pattern
    # Ray lengths exceed optical-Z depths at the image corners.
    sensor.max_distance = 3. * math.sqrt(1 + math.tan(math.radians(pattern.fovx / 2)) ** 2
                                        + math.tan(math.radians(pattern.fovy / 2)) ** 2)
    terms = env_cfg.observations.get('perception')
    if terms is not None:
        params = terms.terms['depth_feature'].params
        cfg = params['cfg']
        cfg = MGPDPerceptionCfg.from_dict(cfg) if isinstance(cfg, dict) else cfg
        params['cfg'] = replace(cfg, depth_hw=PROFILES[profile]['output'], depth_min=.2, depth_max=3.)
    return camera_contract(env_cfg)


def reduce_z_depth(depth_m, profile):
    """Real D435i Z16 input, after conversion to metres; zero/NaN/Inf are invalid.

    Returns normalized depth AND an output validity mask. Invalid samples never
    become foreground minima. All-invalid cells remain unknown (depth=far).
    No temporal persistence or hole filling is performed.
    """
    output = PROFILES[profile]['output']
    if depth_m.ndim != 4 or depth_m.shape[1] != 1:
        raise ValueError('Expected depth [B,1,H,W] in metres')
    if depth_m.shape[-2] < output[0] or depth_m.shape[-1] < output[1]:
        raise ValueError('Input depth must not be smaller than encoder image')
    valid = torch.isfinite(depth_m) & (depth_m > 0) & (depth_m <= 3.)
    clean = torch.where(valid, depth_m.clamp(.2, 3.), 3.)
    depth = -F.adaptive_max_pool2d(-clean, output) / 3.
    valid = F.adaptive_max_pool2d(valid.to(clean.dtype), output) > 0
    return depth, valid


def ray_depth(distances, pattern, cfg, z_scale=None):
    if distances.shape[-1] != pattern.height * pattern.width:
        raise ValueError('Ray count differs from camera profile')
    if tuple(cfg.depth_hw) != PROFILES[pattern.profile]['output'] or cfg.depth_max != 3. or cfg.depth_min != .2:
        raise ValueError('D435i encoder depth convention differs from sensor profile')
    if z_scale is None:
        z_scale = pattern.generate_rays(None, distances.device)[1][:, 2].abs()
    valid = torch.isfinite(distances) & (distances > 0)
    z = torch.where(valid, distances * z_scale, 0.)
    return reduce_z_depth(z.reshape(len(z), 1, pattern.height, pattern.width), pattern.profile)[0]
