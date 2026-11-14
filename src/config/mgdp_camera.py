"""Shared camera/task identifiers; safe to import in the stdlib workbench."""
import math

D435I20_TASK = 'LgDog1-Race-MGDP-WallTraining-D435i20'
D435I20_ALIAS = 'race_mgdp_wall_d435i20'
D435I20_PROFILE = 'd435i_edge16_down20'
D435I20_CHECKPOINT = 'logs/mgdp/perception/mgdp_wall_d435i20.pt'
D435I20_EXPERIMENT = 'lgdog1_velocity_race_mgdp_wall_d435i20_v6'
DEPTH_MOUNT_POS = (.02, 0., .075)
LEGACY_DEPTH_QUAT = (.7071067812, 0., -.7071067812, 0.)


def is_wall_task(task):
  return task in ('LgDog1-Race-MGDP-WallTraining', D435I20_TASK)


def tilt_quaternion(quat, pitch_down_deg):
  """Pre-multiply a camera-to-body quaternion by body-Y pitch."""
  if pitch_down_deg == 0:
    return tuple(quat)
  a=math.cos(math.radians(pitch_down_deg)/2);c=math.sin(math.radians(pitch_down_deg)/2)
  norm=math.sqrt(sum(v*v for v in quat))
  w,x,y,z=(v/norm for v in quat)
  return tuple(round(v,12) for v in (a*w-c*y, a*x+c*z, a*y+c*w, a*z-c*x))


def mount_pose(pitch_down_deg):
  return {'position_m':list(DEPTH_MOUNT_POS),
          'quaternion_wxyz':list(tilt_quaternion(LEGACY_DEPTH_QUAT,pitch_down_deg))}
