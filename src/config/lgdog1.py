"""LgDog1 nominal simulation parameters used by the velocity tasks.

These values preserve the existing mjlab configuration. They are simulation
settings, not a new measurement or validation of the physical robot.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class JointControl:
  target_names_expr: tuple[str, ...]
  stiffness: float
  damping: float
  effort_limit: float
  armature: float
  frictionloss: float


HIP = JointControl((r".*_hip_joint",), 20.0, 1.0, 10.0, 0.01, 0.2)
THIGH = JointControl((r".*_thigh_joint",), 20.0, 1.0, 10.0, 0.01, 0.2)
CALF = JointControl((r".*_calf_joint",), 40.0, 2.0, 20.0, 0.02, 0.2)

INITIAL_POSITION = (0.0, 0.0, 0.31)
INITIAL_JOINT_POSITIONS = (
  (r".*_hip_joint", 0.0),
  (r".*_thigh_joint", 0.0),
  (r".*_calf_joint", 0.2),
)
INITIAL_JOINT_VELOCITY = 0.0
SOFT_JOINT_POS_LIMIT_FACTOR = 0.9
ACTION_SCALE = 0.25
