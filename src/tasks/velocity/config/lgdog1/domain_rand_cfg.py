"""Domain randomization shared by race MGDP training and perception collection."""

import math
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg


RACE_MASS_INERTIA_SCALE_RANGE = (0.9, 1.1)


def add_race_mass_inertia_randomization(cfg):
  """Scale trunk mass and its complete inertia tensor by the same factor.

  Native pseudo-inertia uses scale=exp(2*alpha). Only density scaling is
  enabled: no shape/shear/COM perturbation, and no leg body is selected.
  The resulting scale is log-uniform in [0.9, 1.1].
  """
  event = EventTermCfg(func=dr.pseudo_inertia, mode="startup", params={
    "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
    "alpha_range": tuple(.5*math.log(s) for s in RACE_MASS_INERTIA_SCALE_RANGE),
  })
  # pseudo_inertia reconstructs from nominal COM. Run it before base_com so
  # the existing COM-offset randomization is retained, not overwritten.
  cfg.events = {"base_mass_inertia": event, **cfg.events}
