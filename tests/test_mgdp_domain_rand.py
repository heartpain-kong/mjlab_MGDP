"""Mass/inertia physics invariants, task isolation and event ordering."""

import math
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch
from mjlab.utils.lab_api.math import matrix_from_quat
from src.tasks.velocity.config.lgdog1.domain_rand_cfg import RACE_MASS_INERTIA_SCALE_RANGE
from src.tasks.velocity.config.lgdog1.mgdp_env_cfg import lgdog1_race_mgdp_env_cfg,lgdog1_mgdp_env_cfg


class MGDPMassInertiaTests(unittest.TestCase):
  def test_race_training_scope_startup_order_and_exact_bounds(self):
    cfg=lgdog1_race_mgdp_env_cfg(with_perception=False)
    event=cfg.events["base_mass_inertia"]
    self.assertEqual(event.mode,"startup")
    self.assertEqual(event.params["asset_cfg"].body_names,("trunk",))
    self.assertEqual(RACE_MASS_INERTIA_SCALE_RANGE,(.9,1.1))
    for alpha,scale in zip(event.params["alpha_range"],(.9,1.1)):
      self.assertAlmostEqual(math.exp(2*alpha),scale)
    self.assertLess(list(cfg.events).index("base_mass_inertia"),list(cfg.events).index("base_com"))
    self.assertNotIn("base_mass_inertia",lgdog1_race_mgdp_env_cfg(play=True,with_perception=False).events)
    self.assertNotIn("base_mass_inertia",lgdog1_mgdp_env_cfg(with_perception=False).events)

  def test_race_perception_collection_uses_same_randomization(self):
    from scripts import pretrain_mgdp
    for task,expected in (("LgDog1-Race-MGDP",True),("LgDog1-Rough-MGDP",False)):
      args=NS(task=task,num_envs=2,seed=42,terrain_rows=2,device="cpu")
      with patch.object(pretrain_mgdp,"ManagerBasedRlEnv") as create:
        pretrain_mgdp.build_env(args)
      cfg=create.call_args.kwargs["cfg"]
      self.assertEqual("base_mass_inertia" in cfg.events,expected)
      if expected:
        self.assertEqual(cfg.events["base_mass_inertia"].params["alpha_range"],
          lgdog1_race_mgdp_env_cfg(with_perception=False).events["base_mass_inertia"].params["alpha_range"])

  def test_native_density_scaling_couples_inertia_and_leaves_other_bodies_unchanged(self):
    n=32
    defaults={
      "body_mass":torch.tensor([3.487,.9]),
      "body_ipos":torch.tensor([[-.21,.003,.004],[.02,0.,0.]]),
      "body_inertia":torch.tensor([[.09,.075,.019],[.003,.003,.001]]),
      "body_iquat":torch.tensor([[0.,2**-.5,0.,2**-.5],[1.,0.,0.,0.]]),
    }
    fields={k:v.unsqueeze(0).expand(n,*v.shape).clone() for k,v in defaults.items()}
    env=NS(num_envs=n,device="cpu",scene={"robot":NS(indexing=NS(body_ids=torch.arange(2,dtype=torch.int)))},
      sim=NS(model=NS(**fields),get_default_field=lambda name:defaults[name]))
    event=lgdog1_race_mgdp_env_cfg(with_perception=False).events["base_mass_inertia"]
    event.params["asset_cfg"].body_ids=[0]
    with torch.random.fork_rng(devices=[]):
      torch.manual_seed(42)
      event.func(env,None,**event.params)
      self.check_scaling(fields,defaults)
      # A repeated call starts from nominal values; the factor does not compound.
      before={k:v.clone() for k,v in fields.items()}
      ids=torch.tensor([1,7,21])
      event.func(env,ids,**event.params)
      self.check_scaling(fields,defaults)
      mask=torch.ones(n,dtype=torch.bool);mask[ids]=False
      for key in fields: torch.testing.assert_close(fields[key][mask],before[key][mask])

  def check_scaling(self,fields,defaults):
    factor=fields["body_mass"][:,0]/defaults["body_mass"][0]
    self.assertTrue(((factor>=.9-1e-6)&(factor<=1.1+1e-6)).all())
    self.assertGreater(factor.std().item(),.005)
    rotation=matrix_from_quat(fields["body_iquat"][:,0])
    inertia=rotation@torch.diag_embed(fields["body_inertia"][:,0])@rotation.mT
    nominal_rotation=matrix_from_quat(defaults["body_iquat"][0])
    nominal=nominal_rotation@torch.diag(defaults["body_inertia"][0])@nominal_rotation.T
    torch.testing.assert_close(inertia,factor[:,None,None]*nominal,rtol=2e-4,atol=1e-7)
    torch.testing.assert_close(fields["body_ipos"][:,0],defaults["body_ipos"][0].expand_as(fields["body_ipos"][:,0]),rtol=1e-5,atol=1e-6)
    self.assertTrue((fields["body_inertia"][:,0]>0).all())
    for key in fields:
      torch.testing.assert_close(fields[key][:,1],defaults[key][1].expand_as(fields[key][:,1]))


if __name__=="__main__":
  unittest.main()
