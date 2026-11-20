"""An ordinary Gaussian with a bounded mean, retaining exact PPO likelihoods.

Samples can exceed [-1, 1] and still pass through the environment clip. This is
deliberately NOT a squashed Gaussian; KL and log probability use the actual
Normal distribution. Export and deterministic replay use the same mean map.
"""
import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.modules.distribution import GaussianDistribution


class SoftsignMean(nn.Module):
  def forward(self,x:torch.Tensor)->torch.Tensor:
    return x/(1.+x.abs())


class BoundedMeanGaussian(GaussianDistribution):
  def __init__(self,output_dim,init_std=.25,std_type='scalar'):
    if std_type!='scalar' or not .05<=init_std<=.35:
      raise ValueError('BoundedMeanGaussian requires scalar std in [0.05, 0.35]')
    super().__init__(output_dim,init_std,std_type)

  def update(self,mlp_output):
    mean=self.deterministic_output(mlp_output)
    std=self.std_param.clamp(.05,.35).expand_as(mean)
    self._distribution=Normal(mean,std)

  def deterministic_output(self,mlp_output):
    return mlp_output/(1.+mlp_output.abs())

  def as_deterministic_output_module(self):
    return SoftsignMean()
