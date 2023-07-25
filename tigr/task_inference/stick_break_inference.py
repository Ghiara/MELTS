import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F

from tigr.task_inference.base_inference import DecoupledEncoder as BaseEncoder


class DecoupledEncoder(BaseEncoder):
	def __init__(self, *args, **kwargs):
		super(DecoupledEncoder, self).__init__(*args, **kwargs)

		# Define encoder model
		self.fc_alpha = nn.Sequential(
			nn.Linear(self.shared_dim, self.latent_dim),
			nn.Softplus()
		)
		self.fc_beta = nn.Sequential(
			nn.Linear(self.shared_dim, self.latent_dim),
			nn.Softplus()
		)

		self.encode = self.encode_shared_y if self.encoding_mode == 'transitionSharedY' else self.encode_trajectory

	def forward(self, x, sampler='mean', return_probabilities=False):
		# Encode
		latent_distributions, alpha, beta = self.encode(x)
		# Sample
		latent_samples = self.sample(latent_distributions, sampler=sampler, alpha=alpha, beta=beta)

		if not return_probabilities:
			# Calculate max class
			return latent_samples, torch.argmax(latent_samples, dim=-1)
		else:
			return latent_samples, latent_samples

	def encode_trajectory(self, x):

		# Compute shared encoder forward pass
		m = self.shared_encoder(x)

		alpha = self.fc_alpha(m)
		beta = self.fc_beta(m)
		return torch.distributions.kumaraswamy.Kumaraswamy(alpha, beta), alpha, beta

	def encode_shared_y(self, x):
		raise NotImplementedError(f'Encoder mode "transitionSharedY" is not yet supported!')

	def sample(self, latent_distributions, sampler='random', alpha=None, beta=None):
		# Select from which Gaussian to sample
		if sampler == 'random':
			# Sample from specified Gaussian using reparametrization trick
			# v = latent_distributions.rsample()
			u = torch.FloatTensor(alpha.shape).uniform_(0.01, 0.99).to(alpha.device)
			v = (1 - u.pow(1 / beta)).pow(1 / alpha)    # formula for the CDF
		else:
			# v = latent_distributions.mean
			v = (1 - latent_distributions.mean.pow(1 / beta)).pow(1 / alpha)

		# set Kth fraction v_i,K to one to ensure the stick segments sum to one
		if v.ndim > 2:
			v = v.squeeze()
		v0 = v[:, -1].pow(0).reshape(v.shape[0], 1)
		v1 = torch.cat([v[:, :self.latent_dim - 1], v0], dim=1)

		# get stick segements
		n_samples = v1.size()[0]
		n_dims = v1.size()[1]
		pi = torch.zeros((n_samples, n_dims)).to(alpha.device)

		for k in range(n_dims):
			if k == 0:
				pi[:, k] = v1[:, k]
			else:
				pi[:, k] = v1[:, k] * torch.stack([(1 - v[:, j]) for j in range(n_dims) if j < k]).prod(axis=0)

		# ensure stick segments sum to 1
		np.testing.assert_almost_equal(torch.ones(n_samples), pi.sum(axis=1).detach().cpu().numpy(),
									   decimal=2, err_msg='stick segments do not sum to 1')
		return pi



