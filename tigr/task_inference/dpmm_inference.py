import torch
from torch import nn as nn
import torch.nn.functional as F
from tigr.task_inference.base_inference import DecoupledEncoder as BaseEncoder


class DecoupledEncoder(BaseEncoder):
    def __init__(self, *args, **kwargs):
        super(DecoupledEncoder, self).__init__(*args, **kwargs)

        # Define encoder model
        self.fc_mu = nn.Linear(self.shared_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(self.shared_dim, self.latent_dim)

        self.encode = self.encode_shared_y if self.encoding_mode == 'transitionSharedY' else self.encode_trajectory

    def forward(self, x, sampler='mean', return_probabilities=False):
        # Encode
        mu, log_var = self.encode(x)
        # Sample
        latent_samples = self.sample(mu, log_var)
        if self.bnp_model.model:
            probabilities, assignments = self.bnp_model.cluster_assignments(latent_samples)
            assignments = torch.as_tensor(assignments, dtype=torch.long)
        else:
            probabilities, assignments = None, torch.zeros([x.shape[0]], dtype=torch.long)

        # TODO: convert cluster assigments to true base task
        # The problem here is that the assigments do not match to tasks
        # e.g. We can have 20 clusters in the BNP model, so assignments
        # in set {0, 1, ..., 19}, whereas we only have 8 tasks.
        if not return_probabilities:
            # Calculate max class
            return latent_samples, assignments
            # Calculate probability
        else:
            return latent_samples, probabilities

    def encode_trajectory(self, x):

        # Compute shared encoder forward pass
        m = self.shared_encoder(x)

        # Compute class encoder forward pass
        mu = self.fc_mu(m)
        log_var = self.fc_log_var(m)
        return mu, log_var

    def encode_shared_y(self, x):
        raise NotImplementedError(f'Encoder mode "transitionSharedY" is not yet supported!')

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
