import json
import os
# import jsonlines
import numpy as np
import torch
import torch.nn as nn
from jsonlines import jsonlines

import rlkit.torch.pytorch_util as ptu

from rlkit.core import logger
from rlkit.launchers.launcher_util import dict_to_safe_json
from tqdm import tqdm

from tigr.trainer.base_trainer import AugmentedTrainer as BaseTrainer

import vis_utils.tb_logging as TB


def classification_accuracy(comps, targets):
    # this is  not the real accuracy, rather the "purity" of the
    # clusters, or rather an upperbound of the accuracy
    d = {}
    for comp, target in zip(comps, targets):
        if comp not in d:
            d[comp] = [target]
        else:
            d[comp].append(target)

    correct = 0
    for comp in d:
        task, count = np.unique(d[comp], return_counts=True)
        correct += max(count)
    acc = correct / len(comps)
    return acc


class AugmentedTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super(AugmentedTrainer, self).__init__(*args, **kwargs)

        self.optimizer_mixture_model = self.optimizer_class(
            [{'params': self.encoder.shared_encoder.parameters()},
             {'params': self.encoder.fc_mu.parameters()},
             {'params': self.encoder.fc_log_var.parameters()}],
            lr=self.lr_encoder
        )

        self.loss_ce = nn.CrossEntropyLoss(reduction='none')
        # self.prev_reward_loss = torch.zeros(self.batch_size)
        self.current_epoch = None

    def train(self, mixture_steps, w_method='val_value_based', current_epoch=0):
        self.current_epoch = current_epoch
        print("Epoch:", self.current_epoch)
        train_indices, val_indices = self.replay_buffer.get_train_val_indices(self.train_val_percent)

        # Reset lowest loss for mixture
        self.lowest_loss_epoch = 0
        self.lowest_loss = np.inf

        '''
        MIXTURE TRAINING EPOCHS
        '''
        mixture_training_step = 0
        for mixture_training_step in tqdm(range(mixture_steps), desc='Reconstruction trainer'):

            # Perform training step
            _, z = self.mixture_training_step(train_indices)
            # # train bnp_model for every step
            if self.encoder.bnp_model.fit_interval == 'step' and current_epoch >= self.encoder.bnp_model.start_epoch:
                self.encoder.bnp_model.fit(z)

            self._n_train_steps_mixture += 1

        logger.record_tabular('Mixture_steps', mixture_training_step + 1)

        '''
        DPMM TRAINING EPOCHS
        '''
        # train bnp_model for every epoch
        if self.encoder.bnp_model.fit_interval == 'epoch' and current_epoch >= self.encoder.bnp_model.start_epoch:
            self.encoder.bnp_model.fit(z)
            self.encoder.bnp_model.plot_clusters(z, suffix=str(current_epoch))

        return self.lowest_loss_epoch

    def mixture_training_step(self, indices):

        '''
        Computes a forward pass to encoder and decoder with sampling at the encoder
        '''

        # Get data from real replay buffer

        e_data, d_data = self.replay_buffer.sample_random_few_step_batch(indices, self.batch_size,
                                                               normalize=self.use_data_normalization,
                                                               prio='linear')

        # Prepare for usage in decoder
        actions = ptu.from_numpy(d_data['actions'])[:, 1:, :]
        states = ptu.from_numpy(d_data['observations'])[:, 1:, :]
        next_states = ptu.from_numpy(d_data['next_observations'])[:, 1:, :]
        rewards = ptu.from_numpy(d_data['rewards'])[:, 1:, :]   # torch.Size([batch_size_reconstruction, time_step, 1])
        terminals = ptu.from_numpy(d_data['terminals'])[:, 1:, :]

        # Remove last (trailing) dimension here
        true_task = np.array([a['base_task'] for a in d_data['true_tasks'][:, -1, 0]], dtype=np.int)
        unique_tasks = torch.unique(ptu.from_numpy(true_task).long()).tolist()
        targets = ptu.from_numpy(true_task).long()

        decoder_state_target = next_states[:, :, :self.state_reconstruction_clip]    # torch.Size([batch_size_reconstruction, time_step, state_dim])

        '''
        MIXTURE MODEL TRAINING
        '''

        # Prepare for usage in encoder
        encoder_input = self.replay_buffer.make_encoder_data(e_data, self.batch_size)

        # Forward pass through encoder
        mu, log_var = self.encoder.encode(encoder_input)
        # latent_distributions, alpha, beta = self.encoder.encode(encoder_input)
        assert not torch.isnan(mu).any(), mu
        assert not torch.isnan(log_var).any(), log_var

        # Sample latent variables
        latent_variables = self.encoder.sample(mu, log_var)

        '''
        Dynamics Prediction Loss
        '''
        # Calculate standard losses
        # Put in decoder to get likelihood
        # Decoding for all time steps

        state_estimate, reward_estimate = self.decoder(states, actions, decoder_state_target,
                                                       latent_variables.unsqueeze(1).repeat(1, states.shape[1], 1))
        mixture_state_loss = torch.mean((decoder_state_target - state_estimate) ** 2, dim=[-2, -1])
        mixture_reward_loss = torch.mean((rewards - reward_estimate) ** 2, dim=[-2, -1])

        # if torch.mean(mixture_reward_loss - self.prev_reward_loss.to(mixture_reward_loss.device)) > 0.01:
        #     # print("change in reward loss", torch.mean(mixture_reward_loss - self.prev_reward_loss))
        #     data["reward_estimates"] = reward_estimate.detach().cpu().tolist()
        #     data["epoch"] = self.current_epoch
        #     data["mixture_reward_loss"] = mixture_reward_loss.detach().cpu().tolist()
        #     data["prev_reward_loss"] = self.prev_reward_loss.detach().cpu().tolist()
        #     to_json = dict_to_safe_json({self._n_train_steps_mixture: data})
        #     with jsonlines.open(os.path.join(self.log_dir, "decoder_debug.jl"), mode="a") as f:
        #         f.write(to_json)
        #     # print(to_json)
        # self.prev_reward_loss = mixture_reward_loss

        mixture_nll = self.loss_weight_state * mixture_state_loss + self.loss_weight_reward * mixture_reward_loss
        assert not torch.isnan(latent_variables).any(), latent_variables
        assert not torch.isnan(state_estimate).any(), state_estimate
        assert not torch.isnan(reward_estimate).any(), reward_estimate
        assert not torch.isnan(mixture_state_loss).any(), mixture_state_loss
        assert not torch.isnan(mixture_reward_loss).any(), mixture_reward_loss

        # Calculate extra losses
        # KL divergence based on dpmm clusters
        if not self.encoder.bnp_model.model:
            kl_qz_pz = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)
            classification_acc = 0
        else:
            prob_comps, comps = self.encoder.bnp_model.cluster_assignments(latent_variables)
            var = torch.exp(0.5 * log_var)**2
            if self.encoder.bnp_model.kl_method == 'soft':
                # get a distribution of the latent variable
                dist = torch.distributions.MultivariateNormal(
                    loc=mu.cpu(),
                    covariance_matrix=torch.diag_embed(var).cpu()
                )
                # get a distribution for each cluster
                B, K = prob_comps.shape # batch_shape, number of active clusters
                kl_qz_pz = torch.zeros(B)
                for k in range(K):
                      prob_k = prob_comps[:, k]
                      dist_k = torch.distributions.MultivariateNormal(
                          loc=self.encoder.bnp_model.comp_mu[k],
                          covariance_matrix=torch.diag_embed(self.encoder.bnp_model.comp_var[k])
                      )
                      expanded_dist_k = dist_k.expand(dist.batch_shape)    # batch_shape [batch_size], event_shape [latent_dim]
                      kld_k = torch.distributions.kl_divergence(dist, expanded_dist_k)   #  shape [batch_shape, ]
                      kl_qz_pz += torch.from_numpy(prob_k) * kld_k

            else:
                # calcualte kl divergence via hard assignment: assigning to the most  likely learned DPMM cluster
                mu_comp = torch.zeros_like(mu)
                var_comp = torch.zeros_like(log_var)
                for i, k in enumerate(comps):
                    mu_comp[i, :] = self.encoder.bnp_model.comp_mu[k]
                    var_comp[i, :] = self.encoder.bnp_model.comp_var[k]
                var = torch.exp(0.5 * log_var)**2
                kl_qz_pz = self.encoder.bnp_model.kl_divergence_diagonal_gaussian(mu, mu_comp, var, var_comp)
            classification_acc = classification_accuracy(comps, true_task)

        clustering_loss = self.alpha_kl_z * kl_qz_pz

        # Overall elbo, but weight KL div takes up self.alpha_kl_z fraction of the loss!
        elbo = mixture_nll + clustering_loss.to(mixture_nll.device)

        # Take mean over each true task so one task won't dominate
        mixture_loss = torch.sum(elbo)
        assert not torch.isnan(mixture_nll).any(), mixture_nll
        assert not torch.isnan(kl_qz_pz).any(), kl_qz_pz
        assert not torch.isnan(elbo).any(), elbo
        assert not torch.isnan(mixture_loss).any(), mixture_loss

        if self.use_PCGrad:
            # Find according class for every sample
            if self.PCGrad_option == 'true_task':
                task_indices = targets
            elif self.PCGrad_option == 'most_likely_task':
                task_indices = torch.argmax(latent_variables, dim=-1)
            elif self.PCGrad_option == 'random_prob_task':
                task_indices = torch.distributions.categorical.Categorical(latent_variables).sample()
            else:
                raise NotImplementedError(f'Option {self.PCGrad_option} for PCGrad was not implemented yet.')

            # Group all elements according to class, also elbo should be maximized, and backward function assumes minimization
            per_class_total_loss = [torch.sum(elbo[task_indices == current_class]) for current_class in unique_tasks]

            self.PCGrad_mixture_model_optimizer.minimize(per_class_total_loss)

        else:
            # Optimize mixture model first and afterwards do the same with activation encoder
            self.optimizer_mixture_model.zero_grad()
            self.optimizer_decoder.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            mixture_loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            self.optimizer_mixture_model.step()
            self.optimizer_decoder.step()

        total_state_loss = ptu.get_numpy(torch.sum(mixture_state_loss)) / self.batch_size
        total_reward_loss = ptu.get_numpy(torch.sum(mixture_reward_loss)) / self.batch_size

        if TB.LOG_INTERVAL > 0 and TB.TI_LOG_STEP % TB.LOG_INTERVAL == 0:
            # Write new stats to TB
            # Normalize all with batch size
            TB.TENSORBOARD_LOGGER.add_scalar('training/ti_mixture_loss',
                                             (torch.sum(mixture_loss) / self.batch_size).item(),
                                             global_step=TB.TI_LOG_STEP)
            TB.TENSORBOARD_LOGGER.add_scalar('training/ti_mixture_elbo_loss',
                                             (torch.sum(elbo) / self.batch_size).item(), global_step=TB.TI_LOG_STEP)
            TB.TENSORBOARD_LOGGER.add_scalar('training/ti_mixture_state_losses', total_state_loss.item(),
                                             global_step=TB.TI_LOG_STEP)
            TB.TENSORBOARD_LOGGER.add_scalar('training/ti_mixture_reward_losses', total_reward_loss.item(),
                                             global_step=TB.TI_LOG_STEP)

            TB.TENSORBOARD_LOGGER.add_scalar('training/ti_mixture_nll',
                                             (torch.sum(mixture_nll) / self.batch_size).item(),
                                             global_step=TB.TI_LOG_STEP)

            TB.TENSORBOARD_LOGGER.add_scalar('training/ti_mixture_clustering_losses',
                                             (torch.sum(clustering_loss) / self.batch_size).item(),
                                             global_step=TB.TI_LOG_STEP)

            TB.TENSORBOARD_LOGGER.add_scalar('training/ti_mixture_klz_loss',
                                             (torch.sum(kl_qz_pz) / self.batch_size).item(), global_step=TB.TI_LOG_STEP)
            # TB.TENSORBOARD_LOGGER.add_scalar('training/ti_mixture_sparsity_loss', (torch.sum(sparsity_loss) / self.batch_size).item(), global_step=TB.TI_LOG_STEP)
            # if self.num_classes > 1:
            #     TB.TENSORBOARD_LOGGER.add_scalar('training/ti_mixture_euclid_loss', (torch.sum(euclid_loss) / self.batch_size).item(), global_step=TB.TI_LOG_STEP)

            # TB.TENSORBOARD_LOGGER.add_scalar('training/ti_classification_acc', (torch.argmax(gammas, dim=-1) == targets).float().mean().item(), global_step=TB.TI_LOG_STEP)
            # TODO: Check the accuracy calculation
            TB.TENSORBOARD_LOGGER.add_scalar('training/ti_classification_acc',
                                             classification_acc,
                                             global_step=TB.TI_LOG_STEP)
            # if self.use_regularization_loss:
            #     TB.TENSORBOARD_LOGGER.add_scalar('training/ti_mixture_regularization_loss', reg_loss.mean().item(), global_step=TB.TI_LOG_STEP)
        TB.TI_LOG_STEP += 1

        return ((torch.sum(mixture_loss) / self.batch_size),
                total_state_loss,
                total_reward_loss), latent_variables

    def validate_mixture(self, indices):

        # Get data from real replay buffer
        e_data, d_data = self.replay_buffer.sample_random_few_step_batch(indices, self.batch_size,
                                                               normalize=self.use_data_normalization)

        # Prepare for usage in decoder
        actions = ptu.from_numpy(d_data['actions'])[:, -1, :]
        states = ptu.from_numpy(d_data['observations'])[:, -1, :]
        next_states = ptu.from_numpy(d_data['next_observations'])[:, -1, :]
        rewards = ptu.from_numpy(d_data['rewards'])[:, -1, :]
        terminals = ptu.from_numpy(d_data['terminals'])[:, -1, :]

        # Remove last (trailing) dimension here
        true_task = np.array([a['base_task'] for a in d_data['true_tasks'][:, -1, 0]], dtype=np.int)
        targets = ptu.from_numpy(true_task).long()

        decoder_state_target = next_states[:, :self.state_reconstruction_clip]

        '''
        MIXTURE MODEL
        '''

        with torch.no_grad():
            # Prepare for usage in encoder
            encoder_input = self.replay_buffer.make_encoder_data(e_data, self.batch_size)
            # Forward pass through encoder
            mu, log_var = self.encoder.encode(encoder_input)
            # latent_distributions, alpha, beta = self.encoder.encode(encoder_input)
            assert not torch.isnan(mu).any(), mu
            assert not torch.isnan(log_var).any(), log_var

            # Sample latent variables
            latent_variables = self.encoder.sample(mu, log_var)

            '''
            Dynamics Prediction Loss
            '''

            # Calculate standard losses
            # Put in decoder to get likelihood
            state_estimate, reward_estimate = self.decoder(states, actions, decoder_state_target, latent_variables)
            mixture_state_loss = torch.mean((state_estimate - decoder_state_target) ** 2, dim=-1)
            mixture_reward_loss = torch.mean((reward_estimate - rewards) ** 2, dim=-1)

            mixture_nll = self.loss_weight_state * mixture_state_loss + self.loss_weight_reward * mixture_reward_loss

            assert not torch.isnan(latent_variables).any(), latent_variables
            assert not torch.isnan(state_estimate).any(), state_estimate
            assert not torch.isnan(reward_estimate).any(), reward_estimate
            assert not torch.isnan(mixture_state_loss).any(), mixture_state_loss
            assert not torch.isnan(mixture_reward_loss).any(), mixture_reward_loss

            # KL divergence based on dpmm clusters
            if not self.encoder.bnp_model.model:
                kl_qz_pz = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)
            else:
                prob_comps, comps = self.encoder.bnp_model.cluster_assignments(latent_variables)
                var = torch.exp(0.5 * log_var) ** 2
                if self.encoder.bnp_model.kl_method == 'soft':
                    # get a distribution of the latent variable
                    dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=torch.diag_embed(var))
                    # get a distribution for each cluster
                    B, K = prob_comps.shape  # batch_shape, number of active clusters
                    kl_qz_pz = torch.zeros(B)
                    for k in range(K):
                        prob_k = prob_comps[:, k]
                        dist_k = torch.distributions.MultivariateNormal(
                            loc=self.encoder.bnp_model.comp_mu[k].to(mu.device),
                            covariance_matrix=torch.diag_embed(self.encoder.bnp_model.comp_var[k]).to(var.device)
                        )
                        expanded_dist_k = dist_k.expand(
                            dist.batch_shape)  # batch_shape [batch_size], event_shape [latent_dim]
                        kld_k = torch.distributions.kl_divergence(dist, expanded_dist_k)  # shape [batch_shape, ]
                        kl_qz_pz += torch.from_numpy(prob_k) * kld_k
                else:
                    # calcualte kl divergence via hard assignment: assigning to the most  likely learned DPMM cluster
                    mu_comp = torch.zeros_like(mu)
                    var_comp = torch.zeros_like(log_var)
                    for i, k in enumerate(comps):
                        mu_comp[i, :] = self.encoder.bnp_model.comp_mu[k]
                        var_comp[i, :] = self.encoder.bnp_model.comp_var[k]
                    var = torch.exp(0.5 * log_var) ** 2
                    kl_qz_pz = self.encoder.bnp_model.kl_divergence_diagonal_gaussian(mu, mu_comp, var, var_comp)

            # Overall elbo, but weight KL div takes up self.alpha_kl_z fraction of the loss!
            elbo = - mixture_nll - self.alpha_kl_z * kl_qz_pz
            # Take mean over each true task so one task won't dominate
            mixture_loss = -torch.sum(elbo)
            assert not torch.isnan(mixture_nll).any(), mixture_nll
            assert not torch.isnan(kl_qz_pz).any(), kl_qz_pz
            assert not torch.isnan(elbo).any(), elbo
            assert not torch.isnan(mixture_loss).any(), mixture_loss

        return (ptu.get_numpy(mixture_loss) / self.batch_size,
                ptu.get_numpy(torch.sum(mixture_state_loss)) / self.batch_size,
                ptu.get_numpy(torch.sum(mixture_reward_loss)) / self.batch_size,
                0.)

