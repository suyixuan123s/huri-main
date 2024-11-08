import copy
import time

import wandb
import torch
import torch.nn.functional as F
import numpy as np

from pathlib import Path
from huri.learning.env.rack_v3.env import RackState

prior_eps: float = 1e-6


class Learner(object):
    def __init__(self,
                 env,
                 critic_net,
                 actor_net,
                 shared_net,
                 shared_state,
                 replay_buffer,
                 cfg,
                 save_path=None):
        """
        [Important] Learner is assumed to run in the main thread. (Pytorch has some strange bug)

        :param args: Args have following parameters:
                    - lr: learning rate
                    - device: cpu or gpu
        :param replay_buffer: Shared replay buffer
        """
        self.env = env
        state = self.env.reset()
        self.abs_state_dim = state.abs_size
        self.env_action_dim = self.env.action_space_dim
        self.shared_net = shared_net
        self.shared_state = shared_state
        self.replay_buffer = replay_buffer
        # cfg
        self.c_lr = cfg['c_lr']
        self.a_lr = cfg['a_lr']
        self.gamma = cfg['gamma']
        self.target_update_freq = cfg['update_freq']
        self.save_period = cfg['save_period']
        self.device = cfg['device']
        self.save_checkpnt_path = Path(save_path).joinpath('model.chkpt') if save_path is not None else None

        # deep learning
        ## nn
        self.critic = critic_net
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()
        self.actor = actor_net
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.eval()
        # optimizer
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr, weight_decay=1e-6)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr, weight_decay=1e-6)

        # TD3
        self.policy_noise = cfg['policy_noise_gaussian_stdv']
        self.policy_noise_clip = cfg['policy_noise_clip']
        # self.policy_freq = policy_freq

        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
        #                                                             mode='min',
        #                                                             factor=0.1,
        #                                                             patience=1000,
        #                                                             cooldown=0,
        #                                                             verbose=True)

    def _compute_loss(self, samples):
        device = self.device  # for shortening the following lines
        gamma = self.gamma
        action_dim = self.env_action_dim
        goal = torch.as_tensor(samples["goal"], dtype=torch.float32, device=device)
        state = torch.as_tensor(samples["obs"], dtype=torch.float32, device=device)
        next_state = torch.as_tensor(samples["next_obs"], dtype=torch.float32, device=device)
        action = torch.as_tensor(samples["acts"].reshape(-1, 1), dtype=torch.int64, device=device)
        reward = torch.as_tensor(samples["rews"].reshape(-1, 1), dtype=torch.float32, device=device)
        done = torch.as_tensor(samples["done"].reshape(-1, 1), dtype=torch.float32, device=device)
        c_abs_state = torch.as_tensor(samples["c_abs_obs"], dtype=torch.float32, device=device)
        c_next_abs_state = torch.as_tensor(samples["c_abs_next_obs"], dtype=torch.float32, device=device)
        abs_state = decompress_abs_state(c_abs_state, self.abs_state_dim[0])
        abs_next_state = decompress_abs_state(c_next_abs_state, self.abs_state_dim[0])

        # ------------------------------

        with torch.no_grad():
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.policy_noise_clip, self.policy_noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            mask = 1 - done
            target_Q = (reward + gamma * target_Q * mask).to(self.device)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = (F.mse_loss(current_Q1, target_Q, reduction='none'),
                       F.mse_loss(current_Q2, target_Q, reduction='none'))

        # -----------------------------------------------------------------------------------
        return critic_loss

        next_state_feasible_action_set = torch.as_tensor(
            np.array([padding(action_dim,
                              RackState.get_feasible_action_set(next_obs_tmp.cpu().numpy())
                              ) for next_obs_tmp in samples["next_obs"]]),
            dtype=torch.int64, device=device)
        curr_q_table = self.dqn(state, goal, abs_state)
        curr_q_value = curr_q_table.gather(1, action)

        argmax_next_state_feasible_action = next_state_feasible_action_set.gather(1,
                                                                                  self.dqn(next_state,
                                                                                           goal,
                                                                                           abs_next_state).detach().gather(
                                                                                      1,
                                                                                      next_state_feasible_action_set)
                                                                                  .argmax(dim=1, keepdim=True))
        next_q_value = self.dqn_target(next_state, goal, abs_next_state).detach().gather(  # Double DQN
            1, argmax_next_state_feasible_action).detach()
        mask = 1 - done
        target = (reward + gamma * next_q_value * mask).to(self.device)
        # calculate element-wise dqn loss
        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")
        return elementwise_loss

    def update_model(self):
        """Update the model by gradient descent."""
        samples = self.replay_buffer.sample_batch()
        weights = samples["weights"].reshape(-1, 1)
        indices = samples["indices"]
        # update loss
        self.critic_optimizer.zero_grad()
        critic_loss1, critic_loss2 = self._compute_loss(samples)
        critic_loss = torch.mean(critic_loss1 * weights + critic_loss2 * weights)
        # torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 2)
        critic_loss.backward()
        self.critic_optimizer.step()
        # update priorities
        loss_for_prior = torch.mean(critic_loss1.detach() * weights)
        new_priorities = loss_for_prior.flatten() + prior_eps
        self.replay_buffer.update_priorities(indices, new_priorities)

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return loss.item()

    def train(self, num_epoch, min_replay_sz=1000, output_interval=1000):
        # wait for the replay buffer have enough samples
        while self.replay_buffer.size <= min_replay_sz:
            time.sleep(.5)

        # start training
        loss_total = 0
        for epoch in range(1, num_epoch + 1):
            st = time.time()
            loss = self.update_model()
            c_time = time.time() - st
            loss_total += loss

            if epoch % self.target_update_freq == 0:
                self.dqn_target.load_state_dict(self.dqn.state_dict())

            # TODO update shared network
            self.shared_net.load_state_dict(self.dqn.state_dict())

            # TODO update learning rate
            # if epoch > 3000:  # warmup
            #     self.scheduler.step(loss)

            # save checkpnnt
            if epoch % self.save_period == 0:
                if self.save_checkpnt_path is not None:
                    torch.save({
                        'dqn_state_dict': self.dqn.state_dict(),
                        'dqn_target_state_dict': self.dqn_target.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }, self.save_checkpnt_path)

            # logging
            wandb.log({'learner_train_loss': loss, 'learner_computation_time': c_time,
                       'learner_learning_rate': self.optimizer.param_groups[0]['lr']})
            if epoch % output_interval == 0:
                print(f"[Learner]"
                      f"Epoch: {epoch}, "
                      f"Loss average: {loss_total / output_interval}, ")
                loss_total = 0
