import copy
import time

import wandb
import torch
import torch.nn.functional as F
import numpy as np

from pathlib import Path
from huri.learning.env.rack_v3.env import RackState

prior_eps: float = 1e-6


def padding(size, array):
    if len(array) == 0:
        return np.zeros(size, dtype=int)
    max_v = max(array)
    pad_array = np.ones(size, dtype=int) * max_v
    pad_array[:len(array)] = array
    return pad_array


def decompress_abs_state(compressed, n):
    # Decompress the matrix by constructing a symmetric matrix
    decompressed = torch.zeros((compressed.shape[0], n, n), dtype=compressed.dtype, device=compressed.device)
    indices = torch.tril_indices(n, n)
    decompressed[:, indices[0], indices[1]] = compressed
    decompressed = decompressed + decompressed.transpose(2, 1)
    return decompressed


class Learner(object):
    def __init__(self,
                 env,
                 net,
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
        self.lr = cfg['lr']
        self.gamma = cfg['gamma']
        self.tau = cfg['tau']
        self.target_update_freq = cfg['update_freq']
        self.save_period = cfg['save_period']
        self.device = cfg['device']
        self.save_checkpnt_path = Path(save_path).joinpath('model.chkpt') if save_path is not None else None

        # deep learning
        ## nn
        self.dqn = net
        self.dqn_target = copy.deepcopy(self.dqn)
        self.dqn.train()
        self.dqn_target.eval()
        # optimizer
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.lr, weight_decay=1e-6)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode='min',
                                                                    factor=0.1,
                                                                    patience=1000,
                                                                    cooldown=0,
                                                                    verbose=True)

    def _compute_loss2(self, samples):
        device = self.device  # for shortening the following lines
        gamma = self.gamma
        action_dim = self.env_action_dim
        goal = torch.as_tensor(samples["goal"], dtype=torch.float32, device=device)
        state = torch.as_tensor(samples["obs"], dtype=torch.float32, device=device)
        next_state = torch.as_tensor(samples["next_obs"], dtype=torch.float32, device=device)
        action = torch.as_tensor(samples["acts"].reshape(-1, 1), dtype=torch.int64, device=device)
        reward = torch.as_tensor(samples["rews"].reshape(-1, 1), dtype=torch.float32, device=device)
        done = torch.as_tensor(samples["done"].reshape(-1, 1), dtype=torch.float32, device=device)
        if "c_abs_obs" in samples:
            c_abs_state = torch.as_tensor(samples["c_abs_obs"], dtype=torch.float32, device=device)
            c_next_abs_state = torch.as_tensor(samples["c_abs_next_obs"], dtype=torch.float32, device=device)
            abs_state = decompress_abs_state(c_abs_state, self.abs_state_dim[0])
            abs_next_state = decompress_abs_state(c_next_abs_state, self.abs_state_dim[0])
        else:
            abs_state = None
            abs_next_state = None

        with torch.no_grad():
            next_state_feasible_action_set = torch.as_tensor(
                np.array([padding(action_dim,
                                  RackState.get_feasible_action_set(next_obs_tmp.cpu().numpy())
                                  ) for next_obs_tmp in samples["next_obs"]]),
                dtype=torch.int64, device=device)
            Q1, Q2 = self.dqn.Q(next_state, goal, abs_next_state)
            target_Q1, target_Q2 = self.dqn_target.Q(next_state, goal, abs_next_state)
            argmax_next_state_feasible_action1 = next_state_feasible_action_set.gather(1, Q1.gather(1,
                                                                                                   next_state_feasible_action_set)
                                                                                      .argmax(dim=1, keepdim=True))
            argmax_next_state_feasible_action2 = next_state_feasible_action_set.gather(1, Q2.gather(1,
                                                                                                    next_state_feasible_action_set)
                                                                                       .argmax(dim=1, keepdim=True))

            target_Q = torch.min(target_Q1.gather(1, argmax_next_state_feasible_action1),
                                 target_Q2.gather(1, argmax_next_state_feasible_action2))
            mask = 1 - done
            target = (reward + gamma * target_Q * mask).to(self.device)
        curr_q_table1, curr_q_table2 = self.dqn.Q(state, goal, abs_state)
        curr_q_value1 = curr_q_table1.gather(1, action)
        curr_q_value2 = curr_q_table2.gather(1, action)
        elementwise_loss = F.smooth_l1_loss(curr_q_value1, target, reduction="none") + \
                           F.smooth_l1_loss(curr_q_value2, target, reduction="none")
        return elementwise_loss

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
        if "c_abs_obs" in samples:
            c_abs_state = torch.as_tensor(samples["c_abs_obs"], dtype=torch.float32, device=device)
            c_next_abs_state = torch.as_tensor(samples["c_abs_next_obs"], dtype=torch.float32, device=device)
            abs_state = decompress_abs_state(c_abs_state, self.abs_state_dim[0])
            abs_next_state = decompress_abs_state(c_next_abs_state, self.abs_state_dim[0])
        else:
            abs_state = None
            abs_next_state = None

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
        self.optimizer.zero_grad()
        # elementwise_loss = self._compute_loss2(samples)
        # loss = torch.mean(elementwise_loss * weights) / 2
        elementwise_loss = self._compute_loss(samples)
        loss = torch.mean(elementwise_loss * weights)
        # torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 2)
        loss.backward()
        self.optimizer.step()

        # update priorities
        loss_for_prior = elementwise_loss.detach()
        new_priorities = loss_for_prior.flatten() + prior_eps
        self.replay_buffer.update_priorities(indices, new_priorities)
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
                # self.dqn_target.load_state_dict(self.dqn.state_dict())
                for param, target_param in zip(self.dqn.parameters(), self.dqn_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # Update shared network
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
