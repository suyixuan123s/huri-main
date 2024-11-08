#!/usr/bin/env python
import copy
from typing import Dict
import torch
import time
import numpy as np
import torch.nn.functional as F
from pathlib import Path

from huri.learning.network.d3qn_attention import DuelingDQNMini


def padding(size, array):
    if len(array) == 0:
        return np.zeros(size, dtype=int)
    max_v = max(array)
    pad_array = np.ones(size, dtype=int) * max_v
    pad_array[:len(array)] = array
    return pad_array


prior_eps: float = 1e-6


class Learner(object):
    def __init__(self,
                 shared_net,
                 shared_state,
                 shared_replay_memory,
                 env_obs_dim,
                 env_act_dim,
                 env_n_classes,
                 env_abs_state_slice,
                 env_abs_goal_slice,
                 env_to_state_func,
                 lr: int,
                 gamma,
                 target_update_freq: int,
                 min_replay_mem_size: int,
                 device, ):
        # learning rate
        self.lr = lr
        # shared net
        self.shared_net = shared_net
        # device
        self.device = device
        #
        # if self.params['load_saved_state']:
        #     try:
        #         saved_state = torch.load(self.params['load_saved_state'])
        #         self.dqn.load_state_dict(saved_state['Q_state'])
        #     except FileNotFoundError:
        #         print("WARNING: No trained model found. Training from scratch")
        self.shared_state = shared_state

        # shared replay buffer
        self.replay_memory = shared_replay_memory

        # hyper-parameters
        self.target_update_freq = target_update_freq
        self.min_replay_mem_size = min_replay_mem_size
        self.gamma = gamma

        # parameters
        self.env_obs_dim = env_obs_dim
        self.env_act_dim = env_act_dim
        self.env_n_classes = env_n_classes
        self.env_abs_state_slice = env_abs_state_slice
        self.env_abs_goal_slice = env_abs_goal_slice
        self.env_to_state_func = env_to_state_func

        self.num_dqn_updates = 0

    # def abs_to_state(self, abs_state):
    #     return abs_state[5:10, :].detach().cpu().numpy()

    def abs_to_state(self, abs_state):
        return abs_state[self.env_abs_state_slice, :].detach().cpu().numpy()

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        gamma = self.gamma
        action_dim = self.env_act_dim
        env_to_state_func = self.env_to_state_func
        state = torch.as_tensor(samples["obs"], dtype=torch.float32, device=device)
        next_state = torch.as_tensor(samples["next_obs"], dtype=torch.float32, device=device)
        action = torch.as_tensor(samples["acts"].reshape(-1, 1), dtype=torch.int64, device=device)
        reward = torch.as_tensor(samples["rews"].reshape(-1, 1), dtype=torch.float32, device=device)
        done = torch.as_tensor(samples["done"].reshape(-1, 1), dtype=torch.float32, device=device)
        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        next_state_feasible_action_set = torch.as_tensor(
            np.array([padding(action_dim,
                              env_to_state_func(self.abs_to_state(next_obs_tmp)).feasible_action_set
                              ) for next_obs_tmp in samples["next_obs"]]),
            dtype=torch.int64, device=device)
        curr_q_value = self.dqn(state[:, self.env_abs_state_slice, :], state[:, self.env_abs_goal_slice, :]).gather(1,
                                                                                                                    action)

        argmax_next_state_feasible_action = next_state_feasible_action_set.gather(1,
                                                                                  self.dqn(next_state[:,
                                                                                           self.env_abs_state_slice, :],
                                                                                           next_state[:,
                                                                                           self.env_abs_goal_slice,
                                                                                           :]).detach().gather(1,
                                                                                                               next_state_feasible_action_set)
                                                                                  .argmax(dim=1, keepdim=True))
        next_q_value = self.dqn_target(next_state[:, self.env_abs_state_slice, :],
                                       next_state[:, self.env_abs_goal_slice, :]).detach().gather(  # Double DQN
            1, argmax_next_state_feasible_action).detach()
        mask = 1 - done
        target = (reward + gamma * next_q_value * mask).to(self.device)

        # calculate element-wise dqn loss
        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")

        return elementwise_loss

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.replay_memory.sample_batch()
        weights = samples["weights"].reshape(-1, 1)
        indices = samples["indices"]

        # update loss
        elementwise_loss = self._compute_dqn_loss(samples)
        loss = torch.mean(elementwise_loss * weights)
        self.optimizer.zero_grad()
        # torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 2)
        loss.backward()
        self.optimizer.step()

        # update priorities
        loss_for_prior = elementwise_loss.detach()
        new_priorities = loss_for_prior.flatten() + prior_eps
        self.replay_memory.update_priorities(indices, new_priorities)

        return loss.item()

    def learn(self, T, create_agent, init_dqn_path: str = None, ):
        # init agent and optimizer
        self.dqn = create_agent(self.env_obs_dim, self.env_act_dim, self.env_n_classes, self.device)
        self.dqn = self.dqn.to(self.device)
        self.dqn_target = copy.deepcopy(self.dqn)
        self.dqn_target.eval()
        # optimizer
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.lr)

        if init_dqn_path is not None:
            if Path(init_dqn_path).is_file():
                print(f"Load model file from {init_dqn_path}")
                self.dqn.load_state_dict(torch.load(init_dqn_path))

        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.shared_net.load_state_dict(self.dqn.state_dict())
        self.replay_memory.update_beta(0)

        while self.replay_memory.size <= self.min_replay_mem_size:
            time.sleep(1)
        loss_total = 0
        time_average = 0
        max_loss = -np.inf
        for t in range(T):
            self.replay_memory.update_beta(t)
            a = time.time()
            loss = self.update_model()
            b = time.time()
            time_average += (b - a)
            self.num_dqn_updates += 1
            if self.num_dqn_updates % self.target_update_freq == 0:
                self.dqn_target.load_state_dict(self.dqn.state_dict())
            # self.shared_state["dqn_state_dict"] = self.dqn.state_dict()
            self.shared_net.load_state_dict(self.dqn.state_dict())
            loss_total += loss
            max_loss = max(loss, max_loss)
            if self.num_dqn_updates % 1000 == 0:
                print(
                    f"Learner update numbers: {self.num_dqn_updates}, Loss average: {loss_total / 1000:4f}, max loss: {max_loss:4f},"
                    f"Time average is {time_average:2f} ms")
                loss_total = 0
                time_average = 0
                max_loss = -np.inf


class HERLearner2(Learner):
    def to_npstate(self, npstate):
        return npstate.detach().cpu().numpy()

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], toggle_debug: bool = False) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        gamma = self.gamma
        action_dim = self.env_act_dim
        env_to_state_func = self.env_to_state_func
        goal = torch.as_tensor(samples["goal"], dtype=torch.float32, device=device)
        state = torch.as_tensor(samples["obs"], dtype=torch.float32, device=device)
        next_state = torch.as_tensor(samples["next_obs"], dtype=torch.float32, device=device)
        action = torch.as_tensor(samples["acts"].reshape(-1, 1), dtype=torch.int64, device=device)
        reward = torch.as_tensor(samples["rews"].reshape(-1, 1), dtype=torch.float32, device=device)
        done = torch.as_tensor(samples["done"].reshape(-1, 1), dtype=torch.float32, device=device)
        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        next_state_feasible_action_set = torch.as_tensor(
            np.array([padding(action_dim,
                              env_to_state_func(next_obs_tmp.cpu().numpy()).feasible_action_set
                              ) for next_obs_tmp in samples["next_obs"]]),
            dtype=torch.int64, device=device)

        # curr_s_adj_mat = torch.as_tensor(
        #     np.array([env_to_state_func(self.abs_to_state(obs_tmp)).adjacency_matrix for obs_tmp in samples["obs"]]),
        #     dtype=torch.bool, device=device)

        curr_q_value = self.dqn(state, goal).gather(1, action)

        argmax_next_state_feasible_action = next_state_feasible_action_set.gather(1,
                                                                                  self.dqn(next_state,
                                                                                           goal).detach().gather(1,
                                                                                                                 next_state_feasible_action_set)
                                                                                  .argmax(dim=1, keepdim=True))
        next_q_value = self.dqn_target(next_state, goal).detach().gather(  # Double DQN
            1, argmax_next_state_feasible_action).detach()
        mask = 1 - done
        target = (reward + gamma * next_q_value * mask).to(self.device)

        # calculate element-wise dqn loss
        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")

        if toggle_debug:
            cq = curr_q_value.detach().reshape(1, -1)
            tg = target.detach().reshape(1, -1)
            # print(state, next_state)
            # print("Current Q value:", curr_q_value.detach().reshape(1, -1))
            # print("Next Q value:", target.detach().reshape(1, -1))
            print("Loss", tg - cq)
            print("Reward:", reward.reshape(1, -1))
            # print("Next feasible actions:", argmax_next_state_feasible_action.reshape(1, -1))

        return elementwise_loss

    def update_model(self, toggle_debug=False) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.replay_memory.sample_batch()
        weights = samples["weights"].reshape(-1, 1)
        indices = samples["indices"]

        # update loss
        elementwise_loss = self._compute_dqn_loss(samples, toggle_debug=toggle_debug)
        loss = torch.mean(elementwise_loss * weights)
        if toggle_debug:
            print("Replay buffer size", self.replay_memory.size)
            print("Weight is:", weights.reshape(1, -1))
            print("-" * 40)
            print("\n" * 5)
        self.optimizer.zero_grad()
        # torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 2)
        loss.backward()
        self.optimizer.step()

        # update priorities
        loss_for_prior = elementwise_loss.detach()
        new_priorities = loss_for_prior.flatten() + prior_eps
        self.replay_memory.update_priorities(indices, new_priorities)

        return loss.item()

    def learn(self, T, create_agent, init_dqn_path: str = None, lr_update_step=2000):
        # init agent and optimizer
        self.dqn = create_agent(self.env_obs_dim, self.env_act_dim, self.env_n_classes, self.device)
        self.dqn = self.dqn.to(self.device)
        self.dqn_target = copy.deepcopy(self.dqn)
        self.dqn_target.eval()
        # optimizer
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.lr, weight_decay=1e-7)

        state_level = self.shared_state['state_level']
        # output
        output_interval = 1000

        if init_dqn_path is not None:
            if Path(init_dqn_path).is_file():
                print(f"Load model file from {init_dqn_path}")
                self.dqn.load_state_dict(torch.load(init_dqn_path))

        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.shared_net.load_state_dict(self.dqn.state_dict())
        self.replay_memory.update_beta(0)

        while self.replay_memory.size <= self.min_replay_mem_size:
            time.sleep(1)
        loss_total = 0
        time_average = 0
        max_loss = -np.inf

        for t in range(T):
            self.replay_memory.update_beta(t)
            a = time.time()
            loss = self.update_model()
            b = time.time()
            time_average += (b - a)
            self.num_dqn_updates += 1
            if self.num_dqn_updates % self.target_update_freq == 0:
                self.dqn_target.load_state_dict(self.dqn.state_dict())

            if self.num_dqn_updates % lr_update_step == 0:
                if state_level != self.shared_state['state_level']:
                    state_level = self.shared_state['state_level']
                    for p in self.optimizer.param_groups:
                        p['lr'] = self.lr
                else:
                    if self.optimizer.param_groups[0]['lr']>1e-6:
                        for p in self.optimizer.param_groups:
                            p['lr'] *= 0.99

            # self.shared_state["dqn_state_dict"] = self.dqn.state_dict()
            self.shared_net.load_state_dict(self.dqn.state_dict())
            loss_total += loss
            max_loss = max(loss, max_loss)
            if self.num_dqn_updates % output_interval == 0:
                print(
                    f"Learner update numbers: {self.num_dqn_updates}, Loss average: {loss_total / output_interval:4f}, max loss: {max_loss:4f},"
                    f"Time average is {time_average:2f} ms,"
                    f"Learning rate is {self.optimizer.param_groups[0]['lr']:.3e}")
                loss_total = 0
                time_average = 0
                max_loss = -np.inf
