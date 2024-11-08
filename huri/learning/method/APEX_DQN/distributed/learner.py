""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230720osaka

"""
import time
import copy
import torch
import torch.nn.functional as F
import ray
import numpy as np
import huri.core.file_sys as fs
from huri.learning.method.AlphaZero.log import CsvWriter, write_to_csv
from huri.learning.method.AlphaZero.utils import get_time_stamp
from ray.rllib.policy.sample_batch import SampleBatch
from huri.learning.method.APEX_DQN.distributed.utils import to_onehot, swap_values, dummy_abs_state

prior_eps: float = 1e-6


def get_state_to_save(network,
                      target_network,
                      optimizer,
                      lr_scheduler,
                      train_steps):
    return {
        'network': network.state_dict(),
        'target_network': target_network.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'train_steps': train_steps,
    }


def value_rescale(value, eps=1e-3):
    return value.sign() * ((value.abs() + 1).sqrt() - 1) + eps * value


def inverse_value_rescale(value, eps=1e-3):
    temp = ((1 + 4 * eps * (value.abs() + 1 + eps)).sqrt() - 1) / (2 * eps)
    return value.sign() * (temp.square() - 1)


@ray.remote(num_gpus=0.8, num_cpus=1)
class Learner(object):
    def __init__(self,
                 env_action_space_dim: int,
                 env_classes: int,
                 net: 'DDQN',
                 icm: 'ICM',
                 rnd_model: 'RNDModel',
                 cfg: dict,
                 shared_state: ray.actor.ActorClass,
                 replay_buffer: ray.actor.ActorClass,
                 replay_buffer2: ray.actor.ActorClass,
                 log_path=None,
                 save_path=None):
        """
        [Important] Learner is assumed to run in the main thread. (Pytorch has some strange bug)

        :param args: Args have following parameters:
                    - lr: learning rate
                    - device: cpu or gpu
        :param replay_buffer: Shared replay buffer
        """
        # env action space dim
        self.env_action_dim = env_action_space_dim
        self.env_classes = env_classes
        self.shared_state = shared_state
        self.replay_buffer = replay_buffer
        self.replay_buffer2 = replay_buffer2
        # cfg
        self.lr = cfg['lr']
        self.gamma = cfg['gamma']
        self.tau = cfg['tau']
        self.target_update_freq = cfg['update_freq']
        self.batch_sz = cfg['batch_sz']
        self.save_period = cfg['save_period']
        self.device = cfg['device']
        self.n_step = cfg['n_step']
        self.save_checkpnt_path = fs.Path(save_path).joinpath('model.chkpt') if save_path is not None else None
        self.log_path = fs.Path(log_path) if log_path is not None else None
        # deep learning
        ## nn
        self.dqn = net
        self.dqn.to(self.device)
        self.dqn_target = copy.deepcopy(self.dqn)
        self.dqn.train()
        self.dqn_target.eval()
        # intrinsic curiosity module
        if icm is not None:
            self.icm = icm.to(self.device)
            self.forward_scale = cfg['forward_scale']
            self.inverse_scale = cfg['inverse_scale']
            self.intrinsic_scale = cfg['intrinsic_scale']
        if rnd_model is not None:
            self.rnd_model = rnd_model.to(self.device)
            self.optimizer = torch.optim.Adam(list(self.dqn.parameters()) + list(self.rnd_model.parameters()),
                                              lr=self.lr, weight_decay=1e-6)
        else:
            self.rnd_model = None
            self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.lr, weight_decay=1e-6)

        # optimizer
        # self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.lr, weight_decay=1e-6)

        # self.optimizer2 = torch.optim.Adam(list((self.dqn.parameters())) + list(self.icm.parameters()),
        #                                   lr=self.lr,
        #                                   weight_decay=1e-6)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
        #                                                             mode='min',
        #                                                             factor=0.1,
        #                                                             patience=1000,
        #                                                             cooldown=0,
        #                                                             verbose=True)

    def _compute_loss3(self, samples):
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
        # goal = torch.as_tensor(samples["goal"].copy(), dtype=torch.float32, device=device)
        state = torch.as_tensor(samples["state"].copy(), dtype=torch.float32, device=device)
        next_state = torch.as_tensor(samples["next_state"].copy(), dtype=torch.float32, device=device)
        action = torch.as_tensor(samples["action"].copy().reshape(-1, 1), dtype=torch.int64, device=device)
        reward = torch.as_tensor(samples["reward"].copy().reshape(-1, 1), dtype=torch.float32, device=device)
        done = torch.as_tensor(samples["done"].copy().reshape(-1, 1), dtype=torch.float32, device=device)
        next_state_feasible_action_set = torch.as_tensor(samples['next_state_feasible_action'].copy(),
                                                         dtype=torch.int64, device=device)
        # -----------------------
        state_feasible_category = [
            torch.as_tensor(samples['state_feasible_category'][:, c].copy(), dtype=torch.float32, device=device) for c
            in
            range(samples['state_feasible_category'].shape[1])]
        next_state_feasible_category = [
            torch.as_tensor(samples['next_state_feasible_category'][:, c].copy(), dtype=torch.float32, device=device)
            for c in
            range(samples['next_state_feasible_category'].shape[1])]
        # ----------------
        # if "c_abs_obs" in samples:
        #     c_abs_state = torch.as_tensor(samples["c_abs_obs"], dtype=torch.float32, device=device)
        #     c_next_abs_state = torch.as_tensor(samples["c_abs_next_obs"], dtype=torch.float32, device=device)
        #     abs_state = decompress_abs_state(c_abs_state, self.abs_state_dim[0])
        #     abs_next_state = decompress_abs_state(c_next_abs_state, self.abs_state_dim[0])
        # else:
        #     abs_state = None
        #     abs_next_state = None
        # one hot design
        # state_t = torch.cat((to_onehot(state[:, None, ...], self.env_classes),
        #                      to_onehot(goal[:, None, ...], self.env_classes)), axis=1)
        # next_state_t = torch.cat((to_onehot(next_state[:, None, ...], self.env_classes),
        #                           to_onehot(goal[:, None, ...], self.env_classes)), axis=1)
        # state_t = torch.cat((state[:, None, ...], goal[:, None, ...]), axis=1)
        # next_state_t = torch.cat((next_state[:, None, ...], goal[:, None, ...]), axis=1)
        # state_t = st1, st2, (s_mask1, s_mask2) = abs_state(state, goal, self.env_classes, state_feasible_category)
        # next_state_t = nst1, nst2, (ns_mask1, ns_mask2) = abs_state(next_state, goal, self.env_classes,
        #                                                             next_state_feasible_category)
        # state_feasible_category = []
        # next_state_feasible_category =[]
        state_t = dummy_abs_state(state, state_feasible_category,
                                  device=device)  # = st1, st2, (s_mask1, s_mask2)
        next_state_t = dummy_abs_state(next_state, next_state_feasible_category,
                                       device=device)  # = nst1, nst2, (ns_mask1, ns_mask2)

        # for Curiosity-driven(Random Network Distillation)
        # update RND model
        if self.rnd_model is not None:
            predict_next_state_feature, target_next_state_feature = self.rnd_model(next_state_t)
            rnd_loss = F.mse_loss(predict_next_state_feature, target_next_state_feature.detach(),
                                  reduction='none').mean(-1).sum()
        else:
            rnd_loss = 0
        # Q net, Q' target net, s current state, s' next state
        # double DQN Q(s, a) = r + y * Q'(s', argmax_a Q(s', a))
        curr_q_table = self.dqn(state_t)
        next_q_table = self.dqn(next_state_t)
        curr_q_value = curr_q_table.gather(1, action)

        argmax_next_state_feasible_action = next_state_feasible_action_set.gather(1, next_q_table.detach()
                                                                                  .gather(1,
                                                                                          next_state_feasible_action_set)
                                                                                  .argmax(dim=1, keepdim=True))
        next_q_value = self.dqn_target(next_state_t).detach().gather(  # Double DQN
            1, argmax_next_state_feasible_action).detach()
        mask = 1 - done
        # target = value_rescale((reward + gamma * inverse_value_rescale(next_q_value) * mask).to(self.device))
        target = value_rescale((reward + (gamma ** self.n_step) * inverse_value_rescale(next_q_value) * mask)).to(
            self.device)
        # calculate element-wise dqn loss
        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")

        # multiclass loss
        if self.env_classes > 1:
            # print("multiclass loss")
            # m = np.concatenate((samples["goal"].copy(),
            #                     samples["state"].copy(),
            #                     samples["next_state"].copy()))
            # m_swap_torch = torch.as_tensor(swap_values(m, self.env_classes),
            #                                dtype=torch.float32,
            #                                device=device)
            # goal_c, state_c, next_state_c = m_swap_torch[0: len(samples["goal"])], \
            #     m_swap_torch[len(samples["goal"]): 2 * len(samples["goal"])], \
            #     m_swap_torch[2 * len(samples["goal"]):]
            # state_t_c = torch.cat((to_onehot(state_c[:, None, ...], self.env_classes),
            #                        to_onehot(goal_c[:, None, ...], self.env_classes)), axis=1)
            # next_state_t_c = torch.cat((to_onehot(next_state_c[:, None, ...], self.env_classes),
            #                             to_onehot(goal_c[:, None, ...], self.env_classes)), axis=1)
            # # state_t_c = torch.cat((state_c[:, None, ...], goal_c[:, None, ...]), axis=1)
            # # next_state_t_c = torch.cat((next_state_c[:, None, ...], goal_c[:, None, ...]), axis=1)
            # curr_q_table_c = self.dqn(state_t_c).detach()
            # next_q_table_c = self.dqn(next_state_t_c).detach()
            # curr_q_table_c = self.dqn((st2, st1, (s_mask2, s_mask1,))).detach()
            # next_q_table_c = self.dqn((nst2, nst1, (ns_mask1, ns_mask2,))).detach()
            # curr_q_value_c = self.dqn_target(state_t_c).detach().gather(1, action)
            # next_q_table_c = self.dqn_target(next_state_t_c).detach().gather(1, action)
            # elementwise_loss_c = F.mse_loss(curr_q_table_c, curr_q_table, reduction="none").mean(1,
            #                                                                                      keepdim=True) + \
            #                      F.mse_loss(next_q_table_c, next_q_table, reduction="none").mean(1,
            #                                                                                      keepdim=True)
            elementwise_loss_c = 0
        else:
            elementwise_loss_c = 0
        return elementwise_loss, rnd_loss, elementwise_loss_c * .5

    def update_model(self):
        """Update the model by gradient descent."""
        if self.replay_buffer2 is not None:
            samples: SampleBatch = ray.get(self.replay_buffer.sample.remote(int(self.batch_sz / 2), .6))
            samples2: SampleBatch = ray.get(self.replay_buffer2.sample.remote(int(self.batch_sz / 2), .6))
            samples = samples.concat(samples2)
        else:
            samples: SampleBatch = ray.get(self.replay_buffer.sample.remote(self.batch_sz, .6))
        weights = torch.tensor(samples['weights'].reshape(-1, 1).copy(), dtype=torch.float32, device=self.device)
        indices = samples['batch_indexes']
        # update loss
        self.optimizer.zero_grad()
        # elementwise_loss = self._compute_loss2(samples)
        # loss = torch.mean(elementwise_loss * weights) / 2
        elementwise_loss, rnd_loss, elementwise_loss_c = self._compute_loss(samples)
        # print(elementwise_loss_c.shape, weights.shape)
        if self.rnd_model is not None:
            loss = torch.mean((elementwise_loss + elementwise_loss_c) * weights) + rnd_loss
        else:
            # loss = torch.mean((elementwise_loss + elementwise_loss_c) * weights)
            loss = torch.mean(elementwise_loss * weights)
        # torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 2)
        loss.backward()
        self.optimizer.step()
        # update priorities
        loss_for_prior = elementwise_loss.detach()
        new_priorities = loss_for_prior.flatten() + prior_eps
        if self.replay_buffer2 is not None:
            new_priorities_np = new_priorities.cpu().numpy()
            self.replay_buffer.update_priorities.remote(indices[:int(self.batch_sz / 2)],
                                                        new_priorities_np[:int(self.batch_sz / 2)])
            self.replay_buffer2.update_priorities.remote(indices[int(self.batch_sz / 2):],
                                                         new_priorities_np[int(self.batch_sz / 2):])
        else:
            self.replay_buffer.update_priorities.remote(indices, new_priorities.cpu().numpy())
        if self.rnd_model is not None:
            return loss.item(), rnd_loss.item()
        else:
            return loss.item(), 0
        # return loss.item(), 0

    def sync_dqn_train_steps(self, epoch):
        state_dict = {k: v.cpu() for k, v in self.dqn.state_dict().items()}
        self.shared_state.set_info.remote('weights', state_dict)
        if self.rnd_model is not None:
            state_dict_rnd = {k: v.cpu() for k, v in self.rnd_model.state_dict().items()}
            self.shared_state.set_info.remote('rnd_model_weights', state_dict_rnd)
        self.shared_state.set_info.remote('train_steps', epoch)

    def start(self, num_epoch=1000000000, min_replay_sz=1000, output_interval=1000):
        # wait for the replay buffer have enough samples
        while ray.get(self.replay_buffer.__len__.remote()) <= min_replay_sz \
                or (self.replay_buffer2 is not None and ray.get(self.replay_buffer2.__len__.remote()) <= min_replay_sz):
            time.sleep(1)
        if self.log_path is not None and isinstance(self.log_path, fs.Path):
            writer = CsvWriter(str(self.log_path.joinpath(f'train_log.csv')))
        # start training
        loss_total = 0
        rnd_loss_total = 0
        loss_interval_counter = 0
        for epoch in range(1, num_epoch + 1):
            st = time.time()
            loss, rnd_loss = self.update_model()
            # auto_garbage_collect()
            c_time = time.time() - st
            loss_total += loss
            rnd_loss_total += rnd_loss
            loss_interval_counter += 1
            # update target network
            if epoch % self.target_update_freq == 0:
                # self.dqn_target.load_state_dict(self.dqn.state_dict())
                for param, target_param in zip(self.dqn.parameters(), self.dqn_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                    # TODO update learning rate
            # if epoch > 3000:  # warmup
            #     self.scheduler.step(loss)
            if epoch % self.save_period == 0:
                # Update shared network
                self.sync_dqn_train_steps(epoch)
                # Save checkpoint
                if self.save_checkpnt_path is not None:
                    torch.save(get_state_to_save(
                        network=self.dqn,
                        target_network=self.dqn_target,
                        optimizer=self.optimizer,
                        lr_scheduler=self.scheduler,
                        train_steps=epoch,
                    ), self.save_checkpnt_path)

            if epoch % output_interval == 0:
                # logging
                if self.log_path is not None and isinstance(self.log_path, fs.Path):
                    write_to_csv(writer, [
                        ('timestamp', get_time_stamp(), '%1s'),
                        ('time_consumption', c_time, '%1s'),
                        ('learner_train_loss', loss, '%1f'),
                        ('learner_learning_rate', self.optimizer.param_groups[0]['lr'], '%1f'),
                        ('epoch', epoch, '%1d'),
                        ('loss', (loss_total - rnd_loss_total) / loss_interval_counter, '%1f'),
                        ('rnd_loss', rnd_loss_total / loss_interval_counter, '%1f'),
                    ])
                loss_total = 0
                rnd_loss_total = 0
                loss_interval_counter = 0

    def _compute_loss2(self, samples):
        device = self.device  # for shortening the following lines
        gamma = self.gamma
        action_dim = self.env_action_dim
        goal = torch.as_tensor(samples["goal"].copy(), dtype=torch.float32, device=device)
        state = torch.as_tensor(samples["state"].copy(), dtype=torch.float32, device=device)
        next_state = torch.as_tensor(samples["next_state"].copy(), dtype=torch.float32, device=device)
        action = torch.as_tensor(samples["action"].copy().reshape(-1, 1), dtype=torch.int64, device=device)
        reward = torch.as_tensor(samples["reward"].copy().reshape(-1, 1), dtype=torch.float32, device=device)
        done = torch.as_tensor(samples["done"].copy().reshape(-1, 1), dtype=torch.float32, device=device)
        next_state_feasible_action_set = torch.as_tensor(samples['next_state_feasible_action'].copy(),
                                                         dtype=torch.int64, device=device)
        # if "c_abs_obs" in samples:
        #     c_abs_state = torch.as_tensor(samples["c_abs_obs"], dtype=torch.float32, device=device)
        #     c_next_abs_state = torch.as_tensor(samples["c_abs_next_obs"], dtype=torch.float32, device=device)
        #     abs_state = decompress_abs_state(c_abs_state, self.abs_state_dim[0])
        #     abs_next_state = decompress_abs_state(c_next_abs_state, self.abs_state_dim[0])
        # else:
        #     abs_state = None
        #     abs_next_state = None

        state_t = torch.cat((state[:, None, ...], goal[:, None, ...]), axis=1)
        next_state_t = torch.cat((next_state[:, None, ...], goal[:, None, ...]), axis=1)

        # intrinsic reward
        action_onehot = F.one_hot(action, num_classes=action_dim).view(len(action), -1)
        pred_s_next, pred_a_vec, feature_x_next = self.icm.get_full(state_t, next_state_t, action_onehot)
        # calculate forward prediction and inverse prediction loss
        forward_loss = F.mse_loss(pred_s_next, feature_x_next.detach(), reduction='none')
        inverse_pred_loss = F.cross_entropy(pred_a_vec,
                                            torch.as_tensor(action_onehot, dtype=torch.float32, device=device).detach(),
                                            reduction='none')
        # calculate rewards
        intrinsic_rewards = self.intrinsic_scale * forward_loss.mean(-1).unsqueeze(1)
        reward += intrinsic_rewards
        # Q net, Q' target net, s current state, s' next state
        # double DQN Q(s, a) = r + y * Q'(s', argmax_a Q(s', a))

        curr_q_table = self.dqn(state_t)
        curr_q_value = curr_q_table.gather(1, action)

        argmax_next_state_feasible_action = next_state_feasible_action_set.gather(1, self.dqn(next_state_t).detach()
                                                                                  .gather(1,
                                                                                          next_state_feasible_action_set)
                                                                                  .argmax(dim=1, keepdim=True))
        next_q_value = self.dqn_target(next_state_t).detach().gather(  # Double DQN
            1, argmax_next_state_feasible_action).detach()
        mask = 1 - done
        target = value_rescale((reward + gamma * inverse_value_rescale(next_q_value) * mask).to(self.device))
        # calculate element-wise dqn loss
        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")
        return elementwise_loss, self.forward_scale * forward_loss, self.inverse_scale * inverse_pred_loss, intrinsic_rewards

    def update_model2(self):
        """Update the model by gradient descent."""
        samples: SampleBatch = ray.get(self.replay_buffer.sample.remote(self.batch_sz, .6))
        weights = torch.tensor(samples['weights'].reshape(-1, 1).copy(), dtype=torch.float32, device=self.device)
        indices = samples['batch_indexes']
        # update loss
        self.optimizer.zero_grad()
        # elementwise_loss = self._compute_loss2(samples)
        # loss = torch.mean(elementwise_loss * weights) / 2
        elementwise_loss, forward_loss, inverse_pred_loss, intrinsic_rewards = self._compute_loss2(samples)
        loss = torch.mean(elementwise_loss * weights) + torch.mean(forward_loss) + torch.mean(inverse_pred_loss)
        # torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 2)
        loss.backward()
        self.optimizer.step()

        # update priorities
        loss_for_prior = elementwise_loss.detach()
        new_priorities = loss_for_prior.flatten() + prior_eps
        self.replay_buffer.update_priorities.remote(indices, new_priorities.cpu().numpy())
        return loss.item(), torch.mean(forward_loss).item(), torch.mean(
            inverse_pred_loss).item(), torch.mean(intrinsic_rewards).item()

    # def start2(self, num_epoch=1000000000, min_replay_sz=1000, output_interval=1000):
    #     # wait for the replay buffer have enough samples
    #     while ray.get(self.replay_buffer.__len__.remote()) <= min_replay_sz:
    #         time.sleep(1)
    #     if self.log_path is not None and isinstance(self.log_path, fs.Path):
    #         writer = CsvWriter(str(self.log_path.joinpath(f'train_log.csv')))
    #     # start training
    #     loss_total = 0
    #     forward_loss_total = 0
    #     inverse_loss_total = 0
    #     intrinsic_reward_total = 0
    #     loss_interval_counter = 0
    #     for epoch in range(1, num_epoch + 1):
    #         st = time.time()
    #         loss, forward_loss, inverse_pred_loss, intrinsic_reward = self.update_model()
    #         c_time = time.time() - st
    #         loss_total += loss
    #         forward_loss_total += forward_loss
    #         inverse_loss_total += inverse_pred_loss
    #         intrinsic_reward_total += intrinsic_reward
    #         loss_interval_counter += 1
    #         # update target network
    #         if epoch % self.target_update_freq == 0:
    #             # self.dqn_target.load_state_dict(self.dqn.state_dict())
    #             for param, target_param in zip(self.dqn.parameters(), self.dqn_target.parameters()):
    #                 target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    #                 # TODO update learning rate
    #         # if epoch > 3000:  # warmup
    #         #     self.scheduler.step(loss)
    #         if epoch % self.save_period == 0:
    #             # Update shared network
    #             self.sync_dqn_train_steps(epoch)
    #             # Save checkpoint
    #             if self.save_checkpnt_path is not None:
    #                 torch.save(get_state_to_save(
    #                     network=self.dqn,
    #                     target_network=self.dqn_target,
    #                     optimizer=self.optimizer,
    #                     lr_scheduler=self.scheduler,
    #                     train_steps=epoch,
    #                 ), self.save_checkpnt_path)
    #             # logging
    #         if epoch % output_interval == 0:
    #             if self.log_path is not None and isinstance(self.log_path, fs.Path):
    #                 write_to_csv(writer, [
    #                     ('timestamp', get_time_stamp(), '%1s'),
    #                     ('time_consumption', c_time, '%1s'),
    #                     ('learner_train_loss', loss, '%1f'),
    #                     ('learner_learning_rate', self.optimizer.param_groups[0]['lr'], '%1f'),
    #                     ('epoch', epoch, '%1d'),
    #                     ('loss', loss_total / loss_interval_counter, '%1f'),
    #                     ('forward_loss', forward_loss_total / loss_interval_counter, '%1f'),
    #                     ('inverse_pred_loss', inverse_loss_total / loss_interval_counter, '%1f'),
    #                     ('intrinsic_reward', intrinsic_reward_total / loss_interval_counter, '%1f'),
    #                 ])
    #             loss_total = 0
    #             forward_loss_total = 0
    #             inverse_loss_total = 0
    #             loss_interval_counter = 0
    #             intrinsic_reward_total = 0


if __name__ == '__main__':
    pass
