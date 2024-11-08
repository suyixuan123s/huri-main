""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230718osaka

"""
import ray
import itertools
from collections import namedtuple
import numpy as np
import torch
import cv2
import time
import copy
from pathlib import Path
from typing import NamedTuple
from huri.learning.env.rack_v3.env import RackArrangementEnv, RackStatePlot, RackState
from huri.learning.method.AlphaZero.log import CsvWriter, write_to_csv
from huri.components.utils.img_utils import combine_images
from huri.learning.method.AlphaZero.utils import get_time_stamp
import huri.core.file_sys as fs
from huri.learning.method.APEX_DQN.distributed_fixed_goal.reanalyzer import Trajectory
from huri.learning.method.APEX_DQN.distributed_fixed_goal.utils import to_onehot_np

MAX_NUM_PENDING_TASKS = 100
MAX_SEND_PERIOD = 0
MAX_REPLAY_BUFFER_LEN = 6000


# def inverse_value_rescale(value, eps=1e-3):
#     temp = ((1 + 4 * eps * (value.abs() + 1 + eps)).sqrt() - 1) / (2 * eps)
#     return value.sign() * (temp.square() - 1)


# CheckPoint = namedtuple('CheckPoint', ['weights', 'training_level', 'train_steps'])
@ray.remote
class SharedState:
    """
    Class which run in a dedicated thread to store the network weights and some information.
    """

    def __init__(self, checkpoint):
        self.current_checkpoint = copy.deepcopy(checkpoint)

    # def save_checkpoint(self, path=None):
    #     if not path:
    #         path = self.config.results_path / "model.checkpoint"
    #
    #     torch.save(self.current_checkpoint, path)

    def get_checkpoint(self):
        return copy.deepcopy(self.current_checkpoint)

    def get_info(self, keys):
        if isinstance(keys, str):
            return self.current_checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.current_checkpoint[key] for key in keys}
        else:
            raise TypeError

    def set_info(self, keys, values=None):
        if isinstance(keys, str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_checkpoint.update(keys)
        else:
            raise TypeError

    def extend_info(self, keys, values):
        if isinstance(keys, str) and isinstance(values, list):
            self.current_checkpoint[keys].extend(values)
            if len(self.current_checkpoint[keys]) > MAX_REPLAY_BUFFER_LEN:
                # self.current_checkpoint[keys] = self.current_checkpoint[keys][-MAX_REPLAY_BUFFER_LEN:]
                del self.current_checkpoint[keys][:MAX_REPLAY_BUFFER_LEN // 5]
        else:
            raise TypeError

    def get_info_pop(self, key, n=1):
        assert isinstance(key, str) and isinstance(self.current_checkpoint[key], list)
        if len(self.current_checkpoint[key]) > 0:
            n = min(len(self.current_checkpoint[key]), n)
            popped_elements = self.current_checkpoint[key][-n:]
            del self.current_checkpoint[key][-n:]
            return popped_elements
        else:
            return None

    def get_info_len(self, key):
        assert isinstance(key, str) and isinstance(self.current_checkpoint[key], list)
        return len(self.current_checkpoint[key])

    def set_info_clear(self, key):
        assert isinstance(key, str) and isinstance(self.current_checkpoint[key], list)
        self.current_checkpoint[key].clear()


def dqn_select_action(env: RackArrangementEnv,
                      feasible_action_set: np.ndarray,
                      state: RackState,
                      dqn: torch.nn.Module,
                      device: str,
                      toggle_no_repeat: bool = True,
                      toggle_return_action_value: bool = False) -> int or (int, np.ndarray):
    if toggle_no_repeat:
        feasible_action_set_tmp = feasible_action_set
        repeat_acts = []
        for _ in env.rack_state_history:
            act = env.action_between_states_constraint_free(state, _)
            if act is not None:
                repeat_acts.append(act)
        feasible_action_set = np.setdiff1d(feasible_action_set, repeat_acts)
        if len(feasible_action_set) == 0:
            feasible_action_set = feasible_action_set_tmp
            # self.early_stop = True
            # # print("Potential ERROR Happens")
            # print("state is", self.env.state)
            # print("goal pattern is", self.env.goal_pattern)
    if len(feasible_action_set) < 1:
        return None, None
    with torch.no_grad():
        feasible_action_set = torch.as_tensor(feasible_action_set, dtype=torch.int64, device=device)
        # one hot design
        # dqn_action_value = dqn(
        #     torch.as_tensor(
        #         np.concatenate((state.abs_state(env.num_classes), env.goal_pattern.abs_state(env.num_classes)), axis=0),
        #         dtype=torch.float32,
        #         device=device).unsqueeze(0), ).detach()
        # dqn_action_value = dqn(
        #     torch.as_tensor(np.stack((state.state, env.goal_pattern.state), axis=0), dtype=torch.float32,
        #                     device=device).unsqueeze(0), ).detach()
        # dqn_action_value = dqn(
        #     torch.as_tensor(state.state, dtype=torch.float32,
        #                     device=device).unsqueeze(0).unsqueeze(0), ).detach()
        dqn_action_value = dqn(
            torch.as_tensor(to_onehot_np(state.state, env.num_classes), dtype=torch.float32,
                            device=device).unsqueeze(0), ).detach()

        # input_value = np.array(abs_state_np(state.state, env.goal_pattern.state, env.num_classes))[None, ...]
        # dqn_action_value = dqn(
        #     dummy_abs_state(input_value, [torch.as_tensor(c, dtype=torch.float32, device=device).unsqueeze(0) for c in
        #                                   category_feasible_action(state.state, env.action_space_dim, env.num_classes)],
        #                     device
        #                     )).detach()
        selected_action = feasible_action_set[dqn_action_value.squeeze()[feasible_action_set].argmax()].item()
    if not toggle_return_action_value:
        return selected_action
    else:
        # dqn_action_value = inverse_value_rescale(dqn_action_value)
        return selected_action, dqn_action_value.cpu().numpy().squeeze()


class EpsilonScheduler:
    def __init__(self, initial_epsilon, final_epsilon, decay_rate):
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.decay_rate = decay_rate
        self.current_step = 0

    def get_epsilon(self):
        decay_factor = max(0, 1 - self.decay_rate * self.current_step)
        epsilon = self.initial_epsilon * decay_factor

        return max(epsilon, self.final_epsilon)

    def step(self):
        self.current_step += 1
        return self.get_epsilon()

    def reset(self):
        self.current_step = 0

    def half_reset(self):
        self.initial_epsilon = .5
        self.current_step = 0


def debug_func(traj_list, replay_buffer):
    for i in traj_list:
        replay_buffer.add.remote(i[:])


@ray.remote(num_cpus=1)
class Actor(object):
    """
    Actor object to start simulation and collect data
    """

    def __init__(self,
                 actor_id: int,
                 env: RackArrangementEnv,
                 net: 'DDQN',
                 cfg: dict,
                 H_buffer: ray.actor.ActorClass,
                 replay_buffer: ray.actor.ActorClass,
                 shared_state: ray.actor.ActorClass,
                 shared_traj_buffer: ray.actor.ActorClass,
                 log_path=None,
                 toggle_visual=False,
                 toggle_H_buffer=False,
                 ):
        assert isinstance(env, RackArrangementEnv), f'Only accept the env as RackArrangementEnv. ' \
                                                    f'Instead of {type(env)}'

        self._actor_id = actor_id
        self.env = env
        self.training_level = -1
        self.scheduler = self.env.scheduler
        self.device = cfg['actor_device']
        # network
        self.dqn = net.to(self.device)
        self.dqn.eval()
        # set up replay buffer size
        self.H_buffer = H_buffer
        self.p_init = cfg['p_init']
        self.H_batch_sz = cfg['H_batch_sz']
        if H_buffer is None:
            self.toggle_H_buffer = False
        else:
            self.toggle_H_buffer = toggle_H_buffer

        self.replay_buffer = replay_buffer
        self.shared_state = shared_state
        self.shared_traj_buffer = shared_traj_buffer
        # set up scheduler
        self.epsilon_decay = cfg['eps_decay']
        self.max_epsilon = cfg['eps_max']
        self.min_epsilon = cfg['eps_min']
        self.epsilon_scheduler = EpsilonScheduler(self.max_epsilon, self.min_epsilon, self.epsilon_decay)

        self.store_replay_buffer = cfg['store_replay_buffer']
        # hyper-parameter
        self.reset_num = cfg['reset_num']
        self.reset_num2 = cfg['reset_num2']
        self.reset_evolve = cfg['reset_evolve']
        self.target_update_freq = cfg['update_freq']
        self.send_period = cfg['send_period']  # periodically send data to replay
        # is-debug
        self.toggle_visual = toggle_visual
        # log_path
        self.log_path = fs.Path(log_path) if log_path is not None else None
        # backpressure
        self.result_refs = []
        # store to analyzer
        self.store_reanalyzer = cfg['store_reanalyzer']
        # n-step return
        self.n_step = cfg['n_step']
        self.gamma = cfg['gamma']
        #
        # init replay sz
        self.init_replay_num = cfg['init_replay_num']
        self.average_len = 999

    @property
    def epsilon(self):
        return self.epsilon_scheduler.get_epsilon()

    def sync_dqn_checkpoint(self):
        weights = ray.get(self.shared_state.get_info.remote('weights'))
        self.dqn.load_state_dict(weights)

    def sync_training_checkpoint(self):
        ''' Sync training level + Reset epsilon if training level update'''
        training_level: int = ray.get(self.shared_state.get_info.remote('training_level'))
        if self.toggle_H_buffer:
            self.H_buffer.sync_state.remote(training_level)
        if self.training_level != training_level:
            self.training_level = training_level
            self.env.scheduler.set_training_level(self.training_level)
            self.sync_dqn_checkpoint()
            # self.epsilon_scheduler.reset()
            self.epsilon_scheduler.half_reset()
            self.average_len = 999
            RackState._cache.clear()
            self.env.reset()
            del self.failed_explore_states[:]

    def select_action(self,
                      state: RackState,
                      env: RackArrangementEnv,
                      toggle_no_repeat=True) -> int:
        if self.epsilon > np.random.random():
            selected_action = env.sample(state, toggle_no_repeat)
        else:
            selected_action = dqn_select_action(env=env,
                                                state=state,
                                                dqn=self.dqn,
                                                device=self.device,
                                                feasible_action_set=state.feasible_action_set,
                                                toggle_no_repeat=toggle_no_repeat, )
        return selected_action

    def store_traj_list(self, trajectory_list, store_reanalyzer=False):
        if store_reanalyzer:
            ray.get(self.shared_traj_buffer.extend_info.remote('trajectory_list', trajectory_list))
        samples = trajectory_list[0][:]
        for traj in trajectory_list[1:]:
            samples = samples.concat(traj[:])  # concat trajectory
        self.replay_buffer.add.remote(samples)
        del trajectory_list[:]

    def start(self, output_interval=100):
        """Actor starts"""
        env = self.env
        scheduler = self.scheduler
        target_update_freq = self.target_update_freq
        toggle_visual = self.toggle_visual
        reset_evolve = self.reset_evolve
        trajectory_list = []
        action_dim = self.env.action_space_dim
        if self.log_path is not None and isinstance(self.log_path, Path):
            writer = CsvWriter(str(self.log_path.joinpath(f'actor_{self._actor_id}_log.csv')))
        # ====
        # dqn load shared net weight
        self.sync_dqn_checkpoint()
        # start actors
        step = 0
        H_buffer_tmp = []
        possible_state_goal_pair = []
        retry_state_goal = []
        toggle_need_retry = False
        episode = 0
        self.failed_explore_states = []
        while 1:
            reset_num = self.reset_num if env.scheduler.state_level <= reset_evolve else self.reset_num2
            # rest counter, score
            episode += 1
            reset_cnt = 0
            score = 0
            H_buffer_tmp_tmp = []
            # curriculum learning
            if step % 20 == 0:  # reduce the number of times to access the variables
                self.sync_training_checkpoint()
                if len(RackState._cache) > 10000:
                    RackState._cache.clear()
            # init epoch
            ## state reset
            # if self.epsilon <= self.min_epsilon and np.random.random() > .5 and len(self.failed_explore_states) > 0:
            #     state = self.failed_explore_states.pop(-1)
            #     state = env.reset_state(state)
            # else:
            state = env.reset()
            H_buffer_tmp_tmp.append([state, env.goal_pattern])
            ## create a class to hold trajectory
            traj = Trajectory(goal=env.goal_pattern,
                              action_dim=action_dim,
                              n_step=self.n_step,
                              gamma=self.gamma,
                              num_categories=env.num_classes, )
            start_ep_t = time.time()
            is_success_traj = True
            if toggle_visual:
                rsp = RackStatePlot(env.goal_pattern, )
                plot = rsp.plot_states([state]).get_img()
                img_list = [plot]
            for e_step in itertools.count(1):
                # select action
                action = self.select_action(state, env)
                if action is None:
                    is_success_traj = False
                    break
                # step
                next_state, reward, done, _ = self.env.step(action)  # next_state reward done
                # store trajectory
                traj.add_transition(state, action, reward, next_state, done)
                # reset state
                state = next_state
                # update score, coutner
                score += reward  # reward
                reset_cnt += 1
                step += 1
                # store to H buffer
                if self.toggle_H_buffer:
                    H_buffer_tmp_tmp.append([state, env.goal_pattern])
                # plot
                if toggle_visual:
                    img_list.append(rsp.plot_states([state]).get_img())
                    plot = combine_images(img_list, columns=20)
                    cv2.imshow(f"plot_{self._actor_id}", plot)
                    cv2.waitKey(100)
                # update
                if step % target_update_freq == 0:
                    self.sync_dqn_checkpoint()
                # if episode ends
                if done:  # done
                    # TODO addd
                    trajectory_list.append(traj)
                    break
                if reset_cnt % reset_num == 0:
                    # TODO
                    trajectory_list.append(traj)
                    break
            end_ep_t = time.time()
            # linearly decrease epsilon
            self.epsilon_scheduler.step()
            if len(trajectory_list) >= self.send_period:
                # self.store_traj_list(trajectory_list,
                #                      self.store_reanalyzer if self.epsilon > np.random.random() else False)
                self.store_traj_list(trajectory_list, self.store_reanalyzer)
            if not is_success_traj:
                continue

                # pass
            if step % output_interval == 0 and self.log_path is not None and isinstance(self.log_path, Path):
                write_to_csv(writer, [
                    ('timestamp', get_time_stamp(), '%1s'),
                    ('step', step, '%1d'),
                    ('time_consumption', end_ep_t - start_ep_t, '%1s'),
                    ('episode_lens', len(traj) + 1, '%1d'),
                    ('acc_rewards', score, '%1f'),
                    ('state_level', scheduler.state_level, '%1d'),
                    ('class_level', scheduler.class_level, '%1d'),
                    ('epsilon', self.epsilon, '%1f'),
                    ('episode', episode, '%1f'),
                ])

    def warmup(self, ):
        """Actor starts"""
        env = self.env
        env.is_curriculum_lr = False
        reset_num = self.reset_num
        toggle_visual = self.toggle_visual
        trajectory_list = []
        action_dim = self.env.action_space_dim
        # start actors
        step = 0
        episode = 0
        buff_len = ray.get(self.replay_buffer.__len__.remote())
        while buff_len < self.init_replay_num:
            # rest counter, score
            episode += 1
            reset_cnt = 0
            score = 0
            # init epoch
            ## state reset
            state = env.reset()
            ## create a class to hold trajectory
            traj = Trajectory(goal=env.goal_pattern,
                              action_dim=action_dim,
                              n_step=self.n_step,
                              gamma=self.gamma,
                              num_categories=env.num_classes, )
            start_ep_t = time.time()
            is_success_traj = True
            if toggle_visual:
                rsp = RackStatePlot(env.goal_pattern, )
                plot = rsp.plot_states([state]).get_img()
                img_list = [plot]
            for e_step in itertools.count(1):
                # select action
                action = self.select_action(state, env)
                if action is None:
                    is_success_traj = False
                    break
                # step
                next_state, reward, done, _ = self.env.step(action)  # next_state reward done
                # store trajectory
                traj.add_transition(state, action, reward, next_state, done)
                # reset state
                state = next_state
                # update score, coutner
                score += reward  # reward
                reset_cnt += 1
                step += 1
                # store to H buffer
                # plot
                if toggle_visual:
                    img_list.append(rsp.plot_states([state]).get_img())
                    plot = combine_images(img_list, columns=20)
                    cv2.imshow(f"plot_{self._actor_id}", plot)
                    cv2.waitKey(100)
                # update
                if done:  # done
                    # TODO addd
                    trajectory_list.append(traj)
                    break
                if reset_cnt >= reset_num:
                    # TODO
                    trajectory_list.append(traj)
                    break
            end_ep_t = time.time()
            if not is_success_traj:
                continue
            self.store_traj_list(trajectory_list, self.store_reanalyzer)
            buff_len = ray.get(self.replay_buffer.__len__.remote())
        print(f"Replay buffer size: {buff_len}")
        env.is_curriculum_lr = True
        return True


@ray.remote
class Eval(object):
    def __init__(self,
                 env: RackArrangementEnv,
                 net: 'DDQN',
                 cfg: dict,
                 H_buffer: ray.actor.ActorClass,
                 shared_state: ray.actor.ActorClass,
                 shared_traj_buffer: ray.actor.ActorClass,
                 save_path: str = None,
                 log_path: str = None,
                 toggle_visual=False,
                 ):
        assert isinstance(env, RackArrangementEnv), f'Only accept the env as RackArrangementEnv. ' \
                                                    f'Instead of {type(env)}'

        self.env = env
        self.training_level = self.env.scheduler.state_level
        self.log_path = Path(log_path) if log_path is not None else None
        self.save_path = Path(save_path)
        self.save_checkpnt_path = Path(save_path) if save_path is not None else None
        self.device = cfg['device']
        # network
        self.dqn = net.to(self.device)
        self.dqn.eval()
        # set up replay buffer size
        self.H_buffer = H_buffer
        self.shared_state = shared_state
        self.shared_traj_buffer = shared_traj_buffer
        # hyper-parameter
        self.eval_num = cfg['eval_num']
        self.reset_multiplier = cfg['reset_multiplier']
        self.reset_evolve = cfg['reset_evolve']
        self.reset_multiplier2 = cfg['reset_multiplier2']
        self.eval_interval = cfg['eval_interval']
        self.pass_rate = cfg['pass_rate']
        # is-debug
        self.toggle_visual = toggle_visual

    def sync_dqn_checkpoint(self):
        weights = ray.get(self.shared_state.get_info.remote('weights'))
        # breakpoint()
        self.dqn.load_state_dict(weights)

    def sync_training_checkpoint(self):
        ''' Sync training level + Reset epsilon if training level update'''
        training_level: int = ray.get(self.shared_state.get_info.remote('training_level'))
        # if self.training_level != training_level:
        self.training_level = training_level
        self.env.scheduler.set_training_level(self.training_level)

    def set_average_len(self, len):
        self.shared_state.set_info.remote('eval_average_len', len)

    def start(self):
        # init
        if self.log_path is not None and isinstance(self.log_path, Path):
            writer = CsvWriter(str(self.log_path.joinpath(f'eval_log.csv')))
        env = self.env
        reset_multiplier = self.reset_multiplier
        best_eval_score = 0
        toggle_visual = self.toggle_visual
        train_steps = 0
        pass_rate = self.pass_rate
        while 1:
            time.sleep(self.eval_interval)
            train_steps_tmp = ray.get(self.shared_state.get_info.remote('train_steps'))
            if train_steps_tmp == train_steps:
                time.sleep(.5)
                continue
            train_steps = train_steps_tmp
            self.sync_dqn_checkpoint()
            self.sync_training_checkpoint()
            is_update_training_level = False
            # return/number of dones/total len
            total_return = 0
            num_dones = 0
            total_len = 0
            # a star
            a_star_total_len = 0
            a_star_num_dones = 0

            # env
            candidate_state_goal = []
            # candidate_state_goal += failed_data_list_tmp
            # failed data list
            failed_data_list_tmp = []
            failed_data_path = []
            for i in range(self.eval_num):
                s = env.reset()
                g = env.goal_pattern
                candidate_state_goal.append([s, g])
            feasible_candidate_len = len(candidate_state_goal)

            # if hasattr(self.env.scheduler, 'evolve_level'):
            #     reset_num_multiply = self.env.scheduler.evolve_level
            # else:
            reset_num_multiply = self.env.scheduler.state_level
            if env.scheduler.state_level >= self.reset_evolve:
                reset_multiplier = self.reset_multiplier2

            for i, (i_s, i_g) in enumerate(candidate_state_goal):
                t_score = 0
                t_state = env.reset_state_goal(i_s, i_g)
                traj_list = [[t_state, i_g]]
                traj_tmp = [t_state]
                # print(env.scheduler.state_level, env.is_curriculum_lr)
                if toggle_visual:
                    rsp = RackStatePlot(env.goal_pattern, )
                    plot = rsp.plot_states([t_state]).get_img()
                    img_list = [plot]
                # a start agent
                # is_done, path = TubePuzzle(t_state.state, goal_pattern).atarSearch(max_iter_cnt=50)
                # if is_done:
                #     a_star_total_len += (len(path) - 1)
                #     a_star_num_dones += 1
                # dqn agent
                is_state_goal_feasible = True
                for t in itertools.count(1):
                    t_action = dqn_select_action(env=env,
                                                 feasible_action_set=t_state.feasible_action_set,
                                                 state=t_state,
                                                 dqn=self.dqn,
                                                 device=self.device,
                                                 toggle_no_repeat=True)
                    if t_action is None:
                        if t < 2:
                            is_state_goal_feasible = False
                        break
                    if t_action is None or t_action < 0:
                        t_next_state, t_reward, t_done = None, -10, True
                    else:
                        t_next_state, t_reward, t_done, _ = env.step(t_action)
                    t_reward = t_reward
                    t_state = t_next_state  # state = next_state
                    # accumulate rewards
                    t_score += t_reward  # reward
                    traj_list.append([t_state, i_g])
                    traj_tmp.append(t_state)
                    if toggle_visual:
                        img_list.append(rsp.plot_states([t_state]).get_img())
                        plot = combine_images(img_list, columns=12)
                        # plot = np.concatenate((plot, rsp.plot_states([t_state]).get_img()), axis=1)
                        cv2.imshow(f"plot_eval", plot)
                        cv2.waitKey(100)
                    if t_done:
                        if t_reward >= 0:
                            num_dones += 1
                        total_return += t_score
                        total_len += t
                        break
                    # if t % int(np.ceil(reset_num * self.env.scheduler.state_level)) == 0:
                    if t > reset_multiplier * reset_num_multiply:
                        failed_data_list_tmp.append([i_s, i_g])
                        failed_data_path.append([traj_tmp, i_g])
                        if self.H_buffer is not None:
                            self.H_buffer.add.remote(traj_list)
                        break
                if not is_state_goal_feasible:
                    feasible_candidate_len -= 1
                else:
                    if toggle_visual:
                        print(f'Evaluation Num: {i + 1}/{len(candidate_state_goal)} \n'
                              f'Episode steps: {t} \n'
                              f'History {env.reward_history} \n'
                              f'Return: {t_score:<5.1f} \n')
                        cv2.waitKey(0)
                        del plot
                        plot = None
            # fs.dump_pickle(ray.get(self.shared_traj_buffer.get_info.remote('trajectory_list'))[-100:],
            #                'traj_list_1.pkl', reminder=False)
            if num_dones >= (feasible_candidate_len * pass_rate):
                if len(failed_data_list_tmp) > 0:
                    save_data_path = self.save_path.joinpath('debug_data.pkl')
                    # if save_data_path.exists():
                    #     failed_data_list = fs.load_pickle(str(save_data_path))
                    # else:
                    #     failed_data_list = []
                    # failed_data_list.extend(failed_data_list_tmp)
                    fs.dump_pickle(failed_data_list_tmp, str(save_data_path), reminder=False)
                    # debug
                # set to buffer
                # [buffer.put(copy.deepcopy(failed_data_list_tmp)) for buffer in self.difficult_case_buffer_list]
                mean_return = total_return / num_dones
                if mean_return > best_eval_score:
                    best_eval_score = mean_return
                    if self.save_checkpnt_path is not None:
                        torch.save({
                            'dqn_state_dict': self.dqn.state_dict(),
                        }, self.save_checkpnt_path.joinpath(f'model_best_{self.training_level}.chkpt'))
                if num_dones >= feasible_candidate_len * pass_rate:
                    # update training level
                    self.training_level = self.training_level + 1
                    is_update_training_level = True
                    RackState._cache.clear()
                    self.shared_state.set_info.remote('training_level', self.training_level)
                    #
                    self.shared_traj_buffer.set_info_clear.remote('trajectory_list')

                    best_eval_score = 0
            else:
                save_data_path = self.save_path.joinpath('debug_failed_data.pkl')
                save_data_path2 = self.save_path.joinpath('debug_failed_path.pkl')
                fs.dump_pickle(failed_data_list_tmp, str(save_data_path), reminder=False)
                fs.dump_pickle(failed_data_path, str(save_data_path2), reminder=False)
                torch.save({
                    'dqn_state_dict': self.dqn.state_dict(),
                }, self.save_checkpnt_path.joinpath(f'model_last.chkpt'))
            self.set_average_len(total_len / num_dones if num_dones > 0 else 0)
            if self.log_path is not None and isinstance(self.log_path, Path):
                write_to_csv(writer, [
                    ('timestamp', get_time_stamp(), '%1s'),
                    ('eval_average_len', total_len / num_dones if num_dones > 0 else -1, '%1f'),
                    ('eval_average_score', total_return / num_dones if num_dones > 0 else -1, '%1f'),
                    ('eval_num_dones', num_dones, '%1d'),
                    ('eval_num_dones_ratio', num_dones / self.eval_num, '%.1f'),
                    ('train_steps', train_steps, '%1d'),
                    ('training_level', self.training_level if not is_update_training_level else self.training_level - 1,
                     '%1d'),
                    ('eval_state_level', self.env.scheduler.state_level, '%1d'),
                    ('eval_class_level', self.env.scheduler.class_level, '%1d'),
                    ('reset_multiplier', reset_multiplier, '%1d')
                ])

            # self.logger.log({
            #     'eval_average_len': total_len / num_dones if num_dones > 0 else -1,
            #     'eval_average_score': total_return / num_dones if num_dones > 0 else -1,
            #     'eval_num_dones_ratio': num_dones / self.eval_num,
            #     'eval_state_level': scheduler.state_level,
            #     'eval_class_level': scheduler.class_level,
            # })
            # print(
            #     f">>>>>>> average score: {num_dones}/{self.eval_num}, "
            #     f"average len: {total_len / num_dones if num_dones > 0 else -1:.2f},"
            #     # f"a star solver dones: {a_star_num_dones}/{len(candidate_state_goal)}, "
            #     # f"a star solver average length: {a_star_total_len / a_star_num_dones if a_star_num_dones > 0 else -1:.2f} ",
            #     f"class level is {scheduler.class_level}, "
            #     f"state level is {scheduler.state_level}")

    def single_test(self, env: RackArrangementEnv,
                    eval_num=None,
                    toggle_reset_state=True,
                    toggle_sync_training=True,
                    toggle_show_every_step=False,
                    toggle_show_action_value=False,
                    toggle_reset=False,
                    toggle_visual=True,
                    img_scale=2):
        assert eval_num is None or isinstance(eval_num, int), f'eval_num should be None or int. Instead of {eval_num}'
        assert isinstance(env, RackArrangementEnv), f'Only accept the env as RackArrangementEnv. Instead of {type(env)}'

        ### copy to clipboard
        import win32clipboard, io
        from PIL import Image

        def send_to_clipboard(image):
            output = io.BytesIO()
            image.convert('RGB').save(output, 'BMP')
            data = output.getvalue()[14:]  # Remove the BMP header
            output.close()

            win32clipboard.OpenClipboard()  # Open the clipboard
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
            win32clipboard.CloseClipboard()

        def on_mouse_click_1(event, x, y, flags, param):
            if event == cv2.EVENT_RBUTTONDOWN:  # Right-click event
                # Convert the image from BGR to RGB
                img = IMG_COPY_ACTION_SEQUENCE
                image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                # Save the image to a memory buffer
                send_to_clipboard(pil_image)
                # Notify the user
                print("Image copied to clipboard!")

        def on_mouse_click_2(event, x, y, flags, param):
            if event == cv2.EVENT_RBUTTONDOWN:  # Right-click event
                # Convert the image from BGR to RGB
                img = IMG_COPY_ACTION_VALUE
                image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                # Save the image to a memory buffer
                send_to_clipboard(pil_image)
                # Notify the user
                print("Image copied to clipboard!")

        def inverse_value_rescale(value, eps=1e-3):
            temp = ((1 + 4 * eps * (value.abs() + 1 + eps)).sqrt() - 1) / (2 * eps)
            return value.sign() * (temp.square() - 1)

        if eval_num is None:
            eval_num = self.eval_num
        reset_num = self.reset_multiplier
        best_eval_score = 0
        toggle_visual = self.toggle_visual and toggle_visual
        self.env = env
        self.sync_dqn_checkpoint()
        if toggle_sync_training:
            self.sync_training_checkpoint()
        # return/number of dones/total len
        total_return = 0
        num_dones = 0
        total_len = 0
        # a star
        a_star_total_len = 0
        a_star_num_dones = 0

        # state_cons = RackState(np.array([[0, 0, 0, 1, 1, 0, 1, 0, 1],
        #                                  [0, 1, 0, 1, 1, 0, 0, 1, 0],
        #                                  [1, 1, 1, 1, 0, 1, 0, 0, 1],
        #                                  [1, 1, 0, 1, 1, 0, 1, 1, 1],
        #                                  [0, 1, 0, 1, 0, 0, 1, 1, 1],
        #                                  [1, 1, 1, 1, 1, 1, 0, 1, 1],
        #                                  [1, 1, 0, 1, 0, 1, 1, 0, 1],
        #                                  [0, 1, 0, 1, 1, 1, 0, 1, 1],
        #                                  [0, 1, 0, 1, 1, 1, 0, 0, 1]]))
        # delete_constraints = [self.env.to_action(
        #     self.env.rack_size,
        #     np.array([ii, jj]), np.array([2, 0]))
        #                          for ii in range(np.prod(self.env.state.state.shape[0])) for jj in
        #                          range(np.prod(self.env.state.state.shape[1])) ] + [
        #                          self.env.to_action(
        #                              self.env.rack_size,
        #                              np.array([ii, jj]), np.array([1, 0]))
        #                          for ii in range(np.prod(self.env.state.state.shape[0])) for jj in
        #                          range(np.prod(self.env.state.state.shape[1])) if ii != 1 and jj != 0]
        # print([self.env.from_action(self.env.rack_size, dd) for dd in delete_constraints])
        delete_constraints = []

        # env
        candidate_state_goal = []
        # candidate_state_goal += failed_data_list_tmp
        # failed data list
        failed_data_list_tmp = []
        if toggle_reset_state:
            for i in range(eval_num):
                for i in range(100):
                    s = env.reset()
                    if env.is_state_valid(s):
                        break
                g = env.goal_pattern
                candidate_state_goal.append([s, g])
        else:
            candidate_state_goal.append([env.state, env.goal_pattern])
        feasible_candidate_len = len(candidate_state_goal)
        for i, (i_s, i_g) in enumerate(candidate_state_goal):
            cv2.namedWindow('action_sequence')
            cv2.setMouseCallback('action_sequence', on_mouse_click_1)
            IMG_COPY_ACTION_SEQUENCE = None
            if toggle_show_action_value:
                cv2.namedWindow('action_value')
                cv2.setMouseCallback('action_value', on_mouse_click_2)
                IMG_COPY_ACTION_VALUE = None

            t_score = 0
            t_state = env.reset_state_goal(i_s, i_g)
            reset_mutiply = reset_num if self.env.scheduler.state_level < 25 else 3
            # print(env.scheduler.state_level, env.is_curriculum_lr)
            if toggle_visual:
                rsp = RackStatePlot(env.goal_pattern, )
                st_list = [t_state]
                image_list_plot_gradient = []
            # a start agent
            # is_done, path = TubePuzzle(t_state.state, goal_pattern).atarSearch(max_iter_cnt=50)
            # if is_done:
            #     a_star_total_len += (len(path) - 1)
            #     a_star_num_dones += 1
            # dqn agent
            reward_list = []
            is_state_goal_feasible = True
            for t in itertools.count(1):
                t_action, dqn_action_values = dqn_select_action(env=env,
                                                                # feasible_action_set=t_state.feasible_action_set,
                                                                feasible_action_set=np.setdiff1d(
                                                                    t_state.feasible_action_set,
                                                                    delete_constraints),
                                                                state=t_state,
                                                                dqn=self.dqn,
                                                                device=self.device,
                                                                toggle_no_repeat=True,
                                                                toggle_return_action_value=True, )
                print(delete_constraints, t_action)
                if t_action is None:
                    if t < 2:
                        is_state_goal_feasible = False
                    break
                if t_action is None or t_action < 0:
                    t_next_state, t_reward, t_done = None, -10, True
                else:
                    t_next_state, t_reward, t_done, _ = env.step(t_action, toggle_debug=False)
                t_reward = t_reward
                reward_list.append(t_reward)
                if toggle_visual:
                    image_list_plot_gradient.append([t_state, i_g, dqn_action_values])
                    st_list.append(t_next_state)
                t_state = t_next_state  # state = next_state
                # accumulate rewards
                t_score += t_reward  # reward

                if t_done:
                    if t_reward >= 0:
                        num_dones += 1
                    total_return += t_score
                    total_len += t
                    break
                if (not toggle_reset and t % 150 == 0) or \
                        (toggle_reset and t % int(np.ceil(reset_mutiply * self.env.scheduler.state_level)) == 0):
                    if toggle_visual:
                        print('reset num is', reset_num,
                              'env state level is', self.env.scheduler.state_level,
                              'reset num & state level is', int(np.ceil(reset_num * self.env.scheduler.state_level)),
                              'is done is', t_done, )
                        print(f'Evaluation Num: {i + 1}/{len(candidate_state_goal)} \n'
                              f'Episode steps: {t} \n'
                              f'History {env.reward_history} \n'
                              f'Return: {t_score:<5.1f} \n')
                        print(f"failed matrix is {repr(np.asarray(i_s))}")
                        plot = rsp.plot_states(st_list,
                                               row=8,
                                               img_scale=img_scale,
                                               reward_list=reward_list).get_img()
                        # plot = np.concatenate((plot, rsp.plot_states([t_state]).get_img()), axis=1)
                        IMG_COPY_ACTION_SEQUENCE = plot
                        cv2.imshow(f"action_sequence", plot)
                        image_list = []
                        if toggle_show_action_value:
                            for ps, pg, pv in image_list_plot_gradient:
                                img = rsp.plot_action_values(state=ps,
                                                             goal=pg,
                                                             action_values=pv,
                                                             scale=1).get_img()
                                image_list.append(img)
                                images = combine_images(image_list, columns=1)
                            IMG_COPY_ACTION_VALUE = images
                            cv2.imshow(f"action_value", images)
                        failed_data_list_tmp.append([i_s, i_g])
                        key = cv2.waitKey(0)
                        # check if the key is 's'
                        if key == 115:
                            fs.dump_pickle([st_list, env.goal_pattern],
                                           str(self.save_path.joinpath(
                                               f'{time.strftime("%Y%m%d%H%M%S")}_eval_{i}.pkl')),
                                           reminder=False)
                        del plot
                        plot = None
                    break
            if not is_state_goal_feasible:
                feasible_candidate_len -= 1
            else:
                if toggle_visual:
                    print(f'Evaluation Num: {i + 1}/{len(candidate_state_goal)} \n'
                          f'Episode steps: {t} \n'
                          f'History {env.reward_history} \n'
                          f'Return: {t_score:<5.1f} \n')
                    if toggle_show_every_step:
                        plot = rsp.plot_states(st_list,
                                               row=8,
                                               img_scale=img_scale,
                                               reward_list=reward_list).get_img()
                        # plot = np.concatenate((plot, rsp.plot_states([t_state]).get_img()), axis=1)
                        IMG_COPY_ACTION_SEQUENCE = plot
                        cv2.imshow(f"action_sequence", plot)
                        image_list = []
                        if toggle_show_action_value:
                            for ps, pg, pv in image_list_plot_gradient:
                                img = rsp.plot_action_values(state=ps,
                                                             goal=pg,
                                                             action_values=pv,
                                                             scale=1).get_img()
                                image_list.append(img)
                                images = combine_images(image_list, columns=1)
                            IMG_COPY_ACTION_VALUE = images
                            cv2.imshow(f"action_value", images)
                        failed_data_list_tmp.append([i_s, i_g])
                        key = cv2.waitKey(0)
                        # check if the key is 's'
                        if key == 115:
                            fs.dump_pickle([st_list, env.goal_pattern],
                                           str(self.save_path.joinpath(
                                               f'{time.strftime("%Y%m%d%H%M%S")}_eval_{i}.pkl')),
                                           reminder=False)
                        del plot
                        plot = None
        print(f"number dones: {num_dones}/{self.eval_num}, ")


class Eval2(object):
    def __init__(self,
                 env: RackArrangementEnv,
                 net: 'DDQN',
                 cfg: dict,
                 shared_state: ray.actor.ActorClass,
                 save_path: str = None,
                 log_path: str = None,
                 toggle_visual=False,
                 ):
        assert isinstance(env, RackArrangementEnv), f'Only accept the env as RackArrangementEnv. ' \
                                                    f'Instead of {type(env)}'

        self.env = env
        self.training_level = self.env.scheduler.state_level
        self.log_path = Path(log_path) if log_path is not None else None
        self.save_path = Path(save_path)
        self.save_checkpnt_path = Path(save_path).joinpath('model_best.chkpt') if save_path is not None else None
        self.device = cfg['device']
        # network
        self.dqn = net.to(self.device)
        self.dqn.eval()
        # set up replay buffer size
        self.shared_state = shared_state
        # hyper-parameter
        self.eval_num = cfg['eval_num']
        self.reset_num = cfg['reset_num']
        self.eval_interval = cfg['eval_interval']
        # is-debug
        self.toggle_visual = toggle_visual

    def sync_dqn_checkpoint(self):
        weights = ray.get(self.shared_state.get_info.remote('weights'))
        # breakpoint()
        self.dqn.load_state_dict(weights)

    def sync_training_checkpoint(self):
        ''' Sync training level + Reset epsilon if training level update'''
        training_level: int = ray.get(self.shared_state.get_info.remote('training_level'))
        # if self.training_level != training_level:
        self.training_level = training_level
        self.env.scheduler.set_training_level(self.training_level)

    def single_test(self, env: RackArrangementEnv, eval_num=None):
        assert eval_num is None or isinstance(eval_num, int), f'eval_num should be None or int. Instead of {eval_num}'
        assert isinstance(env, RackArrangementEnv), f'Only accept the env as RackArrangementEnv. Instead of {type(env)}'
        if eval_num is None:
            eval_num = self.eval_num
        reset_num = self.reset_num
        best_eval_score = 0
        toggle_visual = self.toggle_visual
        self.env = env
        self.sync_dqn_checkpoint()
        self.sync_training_checkpoint()
        # return/number of dones/total len
        total_return = 0
        num_dones = 0
        total_len = 0
        # a star
        a_star_total_len = 0
        a_star_num_dones = 0

        # env
        candidate_state_goal = []
        # candidate_state_goal += failed_data_list_tmp
        # failed data list
        failed_data_list_tmp = []
        for i in range(eval_num):
            s = env.reset()
            g = env.goal_pattern
            candidate_state_goal.append([s, g])
        feasible_candidate_len = len(candidate_state_goal)
        for i, (i_s, i_g) in enumerate(candidate_state_goal):
            t_score = 0
            t_state = env.reset_state_goal(i_s, i_g)
            # print(env.scheduler.state_level, env.is_curriculum_lr)
            if toggle_visual:
                rsp = RackStatePlot(env.goal_pattern, )
                st_list = [t_state]
            # a start agent
            # is_done, path = TubePuzzle(t_state.state, goal_pattern).atarSearch(max_iter_cnt=50)
            # if is_done:
            #     a_star_total_len += (len(path) - 1)
            #     a_star_num_dones += 1
            # dqn agent
            is_state_goal_feasible = True
            for t in itertools.count(1):
                t_action = dqn_select_action(env=env,
                                             feasible_action_set=t_state.feasible_action_set,
                                             state=t_state,
                                             dqn=self.dqn,
                                             device=self.device,
                                             toggle_no_repeat=True)
                if t_action is None:
                    if t < 2:
                        is_state_goal_feasible = False
                    break
                if t_action is None or t_action < 0:
                    t_next_state, t_reward, t_done = None, -10, True
                else:
                    t_next_state, t_reward, t_done, _ = env.step(t_action)
                t_reward = t_reward
                t_state = t_next_state  # state = next_state
                # accumulate rewards
                t_score += t_reward  # reward
                if toggle_visual:
                    st_list.append(t_state)
                    input('press any key to continue')
                if t_done:
                    if t_reward >= 0:
                        num_dones += 1
                    total_return += t_score
                    total_len += t
                    break
                if t % int(np.ceil(reset_num * self.training_level)) == 0:
                    failed_data_list_tmp.append([i_s, i_g])
                    break
            if not is_state_goal_feasible:
                feasible_candidate_len -= 1
            else:
                if toggle_visual:
                    plot = rsp.plot_states(st_list).get_img()
                    # plot = np.concatenate((plot, rsp.plot_states([t_state]).get_img()), axis=1)
                    cv2.imshow(f"plot_eval", plot)
                    print(f'Evaluation Num: {i + 1}/{len(candidate_state_goal)} \n'
                          f'Episode steps: {t} \n'
                          f'History {env.reward_history} \n'
                          f'Return: {t_score:<5.1f} \n')
                    cv2.waitKey(0)
                    del plot
                    plot = None
        if num_dones >= (feasible_candidate_len * .99):
            if len(failed_data_list_tmp) > 0:
                save_data_path = self.save_path.joinpath('debug_data.pkl')
                # if save_data_path.exists():
                #     failed_data_list = fs.load_pickle(str(save_data_path))
                # else:
                #     failed_data_list = []
                # failed_data_list.extend(failed_data_list_tmp)
                fs.dump_pickle(failed_data_list_tmp, str(save_data_path), reminder=False)

            # set to buffer
            # [buffer.put(copy.deepcopy(failed_data_list_tmp)) for buffer in self.difficult_case_buffer_list]
            mean_return = total_return / num_dones
            if num_dones >= feasible_candidate_len * 0.99:
                # update training level
                self.training_level = self.training_level + 1
                RackState._cache.clear()
                self.shared_state.set_info.remote('training_level', self.training_level)
                self.shared_traj_buffer.set_info_clear.remote('trajectory_list')


@ray.remote
class FIFOBufferActor:
    def __init__(self, capacity: int,
                 rnd_model: 'RNDModel',
                 shared_state: ray.actor.ActorClass,
                 num_classes: int,
                 device: torch.device = torch.device('cpu')):
        self.buffer = []
        self.capacity = capacity
        self.rnd_model = rnd_model.to(device)
        self.rnd_model.eval()
        self.num_classes = num_classes
        self.device = device
        self.shard_state = shared_state
        self.training_level = ray.get(self.shard_state.get_info.remote('training_level'))

    def sync_state(self, training_level: int = None):
        self.rnd_model.load_state_dict(ray.get(self.shard_state.get_info.remote('rnd_model_weights')))
        if training_level is None:  # if training level is not specified, get from shared state
            training_level: int = ray.get(self.shared_state.get_info.remote('training_level'))
        if self.training_level != training_level:  # if training level is updated, clear buffer
            self.training_level = training_level  # update training level
            self.clear_all()  # clear buffer

    def add(self, items: list):
        for item in items:
            if len(self.buffer) >= self.capacity:
                self.buffer.pop(0)  # Remove the oldest item
            self.buffer.append(item)  # Add as new item

    def local_planning(self, batch_sz, n=1):
        if batch_sz <= len(self.buffer):
            return None
        try:
            indices = np.random.choice(len(self.buffer), size=batch_sz, replace=False)
        except:
            return None
        u_s_list = []
        if len(indices) == 0:
            return None
        with torch.no_grad():
            for i in indices:
                state, goal = self.buffer[i]
                predict_next_state_feature, target_next_state_feature = self.rnd_model(
                    torch.as_tensor(
                        np.concatenate((state.abs_state(self.num_classes), goal.abs_state(self.num_classes)), axis=0),
                        dtype=torch.float32,
                        device=self.device).unsqueeze(0), )
                u_s = np.linalg.norm((target_next_state_feature - predict_next_state_feature).detach().cpu().numpy())
                u_s_list.append(u_s)
        try:
            # Get the indices that would sort the u_s_list
            sorted_indices = np.argsort(u_s_list)
            # Get the last n indices which correspond to the max values
            max_n_indices = sorted_indices[-n:]
            return [self.buffer[indices[i]] for i in max_n_indices]
        except Exception as e:
            print(indices, u_s_list)
            raise Exception(e)

    def get_all(self):
        return self.buffer

    def len(self):
        return len(self.buffer)

    def clear_all(self):
        self.buffer = []


if __name__ == '__main__':
    pass
