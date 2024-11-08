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
from huri.learning.env.rack_v3.env2 import RackArrangementEnv, RackStatePlot, RackState, setdiff2d
from huri.learning.method.AlphaZero.log import CsvWriter, write_to_csv
from huri.components.utils.img_utils import combine_images
from huri.learning.method.AlphaZero.utils import get_time_stamp
import huri.core.file_sys as fs
from ray.rllib.policy.sample_batch import SampleBatch

MAX_NUM_PENDING_TASKS = 50


def padding(size, array):
    if len(array) == 0:
        return np.zeros(size, dtype=int)
    max_v = array[-1]
    pad_array = np.ones((size, *array.shape[1:]), dtype=int) * max_v
    pad_array[:len(array)] = array
    return pad_array


class Trajectory(object):
    action_dim = -1

    @staticmethod
    def transform(x, num_categories=2):
        x = x.state
        # Create a one-hot array of zeros of appropriate size
        one_hot = np.zeros((num_categories + 1, x.shape[0], x.shape[1]))
        # Use fancy indexing to fill the right spots with ones
        one_hot[x, np.arange(x.shape[0])[:, None], np.arange(x.shape[1])] = 1
        return one_hot[1:]

    def __init__(self, goal):
        self.goal = goal
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def add_transition(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        if isinstance(index, slice):
            states = self.states[index]
            actions = self.actions[index]
            rewards = self.rewards[index]
            next_states = self.next_states[index]
            dones = self.dones[index]
            return SampleBatch({'state': states,
                                'action': actions,
                                'reward': rewards,
                                'next_state': next_states,
                                'next_state_feasible_action':
                                    [padding(int(self.action_dim ** 2 / 4) + 1, next_state.feasible_action_set) for
                                     next_state in
                                     next_states],
                                'done': dones,
                                'goal': [self.goal] * len(states)})
        else:
            state = self.states[index]
            action = self.actions[index]
            reward = self.rewards[index]
            next_state = self.next_states[index]
            done = self.dones[index]
            return SampleBatch({'state': [state],
                                'action': [action],
                                'reward': [reward],
                                'next_state': [next_state],
                                'next_state_feasible_action': [
                                    padding(int(self.action_dim ** 2 / 4) + 1, next_state.feasible_action_set)],
                                'done': [done],
                                'goal': [self.goal]})

    def __repr__(self):
        return (self.states[0], *(repr(s) for s in self.next_states))


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

    def get_info_pop(self, key):
        assert isinstance(key, str) and isinstance(self.current_checkpoint[key], list)
        if len(self.current_checkpoint[key]) > 0:
            return self.current_checkpoint[key].pop()
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
                      toggle_no_repeat: bool = True, ):
    if toggle_no_repeat:
        feasible_action_set_tmp = feasible_action_set
        repeat_acts = []
        for _ in env.rack_state_history:
            act = env.action_between_states_constraint_free(state, _)
            if act is not None:
                repeat_acts.append(act)
        feasible_action_set = setdiff2d(feasible_action_set, np.asarray(repeat_acts), env.rack_size)
        if len(feasible_action_set) == 0:
            feasible_action_set = feasible_action_set_tmp
            # self.early_stop = True
            # # print("Potential ERROR Happens")
            # print("state is", self.env.state)
            # print("goal pattern is", self.env.goal_pattern)
    if len(feasible_action_set) < 1:
        return None
    with torch.no_grad():
        # feasible_action_set = torch.as_tensor(feasible_action_set, dtype=torch.int64, device=device)
        unique_head_index = np.unique(feasible_action_set[:, 0])
        dqn_action_value_list = dqn(
            torch.as_tensor(
                np.concatenate((state.abs_state(env.num_classes), env.goal_pattern.abs_state(env.num_classes)), axis=0),
                dtype=torch.float32, device=device).unsqueeze(0),
            k=unique_head_index)
        max_score = -np.inf
        selected_action = None
        for ii, v in enumerate(unique_head_index):
            feasible_action_set_v = feasible_action_set[np.where(feasible_action_set[:, 0] == v), 1].flatten()
            dqn_action_value = dqn_action_value_list[ii].squeeze()[feasible_action_set_v]
            dqn_action_value_max_score = dqn_action_value.max()
            if dqn_action_value_max_score > max_score:
                max_score = dqn_action_value_max_score
                # detect if feasible_action_set_v is single_value
                selected_action = np.array([unique_head_index[ii], feasible_action_set_v[dqn_action_value.argmax()]])
    return selected_action


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


@ray.remote
class Actor(object):
    """
    Actor object to start simulation and collect data
    """

    def __init__(self,
                 actor_id: int,
                 env: RackArrangementEnv,
                 net: 'DDQN',
                 cfg: dict,
                 replay_buffer: ray.actor.ActorClass,
                 shared_state: ray.actor.ActorClass,
                 log_path=None,
                 toggle_visual=False,
                 ):
        assert isinstance(env, RackArrangementEnv), f'Only accept the env as RackArrangementEnv. ' \
                                                    f'Instead of {type(env)}'

        self._actor_id = actor_id
        self.env = env
        self.training_level = self.env.scheduler.state_level
        self.scheduler = self.env.scheduler
        self.device = cfg['actor_device']
        # network
        self.dqn = net.to(self.device)
        self.dqn.eval()
        # set up replay buffer size
        self.replay_buffer = replay_buffer
        self.shared_state = shared_state
        # set up scheduler
        self.epsilon_decay = cfg['eps_decay']
        self.max_epsilon = cfg['eps_max']
        self.min_epsilon = cfg['eps_min']
        self.epsilon_scheduler = EpsilonScheduler(self.max_epsilon, self.min_epsilon, self.epsilon_decay)

        self.store_replay_buffer = cfg['store_replay_buffer']
        # hyper-parameter
        self.reset_num = cfg['reset_num']
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

    @property
    def epsilon(self):
        return self.epsilon_scheduler.get_epsilon()

    def sync_dqn_checkpoint(self):
        weights = ray.get(self.shared_state.get_info.remote('weights'))
        self.dqn.load_state_dict(weights)

    def sync_training_checkpoint(self):
        ''' Sync training level + Reset epsilon if training level update'''
        training_level: int = ray.get(self.shared_state.get_info.remote('training_level'))
        if self.training_level != training_level:
            self.training_level = training_level
            self.env.scheduler.set_training_level(self.training_level)
            self.sync_dqn_checkpoint()
            self.epsilon_scheduler.reset()

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
                                                toggle_no_repeat=toggle_no_repeat)
        return selected_action

    def store_traj_list(self, trajectory_list, store_reanalyzer=False):
        if store_reanalyzer:
            self.shared_state.set_info.remote('trajectory_list', ray.get(
                self.shared_state.get_info.remote('trajectory_list')) + trajectory_list)

        replay_buffer = self.replay_buffer
        for traj in trajectory_list:
            if len(self.result_refs) > MAX_NUM_PENDING_TASKS:
                ready_refs, self.result_refs = ray.wait(self.result_refs, num_returns=MAX_NUM_PENDING_TASKS)
                ray.get(ready_refs)
                self.result_refs = []
            self.result_refs.append(replay_buffer.add.remote(traj[:]))
            # for i in range(len(traj)):
            #     breakpoint()
            #     replay_buffer.add(traj[i])
            # for (_goal, _state, _action, _reward, _next_state, _done) in self.env.synthesize_reflection_data(
            #         goal=goal,
            #         state=state,
            #         reward=reward,
            #         next_state=next_state,
            #         done=done):
            #     replay_buffer.store(goal=_goal, state=_state, action=_action, reward=_reward,
            #                         next_state=_next_state,
            #                         done=_done)
            del traj
        trajectory_list[:] = []

    def start(self):
        """Actor starts"""
        env = self.env
        scheduler = self.scheduler
        reset_num = self.reset_num
        target_update_freq = self.target_update_freq
        toggle_visual = self.toggle_visual
        trajectory_list = []
        Trajectory.action_dim = self.env.action_space_dim
        if self.log_path is not None and isinstance(self.log_path, Path):
            writer = CsvWriter(str(self.log_path.joinpath(f'actor_{self._actor_id}_log.csv')))
        # ====
        # dqn load shared net weight
        self.sync_dqn_checkpoint()
        # start actors
        step = 0
        while 1:
            # rest counter, score
            reset_cnt = 0
            score = 0
            # curriculum learning
            if step % 10 == 0:  # reduce the number of times to access the variables
                self.sync_training_checkpoint()
            # init epoch
            ## state reset
            state = env.reset()
            ## create a class to hold trajectory
            traj = Trajectory(env.goal_pattern)
            start_ep_t = time.time()
            is_success_traj = True
            if toggle_visual:
                rsp = RackStatePlot(env.goal_pattern, )
                plot = rsp.plot_states([state]).get_img()
                img_list = [plot]
            for _ in itertools.count():
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
                # linearly decrease epsilon
                self.epsilon_scheduler.step()
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
                    # if self.save_data:
                    #     self.shared_reanalyzer_mem.put(
                    #         [self.state_level, self.class_level, self.replaybuffer_tmp_store])
                    break
                if reset_cnt % reset_num == 0:
                    # TODO
                    trajectory_list.append(traj)
                    # if self.save_data:
                    #     self.shared_reanalyzer_mem.put(
                    #         [self.state_level, self.class_level, self.replaybuffer_tmp_store])
                    break
            end_ep_t = time.time()
            if not is_success_traj:
                continue
            if len(trajectory_list) >= self.send_period:
                # self.store_traj_list(trajectory_list,
                #                      self.store_reanalyzer and (True if self.epsilon > np.random.random() else False))

                self.store_traj_list(trajectory_list,
                                     self.store_reanalyzer)

                # pass
            if self.log_path is not None and isinstance(self.log_path, Path):
                write_to_csv(writer, [
                    ('timestamp', get_time_stamp(), '%1s'),
                    ('time_consumption', end_ep_t - start_ep_t, '%1s'),
                    ('episode_lens', len(traj), '%1d'),
                    ('acc_rewards', score, '%1f'),
                    ('state_level', scheduler.state_level, '%1d'),
                    ('class_level', scheduler.class_level, '%1d'),
                    ('epsilon', self.epsilon, '%1f'),
                ])


@ray.remote
class Eval(object):
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
        self.save_checkpnt_path = Path(save_path) if save_path is not None else None
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

    def start(self):
        # init
        if self.log_path is not None and isinstance(self.log_path, Path):
            writer = CsvWriter(str(self.log_path.joinpath(f'eval_log.csv')))
        env = self.env
        reset_num = self.reset_num
        best_eval_score = 0
        toggle_visual = self.toggle_visual
        train_steps = 0
        pass_rate = self.pass_rate
        while 1:
            train_steps_tmp = ray.get(self.shared_state.get_info.remote('train_steps'))
            if train_steps_tmp == train_steps:
                time.sleep(.5)
                continue
            train_steps = train_steps_tmp
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
            for i in range(self.eval_num):
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
                    if t_action is None:
                        t_next_state, t_reward, t_done = None, -10, True
                    else:
                        t_next_state, t_reward, t_done, _ = env.step(t_action)
                    t_reward = t_reward
                    t_state = t_next_state  # state = next_state
                    # accumulate rewards
                    t_score += t_reward  # reward
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
                    if t % int(np.ceil(reset_num * self.env.scheduler.state_level)) == 0:
                        failed_data_list_tmp.append([i_s, i_g])
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
            if num_dones >= (feasible_candidate_len * pass_rate):
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
                if mean_return > best_eval_score:
                    best_eval_score = mean_return
                    if self.save_checkpnt_path is not None:
                        torch.save({
                            'dqn_state_dict': self.dqn.state_dict(),
                        }, self.save_checkpnt_path.joinpath(f'model_best_{self.training_level}.chkpt'))
                if num_dones >= feasible_candidate_len * pass_rate:
                    # update training level
                    self.training_level = self.training_level + 1
                    RackState._cache.clear()
                    self.shared_state.set_info.remote('training_level', self.training_level)
                    self.shared_state.set_info_clear.remote('trajectory_list')
                    best_eval_score = 0
            else:
                torch.save({
                    'dqn_state_dict': self.dqn.state_dict(),
                }, self.save_checkpnt_path.joinpath(f'model_last.chkpt'))
            if self.log_path is not None and isinstance(self.log_path, Path):
                write_to_csv(writer, [
                    ('timestamp', get_time_stamp(), '%1s'),
                    ('eval_average_len', total_len / num_dones if num_dones > 0 else -1, '%1f'),
                    ('eval_average_score', total_return / num_dones if num_dones > 0 else -1, '%1f'),
                    ('eval_num_dones', num_dones, '%1d'),
                    ('eval_num_dones_ratio', num_dones / self.eval_num, '%.1f'),
                    ('train_steps', train_steps, '%1d'),
                    ('training_level', self.training_level, '%1d'),
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

    def single_test(self, env: RackArrangementEnv, eval_num=None, toggle_sync_training=True):
        assert eval_num is None or isinstance(eval_num, int), f'eval_num should be None or int. Instead of {eval_num}'
        assert isinstance(env, RackArrangementEnv), f'Only accept the env as RackArrangementEnv. Instead of {type(env)}'
        if eval_num is None:
            eval_num = self.eval_num
        reset_num = self.reset_num
        best_eval_score = 0
        toggle_visual = self.toggle_visual
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
                if t_action is None:
                    t_next_state, t_reward, t_done = None, -10, True
                else:
                    t_next_state, t_reward, t_done, _ = env.step(t_action)
                t_reward = t_reward
                t_state = t_next_state  # state = next_state
                # accumulate rewards
                t_score += t_reward  # reward
                if toggle_visual:
                    st_list.append(t_state)
                if t_done:
                    if t_reward >= 0:
                        num_dones += 1
                    total_return += t_score
                    total_len += t
                    break
                if t % int(np.ceil(reset_num * self.env.scheduler.state_level)) == 0:
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
            if mean_return > best_eval_score:
                best_eval_score = mean_return
                if self.save_checkpnt_path is not None:
                    torch.save({
                        'dqn_state_dict': self.dqn.state_dict(),
                    }, self.save_checkpnt_path)
            if num_dones >= feasible_candidate_len * 0.99:
                # update training level
                self.training_level = self.training_level + 1
                RackState._cache.clear()
                self.shared_state.set_info.remote('training_level', self.training_level)
                self.shared_state.set_info_clear.remote('trajectory_list')


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
                self.shared_state.set_info_clear.remote('trajectory_list')


if __name__ == '__main__':
    pass
