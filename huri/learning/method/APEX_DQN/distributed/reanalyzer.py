""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230720osaka

"""

import time
import ray
import cv2
import numpy as np
from typing import List
import huri.core.file_sys as fs
from huri.learning.env.rack_v3.env import RackStatePlot, RackState, RackArrangementEnv
from huri.learning.A_start_teacher.A_star_teacher import rm_ras_actions_recur, rm_ras_actions_recur2, \
    rm_ras_actions_recur3, a_star_solve
from huri.learning.method.AlphaZero.log import CsvWriter, write_to_csv
from huri.learning.method.AlphaZero.utils import get_time_stamp
from huri.learning.method.APEX_DQN.distributed.utils import (synthetic_traj_reflection_lrud,
                                                             synthetic_traj_reflection_lr,
                                                             synthetic_traj_category,
                                                             synthetic_traj_direction,
                                                             synthetic_traj_reflection_ud,
                                                             padding,
                                                             SampleBatch,
                                                             Trajectory)


def merge_dictionaries(dictionaries):
    merged_dict = {}

    for dictionary in dictionaries:
        for key, value in dictionary.items():
            if key not in merged_dict:
                merged_dict[key] = [value]
            else:
                merged_dict[key].append(value)

    return merged_dict


def extract_path_from_traj(traj):
    goal_pattern = np.array(traj.goal)
    return (np.array([*traj.states, traj.next_states[-1]]),
            [str(np.concatenate((goal_pattern,
                                 np.array(traj.states[i]),
                                 np.array(traj.next_states[i])))) for i in range(len(traj))]
            , goal_pattern)


def refined_path_to_transitions_recur(env: RackArrangementEnv,
                                      refined_path: List[List[np.ndarray]],
                                      goal_pattern: np.ndarray):
    assert isinstance(env, RackArrangementEnv), f'env should be RackArrangementEnv instead of {type(env)}'
    transitions = []
    # reward_debug = []
    for _i in np.arange(len(refined_path) - 1):
        s_current, s_next = refined_path[_i], refined_path[_i + 1]
        action = env.action_between_states(s_current=s_current, s_next=s_next)
        is_finsihed = env.is_finished(s_next, goal_pattern)
        # check if it is a unfinished state
        # if _i == len(refined_path) - 2 and not is_finsihed:
        #     reward = -50
        #     print("!!!!!!!!!!", s_current, s_next)
        # else:
        reward = env._get_reward(is_finsihed, s_current, s_next, goal_pattern)
        tran_tmp = {'state': s_current,
                    'action': action,
                    'reward': reward,
                    'next_state': s_next,
                    'next_state_feasible_action': padding(int(env.action_space_dim * 0.575),
                                                          RackState(s_next).feasible_action_set),
                    'done': is_finsihed,
                    'goal': goal_pattern}
        transitions.append(tran_tmp)
        if is_finsihed:
            break
    return transitions


def refined_path_to_traj_recur(env: RackArrangementEnv,
                               refined_path: List[List[np.ndarray]],
                               action_dim: int,
                               n_step: int,
                               gamma: float,
                               goal_pattern: np.ndarray,
                               num_categories):
    traj = Trajectory(goal=goal_pattern,
                      action_dim=action_dim,
                      n_step=n_step,
                      gamma=gamma,
                      num_categories=num_categories)
    for _i in np.arange(len(refined_path) - 1):
        s_current, s_next = refined_path[_i], refined_path[_i + 1]
        action = env.action_between_states(s_current=s_current, s_next=s_next)
        is_finsihed = env.is_finished(s_next, goal_pattern)
        # check if it is a unfinished state
        # if _i == len(refined_path) - 2 and not is_finsihed:
        #     reward = -50
        #     print("!!!!!!!!!!", s_current, s_next)
        # else:
        reward = env._get_reward(is_finsihed, s_current, s_next, goal_pattern)
        traj.add_transition(state=s_current,
                            action=action,
                            reward=reward,
                            next_state=s_next,
                            done=is_finsihed, )
        if is_finsihed:
            break
    return traj


def synthetic_traj(traj, env, sync_func: callable, toggle_debug=False):
    traj = traj.copy()
    input_data = np.array([*traj.states, traj.next_states[-1]]), traj.goal
    if isinstance(sync_func, list):
        for sync_func_ in sync_func:
            input_data = sync_func_(*input_data)
    elif callable(sync_func):
        input_data = sync_func(*input_data)
    else:
        raise NotImplementedError
    states_swap, goal_swap = input_data
    if toggle_debug:
        rsp = RackStatePlot(goal_swap, )
        plot = rsp.plot_states(states_swap, row=15).get_img()
        cv2.imshow(f"plot_swap", plot)

        rsp = RackStatePlot(traj.goal, )
        plot = rsp.plot_states(np.array([*traj.states, traj.next_states[-1]]), row=15).get_img()
        cv2.imshow(f"plot_no_swap", plot)
        cv2.waitKey(0)
    data = merge_dictionaries(refined_path_to_transitions_recur(env, states_swap, goal_swap))
    traj.goal = goal_swap
    traj.states = data['state']
    traj.actions = data['action']
    traj.rewards = data['reward']
    traj.next_states = data['next_state']
    traj.dones = data['done']
    return traj


def synthetic_traj_all(traj, env, toggle_debug=False) -> List:
    sync_traj_list = []
    # sync_func = [synthetic_traj_direction]
    # sync_func += np.random.choice([synthetic_traj_reflection_lrud,
    #                                synthetic_traj_reflection_lr,
    #                                synthetic_traj_reflection_ud], size=1, replace=False).tolist()
    if len(traj.states) < 1:
        breakpoint()
        return sync_traj_list
    # if len(traj.states[0].num_classes) > 1:
    #     sync_traj_list.extend([synthetic_traj(traj, env, synthetic_traj_category, toggle_debug=toggle_debug),
    #                            synthetic_traj(traj, env, [synthetic_traj_category,
    #                                                       synthetic_traj_reflection_lrud], toggle_debug=toggle_debug),
    #                            synthetic_traj(traj, env, [synthetic_traj_category,
    #                                                       synthetic_traj_reflection_lr], toggle_debug=toggle_debug),
    #                            synthetic_traj(traj, env, [synthetic_traj_category,
    #                                                       synthetic_traj_reflection_ud], toggle_debug=toggle_debug)])
    sync_traj_list.extend([synthetic_traj(traj, env, synthetic_traj_reflection_lrud, toggle_debug=toggle_debug),
                           synthetic_traj(traj, env, synthetic_traj_reflection_lr, toggle_debug=toggle_debug),
                           synthetic_traj(traj, env, synthetic_traj_reflection_ud, toggle_debug=toggle_debug), ]
                          # synthetic_traj(traj, env, [synthetic_traj_direction,
                          #                            synthetic_traj_reflection_lrud], toggle_debug=toggle_debug),
                          # synthetic_traj(traj, env, [synthetic_traj_direction,
                          #                            synthetic_traj_reflection_lr], toggle_debug=toggle_debug),
                          # synthetic_traj(traj, env, [synthetic_traj_direction,
                          #                            synthetic_traj_reflection_ud], toggle_debug=toggle_debug)],
                          )
    return sync_traj_list


@ray.remote(num_cpus=1)
class Reanalyzer(object):
    def __init__(self,
                 uid: int,
                 env: RackArrangementEnv,
                 shared_state: ray.actor.ActorClass,
                 replay_buffer: ray.actor.ActorClass,
                 pop_buff_len:int = 10,
                 toggle_her_replay=True,
                 toggle_visual=False,
                 log_path=None, ):
        self._id = uid
        self.env = env
        # set up replay buffer size
        self.replay_buffer = replay_buffer
        self.shared_state = shared_state
        # her replay
        self.toggle_her_replay = toggle_her_replay
        # is-debug
        self.toggle_visual = toggle_visual
        self.log_path = fs.Path(log_path) if log_path is not None else None
        # training level
        self.training_level = 1
        self.pop_buff_len = pop_buff_len

    def gen_bootstrapping_data(self,
                               traj,
                               horizon=50,
                               toggle_debug=False):
        store_transitions = []
        synthetic_traj_list = []
        redundant_path, redundant_abs_state_paired_str, goal_pattern = extract_path_from_traj(traj)
        action_dim = traj.action_dim
        n_step = traj.n_step
        gamma = traj.gamma
        num_categories = traj.num_categories
        del traj
        goal_state_np = goal_pattern
        if not np.array_equal(redundant_path[-1], goal_state_np):
            # find the path to goal
            path = a_star_solve(redundant_path[-1], goal_state_np, )
            if len(path) > 0:
                redundant_path = np.concatenate((redundant_path, path[1:]))
                # redundant_abs_state_paired_str = [  # update the redundant_abs_state_paired_str
                #     str(np.concatenate((goal_state_np, path[i], path[i + 1])))
                #     for i in range(len(path) - 1)]
                if self.toggle_visual:
                    rsp = RackStatePlot(goal_state_np, )
                    img = rsp.plot_states(redundant_path, row=13).get_img()
                    cv2.imshow('complete the goal', img)
                    cv2.waitKey(0)
        else:
            goal_state_np = redundant_path[-1]

        refined_path, refined_path_her = rm_ras_actions_recur3(redundant_path,
                                                               h=horizon,
                                                               goal_pattern=redundant_path[-1],
                                                               max_refine_num=200,
                                                               infeasible_dict={}, )

        # if not np.array_equal(redundant_path[-1], goal_state_np):
        #     goal_state_np = redundant_path[-1]

        for i in range(len(refined_path)):
            if len(refined_path[i]) >= len(redundant_path) - i:
                continue
            if len(refined_path[i]) < 2:
                continue
            if self.toggle_visual:
                rsp = RackStatePlot(goal_state_np, )
                img = rsp.plot_states(refined_path[i], ).get_img()
                cv2.imshow(f"plot_{self._id}", img)
                cv2.waitKey(0)
            refined_traj = refined_path_to_traj_recur(self.env,
                                                      refined_path[i],
                                                      action_dim=action_dim,
                                                      n_step=n_step,
                                                      gamma=gamma,
                                                      goal_pattern=goal_state_np,
                                                      num_categories=num_categories)

            synthetic_traj_list += synthetic_traj_all(refined_traj, self.env, toggle_debug=False)
            if len(refined_traj) > 0 and toggle_debug:
                print(f"Add refined data into replay buffer: {len(refined_traj) + 1}/{len(redundant_path)}")
            samplebatch = refined_traj[:]
            for _ in range(len(samplebatch)):
                t = samplebatch.slice(_, _ + 1)
                _t_paired = str(np.concatenate((refined_traj.goal,
                                 np.array(refined_traj.states[_]),
                                 np.array(refined_traj.next_states[_]))))
                if _t_paired not in redundant_abs_state_paired_str:
                    store_transitions.append(t)
                    redundant_abs_state_paired_str.append(str(_t_paired))

        # her
        if self.toggle_her_replay:
            goal_select_ids = np.arange(1, len(redundant_path) - 1)
            goal_pattern_set = redundant_path[goal_select_ids]
            for i in range(len(refined_path_her)):
                if len(refined_path_her[i]) >= len(redundant_path) - i:
                    continue
                if len(refined_path_her[i]) < 2:
                    continue
                goal_state_tmp_np = goal_pattern_set[i]
                refined_traj = refined_path_to_traj_recur(self.env,
                                                          refined_path_her[i],
                                                          action_dim=action_dim,
                                                          n_step=n_step,
                                                          gamma=gamma,
                                                          goal_pattern=goal_state_tmp_np,
                                                          num_categories=num_categories)
                synthetic_traj_list += synthetic_traj_all(refined_traj, self.env, toggle_debug=False)
                if self.toggle_visual:
                    rsp = RackStatePlot(goal_state_tmp_np, )
                    img = rsp.plot_states(refined_path_her[i]).get_img()
                    cv2.imshow(f"goal_her_plot_{self._id}", img)
                    cv2.waitKey(0)
                samplebatch = refined_traj[:]
                for _ in range(len(samplebatch)):
                    t = samplebatch.slice(_, _ + 1)
                    _t_paired = str(np.concatenate((refined_traj.goal,
                                                    np.array(refined_traj.states[_]),
                                                    np.array(refined_traj.next_states[_]))))
                    if _t_paired not in redundant_abs_state_paired_str:
                        store_transitions.append(t)
                        redundant_abs_state_paired_str.append(str(_t_paired))

        for traj in synthetic_traj_list:
            samplebatch = traj[:]
            for _ in range(len(samplebatch)):
                t = samplebatch.slice(_, _ + 1)
                _t_paired = str(np.concatenate((traj.goal,
                                                np.array(traj.states[_]),
                                                np.array(traj.next_states[_]))))
                if _t_paired not in redundant_abs_state_paired_str:
                    store_transitions.append(t)
                    redundant_abs_state_paired_str.append(str(_t_paired))
        self.store_transition(store_transitions)

    def store_transition(self, transitions):
        if len(transitions) < 1:
            return
        samples = transitions[0]
        for i in range(1, len(transitions)):
            samples = samples.concat(transitions[i])
        self.replay_buffer.add.remote(samples)
        del transitions

    def sync_training_checkpoint(self):
        ''' Sync training level + Reset epsilon if training level update'''
        training_level: int = ray.get(self.shared_state.get_info.remote('training_level'))
        if self.training_level != training_level:
            self.training_level = training_level
            self.env.scheduler.set_training_level(self.training_level)
            RackState._cache.clear()

    def start(self, logging_interval=1):
        if self.log_path is not None and isinstance(self.log_path, fs.Path):
            writer = CsvWriter(str(self.log_path.joinpath(f'reanalyzer_{self._id}_log.csv')))
        step = 0
        while 1:
            traj_buff_len = ray.get(self.shared_state.get_info_len.remote('trajectory_list'))
            pop_buff_len = self.pop_buff_len
            if traj_buff_len > pop_buff_len:
                trajs = ray.get(self.shared_state.get_info_pop.remote('trajectory_list', pop_buff_len))
                if len(RackState._cache) > 1000:
                    RackState._cache.clear()
                if trajs is None or len(trajs) < 1:
                    time.sleep(.5)
                    continue
            else:
                time.sleep(.5)
                continue
            self.sync_training_checkpoint()
            st = time.time()
            for traj in trajs:
                # traj.action_dim = self.env.action_space_dim
                self.gen_bootstrapping_data(traj,
                                            horizon=min(max(self.training_level, 8), 8))
            step += 1
            et = time.time()
            tc = et - st
            if step % logging_interval == 0 and \
                    self.log_path is not None and isinstance(self.log_path, fs.Path):
                write_to_csv(writer, [
                    ('timestamp', get_time_stamp(), '%1s'),
                    ('time_consumption', tc, '%1s'),
                    ('episode_lens', len(trajs), '%1d'),
                    ('training_level', self.training_level, '%1d'),
                ])


if __name__ == '__main__':
    pass
