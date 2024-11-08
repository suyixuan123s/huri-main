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
from huri.learning.env.rack_v3.env2 import RackStatePlot, RackState, RackArrangementEnv
from huri.learning.A_start_teacher.A_star_teacher import rm_ras_actions_recur, rm_ras_actions_recur2, \
    rm_ras_actions_recur3
from huri.learning.method.AlphaZero.log import CsvWriter, write_to_csv
from huri.learning.method.AlphaZero.utils import get_time_stamp
from ray.rllib.policy.sample_batch import SampleBatch
from huri.learning.method.APEX_DQN.distributed_2.pipeline import padding


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
    return (np.array([traj[0]['state'].squeeze(0), *[np.array(i['next_state'].squeeze(0)) for i in traj]]),
            [str(np.concatenate((goal_pattern, i['state'].squeeze(0), i['next_state'].squeeze(0)))) for i in traj],
            goal_pattern)


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
                    'next_state_feasible_action': padding(int(env.action_space_dim ** 2 / 4) + 1,
                                                          RackState(s_next).feasible_action_set),
                    'done': is_finsihed,
                    'goal': goal_pattern}
        transitions.append(tran_tmp)
        if is_finsihed:
            break
    return transitions


@ray.remote
class Reanalyzer(object):
    def __init__(self,
                 uid: int,
                 env: RackArrangementEnv,
                 shared_state: ray.actor.ActorClass,
                 replay_buffer: ray.actor.ActorClass,
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

    def gen_bootstrapping_data(self,
                               traj,
                               horizon=50,
                               toggle_debug=False):
        store_transitions = []
        redundant_path, redundant_abs_state_paired_str, goal_pattern = extract_path_from_traj(traj)

        del traj
        goal_state_np = goal_pattern

        refined_path, refined_path_her = rm_ras_actions_recur3(redundant_path,
                                                               h=horizon,
                                                               goal_pattern=redundant_path[-1],
                                                               infeasible_dict={}, )

        if not np.array_equal(redundant_path[-1], goal_state_np):
            goal_state_np = redundant_path[-1]

        for i in range(len(refined_path)):
            if len(refined_path[i]) >= len(redundant_path) - i:
                continue
            if len(refined_path[i]) < 2:
                continue
            if self.toggle_visual:
                rsp = RackStatePlot(goal_state_np, )
                img = rsp.plot_states(refined_path[i], ).get_img()
                cv2.imshow(f"plot_{self._id}", img)
                cv2.waitKey(20)
            refined_transitions = refined_path_to_transitions_recur(self.env, refined_path[i], goal_state_np)
            if len(refined_transitions) > 0 and toggle_debug:
                print(f"Add refined data into replay buffer: {len(refined_transitions) + 1}/{len(redundant_path)}")
            for t in refined_transitions:
                _t_paired = np.concatenate((t['goal'], t['state'], t['next_state']))
                if str(_t_paired) not in redundant_abs_state_paired_str:
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
                refined_transitions = refined_path_to_transitions_recur(self.env, refined_path_her[i],
                                                                        goal_state_tmp_np)
                if self.toggle_visual:
                    rsp = RackStatePlot(goal_state_tmp_np, )
                    img = rsp.plot_states(refined_path_her[i]).get_img()
                    cv2.imshow(f"goal_her_plot_{self._id}", img)
                    cv2.waitKey(20)
                for t in refined_transitions:
                    _t_paired = np.concatenate((t['goal'], t['state'], t['next_state']))
                    if str(_t_paired) not in redundant_abs_state_paired_str:
                        store_transitions.append(t)
                        redundant_abs_state_paired_str.append(str(_t_paired))

            self.store_transition(store_transitions)

    def store_transition(self, transitions):
        transitions = SampleBatch(merge_dictionaries(transitions))
        self.replay_buffer.add.remote(transitions)
        del transitions

    def sync_training_checkpoint(self):
        ''' Sync training level + Reset epsilon if training level update'''
        training_level: int = ray.get(self.shared_state.get_info.remote('training_level'))
        if self.training_level != training_level:
            self.training_level = training_level
            self.env.scheduler.set_training_level(self.training_level)

    def start(self):
        if self.log_path is not None and isinstance(self.log_path, fs.Path):
            writer = CsvWriter(str(self.log_path.joinpath(f'reanalyzer_{self._id}_log.csv')))
        while 1:
            traj_buff_len = ray.get(self.shared_state.get_info_len.remote('trajectory_list'))
            if traj_buff_len > 0:
                traj = ray.get(self.shared_state.get_info_pop.remote('trajectory_list'))
                if traj is None:
                    continue
                traj.action_dim = self.env.action_space_dim
            else:
                time.sleep(.5)
                continue
            self.sync_training_checkpoint()
            st = time.time()
            self.gen_bootstrapping_data(traj,
                                        horizon=min(self.training_level, 8), )
            et = time.time()
            tc = et - st
            if self.log_path is not None and isinstance(self.log_path, fs.Path):
                write_to_csv(writer, [
                    ('timestamp', get_time_stamp(), '%1s'),
                    ('time_consumption', tc, '%1s'),
                    ('episode_lens', len(traj), '%1d'),
                    ('training_level', self.training_level, '%1d'),
                ])


if __name__ == '__main__':
    pass
