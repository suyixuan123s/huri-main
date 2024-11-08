import pickle
import time

import cv2
import torch
import torch.multiprocessing as mp
from huri.learning.env.env_meta import Gym_Proto
import copy
import numpy as np
import itertools
from typing import Tuple
from huri.learning.env.rack_v3.env import RackStatePlot, RackState
from huri.learning.A_start_teacher.A_star_teacher import rm_ras_actions_recur, rm_ras_actions_recur2
from huri.examples.task_planning.a_star import TubePuzzle
from huri.components.utils.img_utils import combine_images
import huri.core.file_sys as fs
import pydevd_pycharm


class Reanalyzer(mp.Process):
    def __init__(self,
                 uid: int,
                 env: Gym_Proto,
                 shared_state,
                 shared_reanalyzer_mem,
                 shared_mem,
                 is_astar_agent=False,
                 toggle_visual=False):
        super(Reanalyzer, self).__init__()
        self._id = uid

        self.env = env.copy()
        # set up replay buffer size
        self.shared_mem = shared_mem
        self.shared_state = shared_state
        self.shared_reanalyzer_mem = shared_reanalyzer_mem

        # is-debug
        self.toggle_visual = toggle_visual

        # curriculum learning
        self.env.scheduler.state_level = self.state_level = self.shared_state['state_level']
        self.env.scheduler.class_level = self.class_level = self.shared_state['class_level']

        self.is_astar_agent = is_astar_agent

    def refined_path_to_transitions_recur(self, refined_path, goal_pattern):
        # if not np.all(refined_path[-1] == goal_pattern):
        #     print("ERROR ")
        env = self.env
        transitions = []
        for _i in np.arange(len(refined_path) - 1):
            s_current, s_next = refined_path[_i], refined_path[_i + 1]
            action = env.action_between_states(s_current=s_current, s_next=s_next)
            is_finsihed = env.is_finished(s_next, goal_pattern)
            # check if it is a unfinished state
            # if _i == len(refined_path) - 2 and not is_finsihed:
            #     reward = -50
            #     print("!!!!!!!!!!", s_current, s_next)
            # else:
            reward = env._get_reward(is_finsihed, s_current, s_next, s_current)
            tran_tmp = [goal_pattern,
                        s_current,
                        action,
                        reward,
                        s_next,
                        is_finsihed]
            transitions.append(tran_tmp)
            if is_finsihed:
                break
        return transitions

    def refine_path_to_transitions(self, redundant_path, goal_pattern, horizon=8, infeasible_dict=None):
        if infeasible_dict is None:
            infeasible_dict = {}
        refined_path = rm_ras_actions_recur2(redundant_path,
                                             h=horizon,
                                             goal_pattern=goal_pattern,
                                             infeasible_dict=infeasible_dict, )
        if not np.all(refined_path[-1] == goal_pattern):
            print("ERROR <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        if len(refined_path) == len(redundant_path):
            return []
        env = self.env
        transitions = []
        for _i in np.arange(len(refined_path) - 1):
            s_current, s_next = refined_path[_i], refined_path[_i + 1]
            action = env.action_between_states(s_current=s_current, s_next=s_next)
            is_finsihed = env.is_finished(s_next, goal_pattern)
            # check if it is a unfinished state
            # if _i == len(refined_path) - 2 and not is_finsihed:
            #     reward = -50
            #     print("!!!!!!!!!!", s_current, s_next)
            # else:
            reward = env._get_reward(is_finsihed, s_current, s_next, s_current)
            tran_tmp = [goal_pattern,
                        s_current,
                        action,
                        reward,
                        s_next,
                        is_finsihed]
            transitions.append(tran_tmp)
            if is_finsihed:
                break
        return transitions

    def extract_path_from_buffer(self, trainsition_buffer):
        return np.array([trainsition_buffer[0][1], *[i[-2] for i in trainsition_buffer]]), \
            [np.concatenate((i[1], i[-2])) for i in trainsition_buffer], trainsition_buffer[0][0]

    def gen_bootstrapping_data(self, buffer, horizon=8, a_star_teacher_toggle=False, her_relabel_toggle=False,
                               toggle_debug=False):
        redundant_abs_state, redundant_abs_state_paired, goal_pattern = self.extract_path_from_buffer(buffer)
        buffer[:] = []

        _redundant_abs_state_paired_str = [str(_) for _ in redundant_abs_state_paired]
        redundant_abs_state_paired_str = copy.deepcopy(_redundant_abs_state_paired_str)

        redundant_path = redundant_abs_state
        goal_state_np = goal_pattern
        if a_star_teacher_toggle:
            refined_path = rm_ras_actions_recur(redundant_path,
                                                h=horizon,
                                                goal_pattern=redundant_path[-1],
                                                infeasible_dict={}, )
            for i in range(len(refined_path)):
                if len(refined_path[i]) >= len(redundant_path) - i:
                    continue
                if len(refined_path[i]) < 2:
                    continue
                refined_transitions = self.refined_path_to_transitions_recur(refined_path[i], goal_state_np)
                if len(refined_transitions) > 0 and toggle_debug:
                    print(f"Add refined data into replay buffer: {len(refined_transitions) + 1}/{len(redundant_path)}")
                for t in refined_transitions:
                    _t_paired = np.concatenate((t[1], t[-2]))
                    if str(_t_paired) not in redundant_abs_state_paired_str:
                        self._store_transition(t)
                        redundant_abs_state_paired_str.append(str(_t_paired))

        if her_relabel_toggle:
            self.her_relabel(redundant_path,
                             path_pair=copy.deepcopy(_redundant_abs_state_paired_str),
                             horizon=horizon)

    def her_relabel(self, redundant_path, path_pair, max_relabel_goal_num=80, horizon=1, toggle_debug=False):
        result_list = []
        if len(redundant_path) > 1:
            # pydevd_pycharm.settrace('localhost', port=5675, stdoutToServer=True, stderrToServer=True)
            a = time.time()
            goal_select_ids = np.arange(1,len(redundant_path) - 1)
            goal_pattern_set = redundant_path[goal_select_ids]
            refined_path = rm_ras_actions_recur2(redundant_path,
                                                h=horizon,
                                                goal_pattern=redundant_path[-1],
                                                infeasible_dict={}, )
            for i in range(len(refined_path)):
                if len(refined_path[i]) >= len(redundant_path) - i:
                    continue
                if len(refined_path[i]) < 2:
                    continue
                goal_state_np = goal_pattern_set[i]
                refined_transitions = self.refined_path_to_transitions_recur(refined_path[i], goal_state_np)
                for t in refined_transitions:
                    _t_paired = np.concatenate((t[1], t[-2]))
                    if str(_t_paired) not in path_pair:
                        self._store_transition(t)
                        path_pair.append(str(_t_paired))

                if toggle_debug and len(refined_transitions) > 0:
                    result_list.append(refined_transitions)

            b = time.time()
            # print("Time consuming is :", b - a)
        if toggle_debug:
            img_list = []
            for _ in result_list:
                goal_pattern = _[0][0]
                path = [_[0][1], *[i[-2] for i in _]]
                rsp = RackStatePlot(goal_pattern, )
                plot = [rsp.plot_states([_]).get_img() for _ in path]
                img_list.extend(plot)
            cv2.imshow("debug_her_label", combine_images(img_list, columns=15))
            cv2.waitKey(0)

    def _store_transition(self, transition):
        self.shared_mem.put(transition)

    def _update_state_level(self):
        if self.state_level != self.shared_state['state_level']:
            self.state_level = self.shared_state['state_level']
            return True
        else:
            return False

    def run(self):
        try:
            is_astar_agent = self.is_astar_agent
            state_tc = 0
            refined_num = 0
            for _ in itertools.count():
                # pydevd_pycharm.settrace('localhost', port=54321, stdoutToServer=True, stderrToServer=True)
                state_level, replaybuffer_need_refined = self.shared_reanalyzer_mem.get()
                if self.state_level != state_level:
                    continue
                if self._update_state_level():
                    print(f"Reanalyzer {self._id}: state level reset: {self.state_level}. "
                          f"Average time for refinement: {state_tc / refined_num:.4f}. "
                          f"Number of refinement: {refined_num}")
                    state_tc = 0
                    refined_num = 0
                st = time.time()
                self.gen_bootstrapping_data(replaybuffer_need_refined,
                                            a_star_teacher_toggle=is_astar_agent,
                                            horizon=min(self.state_level, 8),
                                            her_relabel_toggle=not is_astar_agent)
                et = time.time()
                tc = et - st
                state_tc = state_tc + tc
                refined_num += 1
                # if _ % 50 == 0:
                #     print(
                #         f"Reanalyzer {self._id}: is_astar_agent:{is_astar_agent} state level reset: {self.state_level}. "
                #         f"Average time for refinement: {state_tc / refined_num:.4f}. "
                #         f"Number of refinement: {refined_num}")
        except Exception as e:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR")
            print(f"{e}")


if __name__ == "__main__":
    from huri.learning.env.rack_v3.env import RackArrangementEnv, GOAL_PATTERN_5x10
    from huri.learning.network.d3qn_attention import DuelingDQN

    num_tube_classes = 5
    rack_size = (5, 10)
    action_space_dim = np.prod(rack_size) ** 2
    observation_space_dim = (1, 2 * rack_size[0], rack_size[1])
    observation_space_dim_nn = (1, *rack_size)
    env = RackArrangementEnv(rack_size=rack_size,
                             num_classes=num_tube_classes,
                             observation_space_dim=observation_space_dim,
                             action_space_dim=action_space_dim,
                             is_curriculum_lr=True,
                             is_goalpattern_fixed=False,
                             seed=777)
    # env.set_goal_pattern(GOAL_PATTERN_5x10)

    net = DuelingDQN(obs_dim=observation_space_dim_nn, action_dim=action_space_dim)

    import torch.multiprocessing as mp

    mp_manager = mp.Manager()
    shared_state = mp_manager.dict()
    shared_mem = mp_manager.Queue()
    shared_state['state_level'] = env.scheduler.state_level
    shared_state['state_level'] = 1
    shared_state['class_level'] = 4

    actor = HERActor(actor_id=1, net=net, env=env,
                     shared_state=shared_state,
                     batch_size=32, max_epsilon=1, min_epsilon=.1, epsilon_decay=1e-5,
                     target_update_freq=200,
                     reset_num=100, shared_net=net, shared_mem=shared_mem, device='cuda')
    actor.run()
    print("Main: replay_mem.size:", shared_mem.qsize())
    # while 1:
    #     time.sleep(1)
    # for i in range(shared_mem.qsize()):
    #     xp_batch = shared_mem.get()
    #     print("priority:", xp_batch)
