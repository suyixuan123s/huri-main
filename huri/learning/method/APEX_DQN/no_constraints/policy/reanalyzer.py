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
from huri.learning.A_start_teacher.A_star_teacher import rm_ras_actions_recur, rm_ras_actions_recur2, \
    rm_ras_actions_recur3


class Reanalyzer(mp.Process):
    def __init__(self,
                 uid: int,
                 env: Gym_Proto,
                 shared_state,
                 replay_buffer,
                 toggle_her_replay=True,
                 toggle_visual=False,
                 logger=None):
        super(Reanalyzer, self).__init__()
        self._id = uid
        self.env = env
        # set up replay buffer size
        self.replay_buffer = replay_buffer
        self.shared_state = shared_state
        # her replay
        self.toggle_her_replay = toggle_her_replay
        # is-debug
        self.toggle_visual = toggle_visual
        self.logger = logger
        # curriculum learning
        self.env.scheduler.state_level = self.state_level = self.shared_state['state_level']
        self.env.scheduler.class_level = self.class_level = self.shared_state['class_level']

    @staticmethod
    def extract_path_from_traj(traj):
        goal_pattern = traj.goal.state
        return np.array([traj[0][0], *[i[-2].state for i in traj]]), \
            [str(np.concatenate((goal_pattern, i[0], i[-2]))) for i in traj], goal_pattern

    def refined_path_to_transitions_recur(self, refined_path, goal_pattern):
        # if not np.all(refined_path[-1] == goal_pattern):
        #     print("ERROR ")
        env = self.env
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

    def gen_bootstrapping_data(self,
                               traj,
                               horizon=50,
                               toggle_debug=False):
        redundant_abs_state, redundant_abs_state_paired_str, goal_pattern = self.extract_path_from_traj(traj)
        del traj

        redundant_path = redundant_abs_state
        goal_state_np = goal_pattern

        # pydevd_pycharm.settrace('localhost', port=54321, stdoutToServer=True, stderrToServer=True)
        a = time.time()
        refined_path, refined_path_her = rm_ras_actions_recur3(redundant_path,
                                                               h=horizon,
                                                               goal_pattern=redundant_path[-1],
                                                               infeasible_dict={}, )
        b = time.time()
        if np.array_equal(redundant_path[-1], goal_state_np):
            for i in range(len(refined_path)):
                if len(refined_path[i]) >= len(redundant_path) - i:
                    continue
                if len(refined_path[i]) < 2:
                    continue
                if self.toggle_visual:
                    rsp = RackStatePlot(goal_state_np, )
                    img = rsp.plot_states(refined_path[i]).get_img()
                    cv2.imshow(f"plot_{self._id}", img)
                    cv2.waitKey(20)
                refined_transitions = self.refined_path_to_transitions_recur(refined_path[i], goal_state_np)
                if len(refined_transitions) > 0 and toggle_debug:
                    print(f"Add refined data into replay buffer: {len(refined_transitions) + 1}/{len(redundant_path)}")
                for t in refined_transitions:
                    _t_paired = np.concatenate((t[0], t[1], t[-2]))
                    if str(_t_paired) not in redundant_abs_state_paired_str:
                        self.store_transition(t)
                        redundant_abs_state_paired_str.append(str(_t_paired))
        else:
            # print("!?><")
            goal_state_np = redundant_path[-1]
            for i in range(len(refined_path)):
                if len(refined_path[i]) >= len(redundant_path) - i:
                    continue
                if len(refined_path[i]) < 2:
                    continue
                if self.toggle_visual:
                    rsp = RackStatePlot(goal_state_np, )
                    img = rsp.plot_states(refined_path[i]).get_img()
                    cv2.imshow(f"goal_changed_plot_{self._id}", img)
                    cv2.waitKey(20)
                refined_transitions = self.refined_path_to_transitions_recur(refined_path[i], goal_state_np)
                if len(refined_transitions) > 0 and toggle_debug:
                    print(f"Add refined data into replay buffer: {len(refined_transitions) + 1}/{len(redundant_path)}")
                for t in refined_transitions:
                    _t_paired = np.concatenate((t[0], t[1], t[-2]))
                    if str(_t_paired) not in redundant_abs_state_paired_str:
                        self.store_transition(t)
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
                refined_transitions = self.refined_path_to_transitions_recur(refined_path_her[i], goal_state_tmp_np)
                if self.toggle_visual:
                    rsp = RackStatePlot(goal_state_tmp_np, )
                    img = rsp.plot_states(refined_path_her[i]).get_img()
                    cv2.imshow(f"goal_her_plot_{self._id}", img)
                    cv2.waitKey(20)
                for t in refined_transitions:
                    _t_paired = np.concatenate((t[1], t[-2]))
                    if str(_t_paired) not in redundant_abs_state_paired_str:
                        self.store_transition(t)
                        redundant_abs_state_paired_str.append(str(_t_paired))

            if self.toggle_visual:
                # if _ % 50 == 0:
                print(
                    f"Reanalyzer {self._id}: state level reset: {self.state_level}. "
                    f"Average time for refinement: {b - a:.4f}. ")

    def store_transition(self, transition):
        goal, state, action, reward, next_state, done = transition
        self._store_transition(goal=goal, state=state, action=action, reward=reward, next_state=next_state, done=done)
        # for (goal, state, action, reward, next_state, done) in self.env.synthesize_reflection_data(goal=goal,
        #                                                                                            state=state,
        #                                                                                            reward=reward,
        #                                                                                            next_state=next_state,
        #                                                                                            done=done):
        #     self._store_transition(goal=goal, state=state, action=action, reward=reward, next_state=next_state,
        #                            done=done)
        del transition

    def _store_transition(self, goal, state, action, reward, next_state, done):
        self.replay_buffer.store(goal=goal,
                                 obs=state,
                                 c_abs_obs=RackState(state).compressed_abs_state,
                                 act=action,
                                 rew=reward,
                                 next_obs=next_state,
                                 c_abs_next_obs=RackState(next_state).compressed_abs_state,
                                 done=done, )

    def _update_level(self):
        if self.state_level != self.shared_state['state_level'] or self.class_level != self.shared_state['class_level']:
            self.state_level = self.shared_state['state_level']
            self.class_level = self.shared_state['class_level']
            return True
        else:
            return False

    def run(self):
        # try:
        state_tc = 0
        refined_num = 0
        for _ in itertools.count():
            # pydevd_pycharm.settrace('localhost', port=54321, stdoutToServer=True, stderrToServer=True)
            # state_level, class_level, replaybuffer_need_refined = self.shared_reanalyzer_mem.get()
            # if self._update_level():
            #     print(f"Reanalyzer {self._id}: state level reset: {self.state_level}. "
            #           f"class level reset: {self.class_level} "
            #           f"Average time for refinement: {state_tc / refined_num:.4f}. "
            #           f"Number of refinement: {refined_num}")
            #     state_tc = 0
            #     refined_num = 0
            #     RackState._cache.clear()
            # if self.state_level != state_level or self.class_level != class_level:
            #     if self.toggle_visual:
            #         print(state_level, class_level)
            #     continue
            st = time.time()
            if len(self.shared_state['trajectory_list']) < 1:
                continue
            traj = self.shared_state['trajectory_list'].pop()
            self.gen_bootstrapping_data(traj,
                                        horizon=min(self.state_level, 8), )
            et = time.time()
            tc = et - st
            state_tc = state_tc + tc
            refined_num += 1
            if self.logger is not None:
                self.logger.log({"reanalyzer_time_consumption": tc})
        # except Exception as e:
        #     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR")
        #     print(f"{e}")
