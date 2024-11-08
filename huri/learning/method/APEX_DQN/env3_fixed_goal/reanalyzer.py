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
from huri.learning.A_start_teacher.A_star_teacher import rm_ras_actions_fixedgoal
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
                 her_replay=False,
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

        # her replay
        self.her_replay = her_replay

    @staticmethod
    def extract_path_from_buffer(trainsition_buffer):
        goal_pattern = trainsition_buffer[0][0]
        return np.array([trainsition_buffer[0][1], *[i[-2] for i in trainsition_buffer]]), \
            [str(np.concatenate((goal_pattern, i[1], i[-2]))) for i in trainsition_buffer], goal_pattern

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
                               buffer,
                               horizon=50,
                               toggle_debug=False):
        redundant_abs_state, redundant_abs_state_paired_str, goal_pattern = self.extract_path_from_buffer(buffer)
        buffer[:] = []

        redundant_path = redundant_abs_state
        goal_state_np = goal_pattern

        # pydevd_pycharm.settrace('localhost', port=54321, stdoutToServer=True, stderrToServer=True)
        a = time.time()
        # is_finished = self.env.is_finished(redundant_path[-1], goal_state_np)
        # if self.env.is_goalpattern_fixed and not is_finished:
        #     return

        refined_path = rm_ras_actions_fixedgoal(redundant_path,
                                                h=max(horizon, 4),
                                                goal_pattern=goal_state_np,
                                                infeasible_dict={}, )
        b = time.time()
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
                    self._store_transition(t)
                    redundant_abs_state_paired_str.append(str(_t_paired))

            # if self.toggle_visual:
            #     # if _ % 50 == 0:
            #     print(
            #         f"Reanalyzer {self._id}: state level reset: {self.state_level}. "
            #         f"Average time for refinement: {b - a:.4f}. ")

    def _store_transition(self, transition):
        self.shared_mem.put(transition)

    def _update_level(self):
        if self.state_level != self.shared_state['state_level'] or self.class_level != self.shared_state['class_level']:
            self.state_level = self.shared_state['state_level']
            self.class_level = self.shared_state['class_level']
            return True
        else:
            return False

    def run(self):
        try:
            state_tc = 0
            refined_num = 0
            for _ in itertools.count():
                # pydevd_pycharm.settrace('localhost', port=54321, stdoutToServer=True, stderrToServer=True)
                state_level, class_level, replaybuffer_need_refined = self.shared_reanalyzer_mem.get()
                if self._update_level():
                    print(f"Reanalyzer {self._id}: state level reset: {self.state_level}. "
                          f"class level reset: {self.class_level} "
                          f"Average time for refinement: {state_tc / refined_num:.4f}. "
                          f"Number of refinement: {refined_num}")
                    state_tc = 0
                    refined_num = 0
                    RackState._cache.clear()
                if self.state_level != state_level or self.class_level != class_level:
                    # if self.toggle_visual:
                    #     print(state_level, class_level)
                    continue
                st = time.time()
                self.gen_bootstrapping_data(replaybuffer_need_refined,
                                            horizon=min(self.state_level, 8), )
                et = time.time()
                tc = et - st
                state_tc = state_tc + tc
                refined_num += 1
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
