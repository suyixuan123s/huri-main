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
from huri.learning.env.rack_v3.env import RackStatePlot, RackState, RackArrangementEnv, isdone
from huri.learning.A_start_teacher.A_star_teacher import rm_ras_actions_recur, rm_ras_actions_recur2, \
    rm_ras_actions_recur3_5, rm_ras_actions_recur3_6, a_star_solve, nx
from huri.learning.method.AlphaZero.log import CsvWriter, write_to_csv
from huri.learning.method.AlphaZero.utils import get_time_stamp
from huri.learning.method.APEX_DQN.distributed_fixed_goal.utils import (synthetic_traj_reflection_lrud,
                                                                        synthetic_traj_reflection_lr,
                                                                        synthetic_traj_category,
                                                                        synthetic_traj_direction,
                                                                        synthetic_traj_reflection_ud,
                                                                        padding,
                                                                        SampleBatch,
                                                                        Trajectory)
from huri.components.utils.img_utils import combine_images
import itertools


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
            [str(np.concatenate((np.array(traj.states[i]),
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
                               num_categories,
                               toggle_debug=False):
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
        reward = env._get_reward(is_finsihed, s_current, s_next, goal_pattern, toggle_debug=toggle_debug)
        traj.add_transition(state=s_current,
                            action=action,
                            reward=reward,
                            next_state=s_next,
                            done=is_finsihed, )
        if toggle_debug and False:
            print(f"s_current: \n{repr(s_current)}\n s_next: \n{repr(s_next)} \n "
                  f"action: {action} action_exp:{env.from_action(env.rack_size, action)} \n"
                  f"is finished: {is_finsihed} \n "
                  f"reward is {reward} ", )
            rsp = RackStatePlot(goal_pattern, )
            img = rsp.plot_states([s_current, s_next], row=13).get_img()
            cv2.imshow('debug', img)
            cv2.waitKey(0)
            print("-" * 30)

        if is_finsihed:
            # if toggle_debug:
            #     cv2.destroyWindow('debug')
            break
    return traj


def sync_data_first(refined_path_i, goal_pattern, env, toggle_debug=False):
    paths = []
    # ---
    for _i in np.arange(len(refined_path_i) - 2):
        state_current, state_next = refined_path_i[_i], refined_path_i[_i + 1]
        move_map = state_next - state_current
        move_from_idx = np.argwhere(move_map < 0)[0]
        move_to_idx = np.argwhere(move_map > 0)[0]
        state_current_intermediate = state_current.copy()
        state_current_intermediate[move_from_idx[0], move_from_idx[1]] = 0
        # check object index
        obj_id = state_current[move_from_idx[0], move_from_idx[1]]
        is_move_to_pattern = goal_pattern[move_to_idx[0], move_to_idx[1]] == state_next[move_to_idx[0], move_to_idx[1]]
        is_move_out_pattern = goal_pattern[move_from_idx[0], move_from_idx[1]] == state_current[
            move_from_idx[0], move_from_idx[1]]
        if is_move_to_pattern and not is_move_out_pattern:
            # find fillable place
            feasible_actions = RackState(state_current).feasible_action_set_detail
            ids = feasible_actions[np.all(feasible_actions[:, :2] == move_from_idx, axis=1)][:, 2:4]
            fillable_map = np.zeros(state_current.shape)
            fillable_map[ids[:, 0], ids[:, 1]] = 1
            fillable_map[(goal_pattern == obj_id)] = 0
            for _ in np.argwhere(fillable_map):
                s = state_current_intermediate.copy()
                s[_[0], _[1]] = obj_id
                # TODO: it seems this is not necessary
                # feasible_action_set = RackState(s).feasible_action_set
                # action_between = env.action_between_states_constraint_free(s, state_next)
                # if action_between not in feasible_action_set:
                #     continue
                # ---------------------
                paths.append([s, state_next])
        # -----------------------------------------------------------------------------
    if toggle_debug:
        image_list = []
        for _ in [(state_current, state_next)] + paths:
            rsp = RackStatePlot(goal_pattern, )
            try:
                img = rsp.plot_states(_, row=2).get_img()
            except:
                print(f">>> Reanalyzer infeasible START: {_}")
            image_list.append(img)
        img = combine_images(image_list, columns=10)
        cv2.imshow('added synthetic data', img)
        cv2.waitKey(0)
    return paths


def sync_data_end(refined_path_i, goal_pattern, env, toggle_debug=False):
    paths = []
    # ---
    state_current, state_next = refined_path_i[-2], refined_path_i[-1]
    if env.is_finished(state_next, goal_pattern):
        move_map = state_next - state_current
        move_from_idx = np.argwhere(move_map < 0)[0]
        move_to_idx = np.argwhere(move_map > 0)[0]
        state_current_intermediate = state_current.copy()
        state_next_intermediate = state_next.copy()
        state_current_intermediate[move_from_idx[0], move_from_idx[1]] = 0
        state_next_intermediate[move_to_idx[0], move_to_idx[1]] = 0
        # check object index
        obj_id = state_current[move_from_idx[0], move_from_idx[1]]
        is_move_to_pattern = goal_pattern[move_to_idx[0], move_to_idx[1]] == state_next[move_to_idx[0], move_to_idx[1]]
        is_move_out_pattern = goal_pattern[move_from_idx[0], move_from_idx[1]] == state_current[
            move_from_idx[0], move_from_idx[1]]
        if is_move_to_pattern and not is_move_out_pattern:
            # find fillable place
            path_tmp_1 = []
            feasible_actions = RackState(state_current).feasible_action_set_detail
            ids = feasible_actions[np.all(feasible_actions[:, :2] == move_from_idx, axis=1)][:, 2:4]
            fillable_map = np.zeros(state_current.shape)
            fillable_map[ids[:, 0], ids[:, 1]] = 1
            fillable_map[(goal_pattern == obj_id)] = 0
            for _ in np.argwhere(fillable_map):
                s = state_current_intermediate.copy()
                s[_[0], _[1]] = obj_id
                # TODO: it seems this is not necessary
                # feasible_action_set = RackState(s).feasible_action_set
                # action_between = env.action_between_states_constraint_free(s, state_next)
                # if action_between not in feasible_action_set:
                #     continue
                # ---------------------
                paths.append([s, state_next])
                path_tmp_1.append(s)
            path_tmp_2 = []
            feasible_actions = RackState(state_next).feasible_action_set_detail
            ids = feasible_actions[np.all(feasible_actions[:, :2] == move_to_idx, axis=1)][:, 2:4]
            fillable_map = np.zeros(state_current.shape)
            fillable_map[ids[:, 0], ids[:, 1]] = 1
            fillable_map[(goal_pattern != obj_id)] = 0
            for _ in np.argwhere(fillable_map):
                s = state_next_intermediate.copy()
                s[_[0], _[1]] = obj_id
                path_tmp_2.append(s)
            if len(path_tmp_1) > 0 and len(path_tmp_2) > 0:
                paths += list(itertools.product(path_tmp_1, path_tmp_2))

        # -----------------------------------------------------------------------------
    if toggle_debug:
        image_list = []
        for _x in [(state_current, state_next)] + paths:
            rsp = RackStatePlot(goal_pattern, )
            try:
                img = rsp.plot_states(_x, row=2).get_img()
            except:
                print(f">>> Reanalyzer infeasible END: {_x}")
            image_list.append(img)
        img = combine_images(image_list, columns=10)
        cv2.imshow('added synthetic data', img)
        cv2.waitKey(0)
    return paths


def sync_data(refined_path_i, goal_pattern, env: RackArrangementEnv, toggle_debug=False):
    if not env.is_finished(refined_path_i[-1], goal_pattern):
        return []
    for _ in refined_path_i:
        print(" ")


@ray.remote(num_cpus=1)
class Reanalyzer(object):
    def __init__(self,
                 uid: int,
                 env: RackArrangementEnv,
                 shared_state: ray.actor.ActorClass,
                 shared_traj_buffer: ray.actor.ActorClass,
                 replay_buffer: ray.actor.ActorClass,
                 pop_buff_len: int = 10,
                 toggle_visual=False,
                 toggle_log_debug=False,
                 toggle_completer=True,
                 toggle_refiner=True,
                 log_path=None, ):
        self._id = uid
        self.env = env
        # set up replay buffer size
        self.replay_buffer = replay_buffer
        self.shared_state = shared_state
        self.shared_traj_buffer = shared_traj_buffer
        # is-debug
        self.toggle_visual = toggle_visual
        self.log_path = fs.Path(log_path) if log_path is not None else None
        # training level
        self.training_level = 1
        self.pop_buff_len = pop_buff_len
        self.data_efficiency = 0
        self.toggle_log_debug = toggle_log_debug
        self.toggle_completer = toggle_completer
        self.toggle_refiner = toggle_refiner
        # calculation
        self.num_reuse = 0
        self.num_need_reuse = 0
        self.num_reduced = 0
        self.num_reduced_added = 0
        self.num_total_data = 0
        self.num_total_data_len = 0

        self.t_A_rescuer = 0
        self.t_A_trimmer = 0
        print(f"Reanalyzer: {self._id}: toggle_complete: {toggle_completer} "
              f"| toogle refiner: {toggle_refiner}")

    def gen_bootstrapping_data(self,
                               traj,
                               horizon=50,
                               toggle_debug=False):
        self.num_total_data += 1
        store_transitions = []
        redundant_path, redundant_abs_state_paired_str, goal_pattern = extract_path_from_traj(traj)
        self.num_total_data_len += len(redundant_path)
        action_dim = traj.action_dim
        n_step = traj.n_step
        gamma = traj.gamma
        num_categories = traj.num_categories
        del traj
        goal_state_np = goal_pattern
        is_goal_reached = False
        A_star_path = None
        if not self.env.is_state_valid(redundant_path[-1]):
            return
        if self.toggle_completer:
            # ------------------------------------
            if not isdone(redundant_path[-1], goal_state_np):
                self.num_need_reuse += 1
                # find the path to goal
                ts = time.time()
                path = a_star_solve(redundant_path[-1], goal_state_np, max_iter_cnt=50)
                te = time.time()
                self.t_A_rescuer += te - ts
                if self.toggle_log_debug:
                    print(f"time consumption is {te - ts}")
                if len(path) > 0:
                    A_star_path = path
                    redundant_path = np.concatenate((redundant_path, path[1:]))
                    is_goal_reached = True
                    self.num_reuse += 1
                    # redundant_abs_state_paired_str = [  # update the redundant_abs_state_paired_str
                    #     str(np.concatenate((goal_state_np, path[i], path[i + 1])))
                    #     for i in range(len(path) - 1)]
                    if self.toggle_visual:
                        rsp = RackStatePlot(goal_state_np, )
                        img = rsp.plot_states(redundant_path, row=10).get_img()
                        cv2.imshow('complete the goal', img)
                        cv2.waitKey(0)
                else:
                    if self.toggle_visual:
                        print("faied to try generate the path to goal")
                    # return
            else:
                if self.toggle_visual:
                    rsp = RackStatePlot(goal_state_np, )
                    # img = rsp.plot_states(redundant_path, row=13).get_img()
                    # cv2.imshow('the path', img)
                    # cv2.waitKey(0)
                    print(len(redundant_path))
                is_goal_reached = True
            # ---------------------
        else:
            if not isdone(redundant_path[-1], goal_state_np):
                self.num_need_reuse += 1
        if self.toggle_refiner:
            ts = time.time()
            refined_path = rm_ras_actions_recur3_6(redundant_path,
                                                   h=13,
                                                   goal_pattern=goal_state_np,
                                                   max_refine_num=100,
                                                   infeasible_dict={}, )
            te = time.time()
            self.t_A_trimmer += te - ts
        else:
            if A_star_path is None:
                return
            refined_path = [A_star_path]

        # if not np.array_equal(redundant_path[-1], goal_state_np):
        #     goal_state_np = redundant_path[-1]
        is_end_calculated = False
        self.num_reduced += len(refined_path[0]) - len(redundant_path)

        for i in range(len(refined_path)):
            if len(refined_path[i]) >= len(redundant_path) - i:
                continue
            if len(refined_path[i]) < 2:
                continue
            if self.toggle_visual:
                rsp = RackStatePlot(goal_state_np, )
                img = rsp.plot_states(refined_path[i], img_scale=1.5).get_img()
                cv2.imshow(f"refined_path", img)
                cv2.waitKey(0)
                imgs_list = []
            paths = [refined_path[i]]
            # sync_data(refined_path[i], goal_pattern, env)
            # if is_goal_reached and not is_end_calculated:
            #     if len(refined_path[i]) > 2:
            #         paths += sync_data_first(refined_path[i], goal_state_np, self.env, toggle_debug=self.toggle_visual)
            #     paths += sync_data_end(refined_path[i][-2:], goal_state_np, self.env, toggle_debug=self.toggle_visual)
            #     #     # if not is_end_calculated:
            #     #     #     time.time()
            #     #     paths += sync_data_end(refined_path[i], goal_state_np, self.env, toggle_debug=self.toggle_visual)
            #     is_end_calculated = True
            for p in paths:
                refined_traj = refined_path_to_traj_recur(self.env,
                                                          p,
                                                          action_dim=action_dim,
                                                          n_step=n_step,
                                                          gamma=gamma,
                                                          goal_pattern=goal_state_np,
                                                          num_categories=num_categories,
                                                          toggle_debug=self.toggle_visual)
                if len(refined_traj) > 0 and toggle_debug:
                    print(f"Add refined data into replay buffer: {len(refined_traj) + 1}/{len(redundant_path)}")
                samplebatch = refined_traj[:]
                for _ in range(len(samplebatch)):
                    t = samplebatch.slice(_, _ + 1)
                    _t_paired = str(np.concatenate((np.array(refined_traj.states[_]),
                                                    np.array(refined_traj.next_states[_]))))
                    if _t_paired not in redundant_abs_state_paired_str:
                        if self.toggle_visual:
                            redundant_abs_state_paired_str.append(str(_t_paired))
                            rsp = RackStatePlot(goal_state_np, )
                            img = rsp.plot_states([t['state'][0], t['next_state'][0]], img_scale=1.5).get_img()
                            # print("-" * 30)
                            # print(
                            #     f"Added reward: {t['reward']} | action: {t['action']} | Next: {t['next_state_feasible_action']}")
                            imgs_list.append(img)
                            # cv2.imshow('the added batch', img)
                            # cv2.waitKey(0)
                        store_transitions.append(t)
                        redundant_abs_state_paired_str.append(str(_t_paired))
            if self.toggle_visual:
                rsp = RackStatePlot(goal_state_np, )
                img = rsp.plot_states(refined_path[i], ).get_img()
                # cv2.imshow(f"refined_states", img)
                if len(imgs_list) > 0:
                    cv2.imshow(f"added", combine_images(imgs_list))
                    cv2.waitKey(0)
                    cv2.destroyWindow("added")
        if self.toggle_visual:
            cv2.destroyAllWindows()
        self.store_transition(store_transitions)

    def store_transition(self, transitions):
        self.num_reduced_added += len(transitions)
        if len(transitions) < 1:
            return
        samples = transitions[0]
        for i in range(1, len(transitions)):
            samples = samples.concat(transitions[i])
        self.replay_buffer.add.remote(samples)
        self.data_efficiency += len(samples)
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
            traj_buff_len = ray.get(self.shared_traj_buffer.get_info_len.remote('trajectory_list'))
            pop_buff_len = self.pop_buff_len
            if traj_buff_len > pop_buff_len:
                trajs = ray.get(self.shared_traj_buffer.get_info_pop.remote('trajectory_list', pop_buff_len))
                if len(RackState._cache) > 1000:
                    RackState._cache.clear()
                if trajs is None or len(trajs) < 1:
                    time.sleep(.5)
                    continue
            else:
                time.sleep(1)
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
            if self.toggle_log_debug:
                print(f"Time consumption is: {tc}")
            if step % logging_interval == 0 and \
                    self.log_path is not None and isinstance(self.log_path, fs.Path):
                write_to_csv(writer, [
                    ('timestamp', get_time_stamp(), '%1s'),
                    ('time_consumption', tc, '%1s'),
                    ('episode_lens', len(trajs), '%1d'),
                    ('training_level', self.training_level, '%1d'),
                    ('data efficiency', self.data_efficiency / tc, '%1d'),
                    ('num_reuse', self.num_reuse, '%1d'),
                    ('num_reduced', self.num_reduced, '%1d'),
                    ('num_reduced_added', self.num_reduced_added, '%1d'),
                    ('num_total_data', self.num_total_data, '%1d'),
                    ('num_total_data_len', self.num_total_data_len, '%1d'),
                    ('num_need_reuse', self.num_need_reuse, '%1d'),
                    ('t_A_rescuer', self.t_A_rescuer, '%1s'),
                    ('t_A_trimmer', self.t_A_trimmer, '%1s'),
                ])

            self.t_A_trimmer = 0
            self.t_A_rescuer = 0
            self.data_efficiency = 0


# num_reuse
# num_reduced
# num_reduced_added
# num_total_data

if __name__ == '__main__':
    pass
