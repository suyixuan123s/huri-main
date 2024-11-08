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
from huri.learning.A_start_teacher.A_star_teacher import rm_ras_actions_recur, rm_ras_actions
from huri.examples.task_planning.a_star import TubePuzzle
from huri.components.utils.img_utils import combine_images
import huri.core.file_sys as fs


class Actor(mp.Process):
    def __init__(self,
                 actor_id: int,
                 net,
                 env: Gym_Proto,
                 max_epsilon,
                 min_epsilon,
                 epsilon_decay,
                 reset_num,
                 target_update_freq,
                 shared_net,
                 shared_state,
                 shared_mem,
                 device,
                 toggle_visual=False):
        super(Actor, self).__init__()
        self._actor_id = actor_id

        self.env = env.copy()
        self.env_abs_state_slice = self.env.abs_state_slice
        self.env_abs_goal_slice = self.env.abs_goal_slice

        self.device = device

        # network
        self.dqn = net.to(self.device)
        self.dqn.eval()
        # if shared_state.get("dqn_state_dict", None) is not None:

        # set up replay buffer size
        self.shared_mem = shared_mem
        self.shared_state = shared_state
        self.shared_net = shared_net

        # set up scheduler
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon

        # transition to store in memory
        self.transition = list()  # state, selected_action， reward next state done

        # hyper-parameter
        self.reset_num = reset_num
        self.target_update_freq = target_update_freq

        # is-debug
        self.toggle_visual = toggle_visual

        # curriculum learning
        self.env.scheduler.state_level = self.state_level = self.shared_state['state_level']
        self.env.scheduler.class_level = self.class_level = self.shared_state['class_level']

    def refine_path_to_transitions(self, redundant_path, goal_pattern, infeasible_dict=None):
        # refined_path = refine_redundant_action(redundant_path,
        #                                        self.env.goal_pattern.state,
        #                                        infeasible_dict)
        goal_state = RackState(goal_pattern)
        if infeasible_dict is None:
            infeasible_dict = {}
        refined_path = rm_ras_actions_recur(redundant_path, h=3, infeasible_dict=infeasible_dict, )
        if len(refined_path) == len(redundant_path):
            return []
        env = self.env
        transitions = []
        for _i in np.arange(len(refined_path) - 1):
            s_current, s_next = refined_path[_i], refined_path[_i + 1]
            action = env.action_between_states(s_current=s_current, s_next=s_next)
            is_finsihed = env.is_finished(s_next, goal_pattern)
            # check if it is a unfinished state
            if _i == len(refined_path) - 2 and not is_finsihed:
                reward = -50
            else:
                reward = env._get_reward(is_finsihed, s_current, s_next, s_current)
            tran_tmp = [self.to_abs_state(RackState(s_current), goal_state=goal_state),
                        action,
                        reward,
                        self.to_abs_state(RackState(s_next), goal_state=goal_state),
                        is_finsihed]
            transitions.append(tran_tmp)
            if is_finsihed:
                break
        return transitions

    def extract_path_from_buffer(self, trainsition_buffer):
        return np.array([trainsition_buffer[0][0], *[i[-2] for i in trainsition_buffer]]), \
            [np.concatenate((i[0], i[-2])) for i in trainsition_buffer]

    def gen_bootstrapping_data(self, buffer, toggle_debug=False):
        redundant_abs_state, redundant_abs_state_paired = self.extract_path_from_buffer(buffer)
        buffer[:] = []

        redundant_path = redundant_abs_state[:, self.env_abs_state_slice, :]
        goal_state_np = redundant_abs_state[0][self.env_abs_goal_slice]
        for i in range(max(len(redundant_path) - 2, 0)):
            refined_transitions = self.refine_path_to_transitions(redundant_path[i:], goal_state_np)
            if len(refined_transitions) > 0 and toggle_debug:
                print(f"Add refined data into replay buffer: {len(refined_transitions) + 1}/{len(redundant_path)}")
            for t in refined_transitions:
                _t_paired = np.concatenate((t[0], t[-2]))
                if not any((_t_paired == x).all() for x in redundant_abs_state_paired):
                    self.shared_mem.put(t)

    def to_abs_state(self, state, env=None, goal_state=None):
        if env is not None:
            return np.vstack((env.goal_pattern.state, state.abs_state))
        elif goal_state is not None:
            return np.vstack((goal_state.state, state.abs_state))
        else:
            raise Exception

    def abs_to_state(self, abs_state):
        return abs_state[self.env_abs_state_slice, :]

    def select_action(self, state, env=None) -> np.ndarray:
        """Select an action from the input state."""
        if env is None:
            env = self.env
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = env.sample(state)
        else:
            selected_action = self.dqn_select_action(state.feasible_action_set, self.to_abs_state(state, env))
        self.transition = [self.to_abs_state(state, env), selected_action]
        return selected_action

    def step(self, action: np.ndarray, env=None, store_in_tmp=False) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        if env is None:
            env = self.env
        next_state, reward, done, _ = env.step(action)  # env step
        # reward clip:
        reward = reward
        self.transition += [reward, self.to_abs_state(next_state, env), done]  #
        if store_in_tmp:
            self.replaybuffer_tmp_store.append(self.transition)
        self.shared_mem.put(self.transition)
        return next_state, reward, done

    def dqn_select_action(self, feasible_action_set, abs_state):
        feasible_action_set = torch.as_tensor(feasible_action_set, dtype=torch.int64, device=self.device)
        dqn_action_value = self.dqn(
            torch.as_tensor(abs_state[self.env_abs_state_slice, :], dtype=torch.float32, device=self.device).unsqueeze(
                0),
            torch.as_tensor(abs_state[self.env_abs_goal_slice, :], dtype=torch.float32, device=self.device).unsqueeze(
                0), ).detach()
        selected_action = feasible_action_set[dqn_action_value.squeeze()[feasible_action_set].argmax()].item()
        return selected_action

    def run(self):
        """Train the agent."""
        # init
        _step = 0
        min_epsilon = self.min_epsilon
        max_epsilon = self.max_epsilon
        env = self.env
        scheduler = env.scheduler
        reset_num = self.reset_num
        target_update_freq = self.target_update_freq
        toggle_visual = self.toggle_visual
        plot = None

        # dqn load shared net weight
        self.dqn.load_state_dict(self.shared_net.state_dict())

        # start training
        print(f"Actor: {self._actor_id} Start Training")
        for i_episode in range(1, 500000000000000):
            state = env.reset()
            self.replaybuffer_tmp_store = []
            _reset_cnt = 0
            _score = 0

            if _step % 50 == 0:
                if self.state_level != self.shared_state['state_level']:
                    scheduler.state_level = self.state_level = self.shared_state['state_level']
                    self.epsilon = 1.

                if self.class_level != self.shared_state['class_level']:
                    scheduler.class_level = self.class_level = self.shared_state['class_level']
                    self.epsilon = 1.

            # plot
            if toggle_visual:
                rsp = RackStatePlot(env.goal_pattern, )
                plot = rsp.plot_states([state]).get_img()
            for _ in itertools.count():
                action = self.select_action(state, env)  # step
                next_state, reward, done = self.step(action, env, store_in_tmp=True)  # next_state reward done
                state = next_state  # state = next_state
                _score += reward  # reward
                _reset_cnt += 1
                _step += 1
                # linearly decrease epsilon
                self.epsilon = max(min_epsilon, self.epsilon - (max_epsilon - min_epsilon) * self.epsilon_decay)

                # plot
                if toggle_visual:
                    plot = np.concatenate((plot, rsp.plot_states([state]).get_img()), axis=1)
                    cv2.imshow(f"plot_{self._actor_id}", plot)
                    cv2.waitKey(100)
                if _step % target_update_freq == 0:
                    # if shared_state.get("dqn_state_dict", None) is not None:
                    # if _step % (target_update_freq * 1000) == 0:
                    #     print(
                    #         f"Agent {self._actor_id} -> Update 1000 state, Epsilon {self.epsilon:.3f}, State Level {scheduler.state_level}, Class Level {scheduler.class_level}")
                    self.dqn.load_state_dict(self.shared_net.state_dict())
                    # self.dqn.load_state_dict(self.shared_state["dqn_state_dict"])
                # if episode ends
                if done:  # done
                    # print(f":: Episode {i_episode}: done with score: {_score}")
                    # print(f"the total length is: {len(env.reward_history)} ", env.reward_history)
                    # self.gen_bootstrapping_data(self.replaybuffer_tmp_store)
                    break
                if _reset_cnt % reset_num == 0:
                    # print(f":: Episode {i_episode}: action cannot find the solution within {reset_num} steps")
                    # self.gen_bootstrapping_data(self.replaybuffer_tmp_store)
                    break

            if toggle_visual:
                print(f'Episode: {i_episode:<4}  '
                      f'Episode steps: {_ + 1:<4}  '
                      f'Return: {_score:<5.1f}')
                cv2.waitKey(0)
                del plot
                plot = None


class HERActor(Actor):
    def __init__(self,
                 actor_id: int,
                 net,
                 env: Gym_Proto,
                 max_epsilon,
                 min_epsilon,
                 epsilon_decay,
                 reset_num,
                 target_update_freq,
                 shared_net,
                 shared_state,
                 shared_reanalyzer_mem,
                 shared_mem,
                 device,
                 toggle_visual=False):

        super().__init__(actor_id,
                         net,
                         env,
                         max_epsilon,
                         min_epsilon,
                         epsilon_decay,
                         reset_num,
                         target_update_freq,
                         shared_net,
                         shared_state,
                         shared_mem,
                         device,
                         toggle_visual, )

        self.shared_reanalyzer_mem = shared_reanalyzer_mem
        if self.shared_reanalyzer_mem is not None:
            self.save_data = True
        else:
            self.save_data = False

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

    def refine_path_to_transitions(self, redundant_path, goal_pattern, infeasible_dict=None):
        if infeasible_dict is None:
            infeasible_dict = {}
        refined_path = rm_ras_actions(redundant_path,
                                      h=8,
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
                refined_transitions = self.refined_path_to_transitions_recur(refined_path[i], goal_state_np)
                if len(refined_transitions) > 0 and toggle_debug:
                    print(f"Add refined data into replay buffer: {len(refined_transitions) + 1}/{len(redundant_path)}")
                for t in refined_transitions:
                    _t_paired = np.concatenate((t[1], t[-2]))
                    if str(_t_paired) not in redundant_abs_state_paired_str:
                        self._store_transition(t)
                        redundant_abs_state_paired_str.append(str(_t_paired))

        if her_relabel_toggle:
            self.her_relabel(redundant_path, path_pair=copy.deepcopy(_redundant_abs_state_paired_str))

    def her_relabel(self, path, path_pair, max_relabel_goal_num=30, toggle_debug=False):
        result_list = []
        if len(path) > 1:

            goal_select_ids = self.env._np_random.choice(np.arange(1, len(path)),
                                                         min(len(path) - 1, max_relabel_goal_num), replace=False)
            goal_pattern_set = path[goal_select_ids]
            a = time.time()
            for idx, goal_pattern_tmp in enumerate(goal_pattern_set):
                if self._reset_eposilon():
                    return
                refined_transitions = self.refine_path_to_transitions(path[:goal_select_ids[idx] + 1], goal_pattern_tmp)
                # if len(refined_transitions) > 0 and toggle_debug:
                #     print(f"Add refined data into replay buffer: {len(refined_transitions) + 1}/{len(redundant_path)}")
                for t in refined_transitions:
                    _t_paired = np.concatenate((goal_pattern_tmp, t[1], t[-2]))
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

    def to_abs_state(self, state, env=None, goal_state=None):
        if env is not None:
            return np.vstack((env.goal_pattern.state, state.abs_state))
        elif goal_state is not None:
            return np.vstack((goal_state.state, state.abs_state))
        else:
            raise Exception

    def abs_to_state(self, abs_state):
        return abs_state[self.env_abs_state_slice, :]

    def select_action(self, state, env=None) -> np.ndarray:
        """Select an action from the input state."""
        if env is None:
            env = self.env
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = env.sample(state)
        else:
            selected_action = self.dqn_select_action(state.feasible_action_set, self.to_abs_state(state, env))
        self.transition = [env.goal_pattern.state, state.state, selected_action]
        # self.her_tmp_transition = [state.state, selected_action]
        return selected_action

    def step(self, action: np.ndarray, env=None, store_in_tmp=False) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        if env is None:
            env = self.env
        next_state, reward, done, _ = env.step(action)  # env step
        # reward clip:
        reward = reward
        self.transition += [reward, next_state.state, done]  #
        # self.her_tmp_transition += [reward, next_state.state]
        if store_in_tmp:
            self.replaybuffer_tmp_store.append(self.transition)

        self._store_transition(self.transition)
        # self.her_tmp_store.append(self.her_tmp_transition)
        return next_state, reward, done

    def dqn_select_action(self, feasible_action_set, abs_state, no_repeat=True):
        if no_repeat:
            feasible_action_set_tmp = feasible_action_set
            repeat_acts = []
            state = RackState(abs_state[self.env_abs_state_slice, :])
            for _ in self.env.rack_state_history:
                act = self.env.action_between_states_constraint_free(state, _)
                if act is not None:
                    repeat_acts.append(act)
            feasible_action_set = np.setdiff1d(feasible_action_set, repeat_acts)
            if len(feasible_action_set) == 0:
                feasible_action_set = feasible_action_set_tmp
                # self.early_stop = True
                # # print("Potential ERROR Happens")
                # print("state is", self.env.state)
                # print("goal pattern is", self.env.goal_pattern)

        with torch.no_grad():
            feasible_action_set = torch.as_tensor(feasible_action_set, dtype=torch.int64, device=self.device)
            dqn_action_value = self.dqn(
                torch.as_tensor(abs_state[self.env_abs_state_slice, :], dtype=torch.float32,
                                device=self.device).unsqueeze(
                    0),
                torch.as_tensor(abs_state[self.env_abs_goal_slice, :], dtype=torch.float32,
                                device=self.device).unsqueeze(
                    0), ).detach()
            selected_action = feasible_action_set[dqn_action_value.squeeze()[feasible_action_set].argmax()].item()
        return selected_action

    def _reset_eposilon(self, ):
        if self.state_level != self.shared_state['state_level'] or \
                self.class_level != self.shared_state['class_level']:
            self.epsilon = 1.
            print(f"Actor {self._actor_id}: eposilon reset,{self.epsilon}. "
                  f"State level {self.state_level} "
                  f"Class level {self.class_level}")
            RackState._cache.clear()
            self.scheduler.state_level = self.state_level = self.shared_state['state_level']
            self.scheduler.class_level = self.class_level = self.shared_state['class_level']
            return True
        else:
            return False

    def _store_transition(self, transition):
        self.shared_mem.put(transition)

    def run(self):
        try:
            """Train the agent."""
            # init
            _step = 0
            min_epsilon = self.min_epsilon
            max_epsilon = self.max_epsilon
            env = self.env
            scheduler = env.scheduler
            self.scheduler = scheduler
            reset_num = self.reset_num
            target_update_freq = self.target_update_freq
            toggle_visual = self.toggle_visual
            plot = None

            # dqn load shared net weight
            self.dqn.load_state_dict(self.shared_net.state_dict())

            # start training
            for i_episode in range(1, 500000000000000):
                state = env.reset()
                self.replaybuffer_tmp_store = []
                self.early_stop = False
                # self.her_tmp_store = []
                _reset_cnt = 0
                _score = 0

                # for curriculumn learning
                if _step % 5 == 0:
                    self._reset_eposilon()
                    # reset_num = reset_num * self.state_level * self.class_level

                # for plot
                if toggle_visual:
                    rsp = RackStatePlot(env.goal_pattern, )
                    plot = rsp.plot_states([state]).get_img()
                    img_list = [plot]

                start_ep_t = time.time()
                for _ in itertools.count():
                    action = self.select_action(state, env)  # step
                    next_state, reward, done = self.step(action, env, store_in_tmp=True)  # next_state reward done
                    state = next_state  # state = next_state
                    _score += reward  # reward
                    _reset_cnt += 1
                    _step += 1
                    # linearly decrease epsilon
                    self.epsilon = max(min_epsilon, self.epsilon - (max_epsilon - min_epsilon) * self.epsilon_decay)
                    # plot
                    if toggle_visual:
                        img_list.append(rsp.plot_states([state]).get_img())
                        plot = combine_images(img_list, columns=20)
                        cv2.imshow(f"plot_{self._actor_id}", plot)
                        cv2.waitKey(100)
                    if _step % target_update_freq == 0:
                        # if shared_state.get("dqn_state_dict", None) is not None:
                        if _step % (target_update_freq * 1000) == 0:
                            print(
                                f"Agent {self._actor_id} -> Update 1000 state, Epsilon {self.epsilon:.3f}, State Level {scheduler.state_level}, Class Level {scheduler.class_level}")
                        self.dqn.load_state_dict(self.shared_net.state_dict())
                        # self.dqn.load_state_dict(self.shared_state["dqn_state_dict"])
                    # if episode ends
                    if done:  # done
                        # print(f":: Episode {i_episode}: done with score: {_score}")
                        # print(f"the total length is: {len(env.reward_history)} ", env.reward_history)
                        # if self.epsilon > np.random.random():
                        #     self.gen_bootstrapping_data(self.replaybuffer_tmp_store, a_star_teacher_toggle=False,
                        #                                 her_relabel_toggle=True)
                        if self.save_data:
                            self.shared_reanalyzer_mem.put(
                                [self.state_level, self.class_level, self.replaybuffer_tmp_store])
                        break
                    if _reset_cnt % reset_num == 0 or self.early_stop:
                        # print(_reset_cnt, self.early_stop)
                        # print(f":: Episode {i_episode}: action cannot find the solution within {reset_num} steps")
                        # print(f"the total length is: {len(env.reward_history)} ", env.reward_history)
                        if self.save_data:
                            self.shared_reanalyzer_mem.put(
                                [self.state_level, self.class_level, self.replaybuffer_tmp_store])
                        break
                end_ep_t = time.time()

                if toggle_visual:
                    print(f'Episode: {i_episode:<4}  '
                          f'Episode steps: {_ + 1:<4}  '
                          f'Return: {_score:<5.1f}')
                    cv2.waitKey(0)
                    del plot
                    plot = None
        except Exception as e:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ERROR")
            print(f"{e}")


import seaborn as sns
import huri.components.utils.matlibplot_utils as mu


class Eval(mp.Process):
    def __init__(self,
                 net,
                 env_test,
                 reset_num,
                 eval_num,
                 eval_interval,
                 shared_net,
                 shared_state,
                 device,
                 save_model_path_best,
                 save_model_path_last,
                 inc_diff_threshold=48,
                 toggle_visual=False):
        super(Eval, self).__init__()
        self.env_test = env_test
        self.env_abs_state_slice = self.env_test.abs_state_slice
        self.env_abs_goal_slice = self.env_test.abs_goal_slice

        self.eval_interval = eval_interval
        self.eval_num = eval_num
        self.reset_num = reset_num
        self.dqn = net.to(device)
        self.dqn.eval()
        self.shared_net = shared_net
        self.shared_state = shared_state
        self.device = device

        self.inc_diff_threshold = inc_diff_threshold

        self.toggle_visual = toggle_visual

        # save
        self.save_model_path_best = save_model_path_best
        self.save_model_path_last = save_model_path_last

    def dqn_select_action(self, feasible_action_set, abs_state, no_repeat=True, toggle_debug=False):

        if no_repeat:
            repeat_acts = []
            state = RackState(abs_state[self.env_abs_state_slice, :])
            for _ in self.env_test.rack_state_history:
                act = self.env_test.action_between_states_constraint_free(state, _)
                if act is not None:
                    repeat_acts.append(act)
            feasible_action_set = np.setdiff1d(feasible_action_set, repeat_acts)
            if len(feasible_action_set) == 0:
                return -1

        feasible_action_set = torch.as_tensor(feasible_action_set, dtype=torch.int64, device=self.device)
        dqn_action_value = self.dqn(
            torch.as_tensor(abs_state[self.env_abs_state_slice, :], dtype=torch.float32, device=self.device).unsqueeze(
                0),
            torch.as_tensor(abs_state[self.env_abs_goal_slice, :], dtype=torch.float32, device=self.device).unsqueeze(
                0), ).detach()
        selected_action = feasible_action_set[dqn_action_value.squeeze()[feasible_action_set].argmax()].item()
        if False and toggle_debug:
            dqn_action_value2 = self.dqn.forward2(
                torch.as_tensor(abs_state[self.env_abs_state_slice, :], dtype=torch.float32,
                                device=self.device).unsqueeze(
                    0),
                torch.as_tensor(abs_state[self.env_abs_goal_slice, :], dtype=torch.float32,
                                device=self.device).unsqueeze(
                    0), ).detach().squeeze()[feasible_action_set]
            dqn_action_value1 = dqn_action_value.squeeze()[feasible_action_set]
            print("Difference: ", abs(dqn_action_value1.max() - dqn_action_value2.max()))
            print("minimum value: ", torch.abs(dqn_action_value1 - dqn_action_value2).min())
            print("Argmax: ", dqn_action_value1.argmax(), dqn_action_value2.argmax())
            v = dqn_action_value.cpu().numpy()
            i = feasible_action_set.cpu().numpy()

            # print(f"'Q:{v}, \n"
            #       f"Feasible action: {v[0][i]} \n")
            # rack_size = abs_state[self.env_abs_state_slice, :].shape
            # a_pick, a_place = self.env_test._expr_action(selected_action)
            # a_pick = np.array([a_pick])
            # a_place = np.array([a_place])

            def to_flatten_ids(rack_size, p_id):
                return p_id[:, 0] * rack_size[1] + p_id[:, 1]

            # vv = self.dqn.self_atten.atten_output_weights.detach().cpu().numpy().squeeze(0)
            # # vv = self.dqn.self_atten.atten_output_weights.detach().cpu().numpy().squeeze(0)[:18, :18]
            # adjacency_mat = self.env_test.to_state(abs_state[self.env_abs_state_slice, :]).adjacency_matrix
            # # vv = vv * adjacency_mat
            # print("START IDS:", np.where(abs_state[self.env_abs_state_slice, :].ravel())[0], )
            # print("GOAL IDS:", np.where(abs_state[self.env_abs_goal_slice, :].ravel())[0])
            # print(f"Selected action: {to_flatten_ids(rack_size, a_pick), to_flatten_ids(rack_size, a_place)}")
            # # print(np.unravel_index(np.argmin(vv * adjacency_mat, axis=None), vv.shape))
            # plot = mu.Plot(w=10, h=10, dpi=100, )
            # sns.heatmap(vv,
            #             linewidth=0.5,
            #             cmap="YlGnBu", ax=plot.axes[0])
            # plot_weight_matrix = plot.get_img()
            # cv2.imshow("weight_matrix", plot_weight_matrix)
        return selected_action

    def to_abs_state(self, state, env=None, goal_state=None):
        if env is not None:
            return np.vstack((env.goal_pattern.state, state.abs_state))
        elif goal_state is not None:
            return np.vstack((goal_state.state, state.abs_state))
        else:
            raise Exception

    def run(self):
        env = self.env_test
        scheduler = env.scheduler
        scheduler.state_level = self.shared_state['state_level']
        scheduler.class_level = self.shared_state['class_level']

        reset_num = self.reset_num
        best_eval_score = 0
        toggle_visual = self.toggle_visual
        failed_data_list_tmp = []
        for _ in itertools.count():
            time.sleep(self.eval_interval)
            self.dqn.load_state_dict(self.shared_net.state_dict())
            total_return = 0
            num_dones = 0
            total_len = 0

            a_star_total_len = 0
            a_star_num_dones = 0

            # env
            candidate_state_goal = []
            candidate_state_goal += failed_data_list_tmp
            # failed data list
            failed_data_list_tmp = []
            for i in range(self.eval_num):
                s = env.reset()
                g = env.goal_pattern
                candidate_state_goal.append([s, g])
            for i, (i_s, i_g) in enumerate(candidate_state_goal):
                t_score = 0
                t_state = env.reset_state_goal(i_s, i_g)
                # print(env.scheduler.state_level, env.is_curriculum_lr)
                goal_pattern = env.goal_pattern.state
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
                for t in itertools.count(1):
                    t_action = self.dqn_select_action(t_state.feasible_action_set,
                                                      self.to_abs_state(t_state, env=env),
                                                      toggle_debug=toggle_visual)
                    if t_action < 0:
                        t_next_state, t_reward, t_done = None, -10, True
                    else:
                        t_next_state, t_reward, t_done, _ = env.step(t_action)
                    t_reward = t_reward
                    t_state = t_next_state  # state = next_state
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
                    if t % int(np.ceil(reset_num * scheduler.state_level)) == 0:
                        failed_data_list_tmp.append([_['rack_history'][0], goal_pattern])
                        break

                if toggle_visual:
                    print(f'Evaluation Num: {i + 1}/{len(candidate_state_goal)} \n'
                          f'Episode steps: {t} \n'
                          f'History {env.reward_history} \n'
                          f'Return: {t_score:<5.1f} \n')
                    cv2.waitKey(0)
                    del plot
                    plot = None
            if num_dones >= (len(candidate_state_goal) * .9):
                if len(failed_data_list_tmp) > 0:
                    save_data_path = fs.Path('debug_data.pkl')
                    # if save_data_path.exists():
                    #     failed_data_list = fs.load_pickle(str(save_data_path))
                    # else:
                    #     failed_data_list = []
                    # failed_data_list.extend(failed_data_list_tmp)
                    fs.dump_pickle(failed_data_list_tmp, str(save_data_path), reminder=False)

                mean_return = total_return / num_dones
                if mean_return > best_eval_score:
                    best_eval_score = mean_return
                    if self.save_model_path_best is not None:
                        torch.save(self.dqn.state_dict(), str(self.save_model_path_best))

                if num_dones >= len(candidate_state_goal):
                    # ==================== update class ====================
                    # if num_dones >= (self.eval_num * .95):
                    new_state_level, new_class_level = scheduler.update_training_level()
                    self.shared_state['state_level'] = new_state_level
                    self.shared_state['class_level'] = new_class_level
                    RackState._cache.clear()
                    # self.shared_state['class_level'] += 1
                    scheduler.state_level = self.shared_state['state_level']
                    scheduler.class_level = self.shared_state['class_level']
                    # if scheduler.state_level > (scheduler.max_state_level_per_class * scheduler.class_level):
                    #     self.shared_state['class_level'] += 1
                    #     scheduler.class_level = self.shared_state['class_level']
                    # self._save_check_point(os.path.join(self.model_path, 'model_best.pth'))
                # print(
                #     f">>>>>>> average score: {num_dones}/{self.eval_num}, average len: {total_len / num_dones if num_dones > 0 else -1:.2f}")
                # self.env.increase_diffculty()
                # self.env_test.increase_diffculty()
                # self._target_hard_update()
            # else:
            #     failed_data_list_tmp = []
            print(
                f">>>>>>> average score: {num_dones}/{len(candidate_state_goal)}, "
                f"average len: {total_len / num_dones if num_dones > 0 else -1:.2f},"
                f"a star solver dones: {a_star_num_dones}/{len(candidate_state_goal)}, "
                f"a star solver average length: {a_star_total_len / a_star_num_dones if a_star_num_dones > 0 else -1:.2f} ",
                f"class level is {scheduler.class_level}, "
                f"state level is {scheduler.state_level}")
            if self.save_model_path_last is not None:
                torch.save(self.dqn.state_dict(), str(self.save_model_path_last))


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
    shared_mem_ = mp_manager.Queue()
    shared_state['state_level'] = env.scheduler.state_level
    shared_state['state_level'] = 1
    shared_state['class_level'] = 4

    actor = HERActor(actor_id=1, net=net, env=env,
                     shared_state=shared_state,
                     batch_size=32, max_epsilon=1, min_epsilon=.1, epsilon_decay=1e-5,
                     target_update_freq=200,
                     reset_num=100, shared_net=net, shared_mem=shared_mem, shared_reanalyzer_mem=shared_mem_,
                     device='cuda')
    actor.run()
    print("Main: replay_mem.size:", shared_mem.qsize())
    # while 1:
    #     time.sleep(1)
    # for i in range(shared_mem.qsize()):
    #     xp_batch = shared_mem.get()
    #     print("priority:", xp_batch)
