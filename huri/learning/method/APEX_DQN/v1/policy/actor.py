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
from pathlib import Path


# TODO sync parameters
class Trajectory(object):
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
            transitions = []
            states = self.states[index]
            actions = self.actions[index]
            rewards = self.rewards[index]
            next_states = self.next_states[index]
            dones = self.dones[index]
            for i in range(len(states)):
                state = states[i]
                action = actions[i]
                reward = rewards[i]
                next_state = next_states[i]
                done = dones[i]
                transitions.append((state, action, reward, next_state, done))
            return transitions
        else:
            state = self.states[index]
            action = self.actions[index]
            reward = self.rewards[index]
            next_state = self.next_states[index]
            done = self.dones[index]
            return (state, action, reward, next_state, done)

    def __repr__(self):
        return (self.states[0], *self.next_states)


class Actor(mp.Process):
    def __init__(self,
                 actor_id: int,
                 net,
                 env: Gym_Proto,
                 cfg,
                 shared_net,
                 shared_state,
                 replay_buffer,
                 logger=None,
                 toggle_visual=False):
        super(Actor, self).__init__()
        self._actor_id = actor_id
        self.env = env
        self.scheduler = self.env.scheduler
        self.device = cfg['device']
        # network
        self.dqn = net.to(self.device)
        self.dqn.eval()
        # if shared_state.get("dqn_state_dict", None) is not None:
        # set up replay buffer size
        self.replay_buffer = replay_buffer
        self.shared_net = shared_net
        self.shared_state = shared_state
        # set up scheduler
        self.epsilon = cfg['eps_max']
        self.epsilon_decay = cfg['eps_decay']
        self.max_epsilon = cfg['eps_max']
        self.min_epsilon = cfg['eps_min']
        self.store_replay_buffer = cfg['store_replay_buffer']
        # hyper-parameter
        self.reset_num = cfg['reset_num']
        self.target_update_freq = cfg['update_freq']
        self.send_period = cfg['send_period']  # periodically send data to replay
        # is-debug
        self.toggle_visual = toggle_visual
        # logger
        self.logger = logger

    def select_action(self, state, env=None, toggle_no_repeat=True) -> np.ndarray:
        """Select an action from the input state."""
        if env is None:
            env = self.env
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = env.sample(state, toggle_no_repeat)
        else:
            selected_action = self.dqn_select_action(state.feasible_action_set, state, toggle_no_repeat)
        return selected_action

    def step(self, action: np.ndarray, env=None) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        if env is None:
            env = self.env
        next_state, reward, done, _ = env.step(action)  # env step
        return next_state, reward, done

    def dqn_select_action(self, feasible_action_set, state, toggle_no_repeat=True):
        if toggle_no_repeat:
            feasible_action_set_tmp = feasible_action_set
            repeat_acts = []
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
        if len(feasible_action_set) < 1:
            return None
        with torch.no_grad():
            feasible_action_set = torch.as_tensor(feasible_action_set, dtype=torch.int64, device=self.device)
            dqn_action_value = self.dqn(torch.as_tensor(state.abs_state, dtype=torch.float32,
                                                        device=self.device).unsqueeze(0),
                                        torch.as_tensor(self.env.goal_pattern.abs_state, dtype=torch.float32,
                                                        device=self.device).unsqueeze(0),
                                        torch.as_tensor(state.to_abs_state, dtype=torch.float32,
                                                        device=self.device).unsqueeze(0),
                                        ).detach()
            selected_action = feasible_action_set[dqn_action_value.squeeze()[feasible_action_set].argmax()].item()
        return selected_action

    def store_traj_list(self, trajectory_list, store_to_replay=False):
        if not store_to_replay:
            self.shared_state['trajectory_list'] = self.shared_state['trajectory_list'] + trajectory_list
        else:
            replay_buffer = self.replay_buffer
            for traj in trajectory_list:
                goal = traj.goal
                for i in range(len(traj)):
                    state, action, reward, next_state, done = traj[i]
                    replay_buffer.store(goal=goal.state,
                                        obs=state.state,
                                        c_abs_obs=state.compressed_abs_state,
                                        act=action,
                                        rew=reward,
                                        next_obs=next_state.state,
                                        c_abs_next_obs=next_state.compressed_abs_state,
                                        done=done, )
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

    def run(self):
        """Train the agent."""
        # init
        min_epsilon = self.min_epsilon
        max_epsilon = self.max_epsilon
        env = self.env
        scheduler = self.scheduler
        reset_num = self.reset_num
        target_update_freq = self.target_update_freq
        toggle_visual = self.toggle_visual
        logger = self.logger
        trajectory_list = []
        plot = None
        # ==========
        # dqn load shared net weight
        self.dqn.load_state_dict(self.shared_net.state_dict())
        # start training
        step = 0
        last_state_level = scheduler.state_level
        last_class_level = scheduler.state_level
        for i_episode in range(1, 500000000000000):
            # rest counter, score
            reset_cnt = 0
            score = 0
            # curriculum learning
            scheduler.state_level = self.shared_state['state_level']
            scheduler.class_level = self.shared_state['class_level']
            if scheduler.state_level != last_state_level or scheduler.class_level != last_class_level:
                self.epsilon = max_epsilon
                last_state_level = scheduler.state_level
                last_class_level = scheduler.state_level
            # state reset
            state = env.reset()
            # create a class to hold trajectory
            traj = Trajectory(env.goal_pattern)
            # for curriculumn learning
            # if step % 5 == 0:
            #     self._reset_eposilon()
            # reset_num = reset_num * self.state_level * self.class_level
            # for plot
            if toggle_visual:
                rsp = RackStatePlot(env.goal_pattern, )
                plot = rsp.plot_states([state]).get_img()
                img_list = [plot]
            # start one epoch
            start_ep_t = time.time()
            is_success_traj = True
            for _ in itertools.count():
                # select action
                action = self.select_action(state, env)
                if action is None:
                    is_success_traj = False
                    break
                # step
                next_state, reward, done = self.step(action, env)  # next_state reward done
                # store trajectory
                traj.add_transition(state, action, reward, next_state, done)
                # reset state
                state = next_state  # state = next_state
                # update score, coutner
                score += reward  # reward
                reset_cnt += 1
                step += 1
                # linearly decrease epsilon
                # self.epsilon = max(min_epsilon, self.epsilon - (max_epsilon - min_epsilon) * self.epsilon_decay)
                self.epsilon = max(min_epsilon, self.epsilon * self.epsilon_decay)
                if logger is not None:
                    logger.log({'actor epsilon': self.epsilon, })
                # plot
                if toggle_visual:
                    img_list.append(rsp.plot_states([state]).get_img())
                    plot = combine_images(img_list, columns=20)
                    cv2.imshow(f"plot_{self._actor_id}", plot)
                    cv2.waitKey(100)

                # update
                if step % target_update_freq == 0:
                    # if shared_state.get("dqn_state_dict", None) is not None:
                    if step % (target_update_freq * 1000) == 0:
                        print(
                            f"Agent {self._actor_id} -> Update 1000 state, Epsilon {self.epsilon:.3f}, State Level {scheduler.state_level}, Class Level {scheduler.class_level}")
                    self.dqn.load_state_dict(self.shared_net.state_dict())
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
                self.store_traj_list(trajectory_list, False if self.epsilon > np.random.random() else True)

            if logger is not None:
                logger.log({'actor epoch time computation': end_ep_t - start_ep_t,
                            'actor epoch return': score,
                            'acotr epoch length': _ + 1,
                            'actor_state_level': scheduler.state_level,
                            'actor_class_level': scheduler.class_level})
            if toggle_visual:
                cv2.waitKey(0)
                del plot
                plot = None


class Eval(mp.Process):
    def __init__(self,
                 net,
                 env: Gym_Proto,
                 cfg,
                 shared_net,
                 shared_state,
                 save_path=None,
                 logger=None,
                 toggle_visual=False):
        super(Eval, self).__init__()
        self.env = env
        self.save_checkpnt_path = Path(save_path).joinpath('model_best.chkpt') if save_path is not None else None
        self.device = cfg['device']

        # network
        self.dqn = net.to(self.device)
        self.dqn.eval()
        # if shared_state.get("dqn_state_dict", None) is not None:
        # set up replay buffer size
        self.shared_net = shared_net
        self.shared_state = shared_state
        # hyper-parameter
        self.eval_num = cfg['eval_num']
        self.reset_num = cfg['reset_num']
        self.eval_interval = cfg['eval_interval']
        # is-debug
        self.toggle_visual = toggle_visual
        # logger
        self.logger = logger

    def dqn_select_action(self, feasible_action_set, state, goal, toggle_no_repeat=True):
        if toggle_no_repeat:
            feasible_action_set_tmp = feasible_action_set
            repeat_acts = []
            for _ in self.env.rack_state_history:
                act = self.env.action_between_states_constraint_free(state, _)
                if act is not None:
                    repeat_acts.append(act)
            feasible_action_set = np.setdiff1d(feasible_action_set, repeat_acts)
            if len(feasible_action_set) == 0:
                feasible_action_set = feasible_action_set_tmp
        if len(feasible_action_set) < 1:
            return None
        with torch.no_grad():
            feasible_action_set = torch.as_tensor(feasible_action_set, dtype=torch.int64, device=self.device)
            dqn_action_value = self.dqn(torch.as_tensor(state.state, dtype=torch.float32,
                                                        device=self.device).unsqueeze(0),
                                        torch.as_tensor(goal.state, dtype=torch.float32,
                                                        device=self.device).unsqueeze(0),
                                        torch.as_tensor(state.state, dtype=torch.float32,
                                                        device=self.device).unsqueeze(0),
                                        ).detach()
            selected_action = feasible_action_set[dqn_action_value.squeeze()[feasible_action_set].argmax()].item()
        return selected_action

    def run(self) -> None:
        env = self.env
        scheduler = env.scheduler
        scheduler.state_level = self.shared_state['state_level']
        scheduler.class_level = self.shared_state['class_level']
        reset_num = self.reset_num
        best_eval_score = 0
        toggle_visual = self.toggle_visual
        for _ in itertools.count():
            # sleep
            time.sleep(self.eval_interval)
            self.dqn.load_state_dict(self.shared_net.state_dict())
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
                    t_action = self.dqn_select_action(t_state.feasible_action_set,
                                                      t_state,
                                                      i_g)
                    if t_action is None:
                        if t < 2:
                            is_state_goal_feasible = False
                        break
                    if t_action < 0:
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
                    if t % int(np.ceil(reset_num * scheduler.state_level)) == 0:
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
            if num_dones >= (feasible_candidate_len * .99):
                if len(failed_data_list_tmp) > 0:
                    save_data_path = fs.Path('debug_data.pkl')
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
                    scheduler.update_training_level()
                    RackState._cache.clear()
                    # self.shared_state['class_level'] += 1
                    self.shared_state['state_level'] = scheduler.state_level
                    self.shared_state['state_level'] = scheduler.class_level
            self.logger.log({
                'eval_average_len': total_len / num_dones if num_dones > 0 else -1,
                'eval_average_score': total_return / num_dones if num_dones > 0 else -1,
                'eval_num_dones_ratio': num_dones / self.eval_num,
                'eval_state_level': scheduler.state_level,
                'eval_class_level': scheduler.class_level,
            })
            print(
                f">>>>>>> average score: {num_dones}/{self.eval_num}, "
                f"average len: {total_len / num_dones if num_dones > 0 else -1:.2f},"
                # f"a star solver dones: {a_star_num_dones}/{len(candidate_state_goal)}, "
                # f"a star solver average length: {a_star_total_len / a_star_num_dones if a_star_num_dones > 0 else -1:.2f} ",
                f"class level is {scheduler.class_level}, "
                f"state level is {scheduler.state_level}")


from tqdm import tqdm


class Playground():
    def __init__(self,
                 net,
                 env: Gym_Proto,
                 cfg,
                 shared_state,
                 toggle_visual,
                 ):
        super(Playground, self).__init__()
        self.env = env
        self.device = cfg['device']

        # network
        self.dqn = net.to(self.device)
        self.dqn.eval()
        self.shared_state = shared_state
        # hyper-parameter
        self.eval_num = cfg['eval_num']
        self.reset_num = cfg['reset_num']
        self.eval_interval = cfg['eval_interval']
        # is-debug
        self.toggle_visual = toggle_visual

    def dqn_select_action(self, feasible_action_set, state, goal, toggle_no_repeat=True):
        if toggle_no_repeat:
            feasible_action_set_tmp = feasible_action_set
            repeat_acts = []
            for _ in self.env.rack_state_history:
                act = self.env.action_between_states_constraint_free(state, _)
                if act is not None:
                    repeat_acts.append(act)
            feasible_action_set = np.setdiff1d(feasible_action_set, repeat_acts)
            if len(feasible_action_set) == 0:
                feasible_action_set = feasible_action_set_tmp
        if len(feasible_action_set) < 1:
            return None
        with torch.no_grad():
            feasible_action_set = torch.as_tensor(feasible_action_set, dtype=torch.int64, device=self.device)
            dqn_action_value = self.dqn(torch.as_tensor(state.abs_state, dtype=torch.float32,
                                                        device=self.device).unsqueeze(0),
                                        torch.as_tensor(goal.abs_state, dtype=torch.float32,
                                                        device=self.device).unsqueeze(0),
                                        torch.as_tensor(state.to_abs_state, dtype=torch.float32,
                                                        device=self.device).unsqueeze(0),
                                        ).detach()
            v = torch.zeros(dqn_action_value.squeeze().shape[0], dtype=torch.float32, device=self.device)
            v[feasible_action_set] = 1
            print((dqn_action_value.squeeze() * v).max())
            indd = torch.triu_indices(*state.abs_size, 1)
            action_table = torch.zeros(state.abs_size, device=self.device)
            action_table[indd[0], indd[1]] = dqn_action_value.squeeze() * v
            action_table = action_table + action_table.T
            print(action_table)

            selected_action = feasible_action_set[dqn_action_value.squeeze()[feasible_action_set].argmax()].item()
        return selected_action

    def run(self) -> None:
        env = self.env
        scheduler = env.scheduler
        scheduler.state_level = self.shared_state['state_level']
        scheduler.class_level = self.shared_state['class_level']
        reset_num = self.reset_num
        best_eval_score = 0
        toggle_visual = self.toggle_visual
        for _ in itertools.count():
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
            for i, (i_s, i_g) in tqdm(enumerate(candidate_state_goal)):
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
                for t in itertools.count(1):
                    t_action = self.dqn_select_action(t_state.feasible_action_set,
                                                      t_state,
                                                      i_g)
                    if t_action is None:
                        break
                    if t_action < 0:
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
                    if t % int(np.ceil(reset_num * scheduler.state_level)) == 0:
                        failed_data_list_tmp.append([i_s, i_g])
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

                # set to buffer
                # [buffer.put(copy.deepcopy(failed_data_list_tmp)) for buffer in self.difficult_case_buffer_list]
                mean_return = total_return / num_dones
                if mean_return > best_eval_score:
                    best_eval_score = mean_return
                if num_dones >= len(candidate_state_goal):
                    # if num_dones >= (self.eval_num * .95):
                    if self.shared_state['class_level'] == scheduler.num_classes:
                        self.shared_state['state_level'] += 1
                        self.shared_state['class_level'] = 1
                    else:
                        self.shared_state['class_level'] += 1
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
                f">>>>>>> average score: {num_dones}/{self.eval_num}, "
                f"average len: {total_len / num_dones if num_dones > 0 else -1:.2f},"
                # f"a star solver dones: {a_star_num_dones}/{len(candidate_state_goal)}, "
                # f"a star solver average length: {a_star_total_len / a_star_num_dones if a_star_num_dones > 0 else -1:.2f} ",
                f"class level is {scheduler.class_level}, "
                f"state level is {scheduler.state_level}")
