from huri.learning.method.DQN.dqn import DQNAgent, padding, F
import itertools
import numpy as np
import torch
from pathlib import Path
import logging
from typing import Tuple, Dict, List
import copy
from huri.examples.task_planning.a_star import TubePuzzle
from huri.learning.env.arrangement_planning_rack.utils import isdone
from huri.components.task_planning.tube_puzzle_learning_solver import action_between_states, refine_redundant_action
import os
from huri.learning.A_start_teacher.A_star_teacher import rm_ras_actions
from huri.learning.env.arrangement_planning_rack_goal_condition.env import RackState


class DQNAgent_r(DQNAgent):
    def __init__(
            self,
            env,
            net: torch.nn.Module,
            replay_size: int,
            batch_size: int,
            target_update_freq: int,
            lr: float,
            epsilon_decay: float,
            max_epsilon: float,
            min_epsilon: float,
            gamma: float,
            # SEED
            torch_seed: int = 777,
            env_seed: int = 555,
            env_test_seed: int = 9999,
            # PER parameters
            alpha: float = 0.2,
            beta: float = 0.6,
            prior_eps: float = 1e-6,
            path=Path("./run/"),
            logger=logging.getLogger(__name__)):
        super(DQNAgent_r, self).__init__(env=env, net=net, replay_size=replay_size, batch_size=batch_size,
                                         target_update_freq=target_update_freq, lr=lr, epsilon_decay=epsilon_decay,
                                         max_epsilon=max_epsilon, min_epsilon=min_epsilon, gamma=gamma,
                                         torch_seed=torch_seed, env_seed=env_seed, env_test_seed=env_test_seed,
                                         alpha=alpha, beta=beta, prior_eps=prior_eps, path=path, logger=logger)
        self.replaybuffer_tmp_store = []

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        gamma = self.gamma
        action_dim = self.action_dim
        env = self.env
        state = torch.as_tensor(samples["obs"], dtype=torch.float32, device=device)
        next_state = torch.as_tensor(samples["next_obs"], dtype=torch.float32, device=device)
        action = torch.as_tensor(samples["acts"].reshape(-1, 1), dtype=torch.int64, device=device)
        reward = torch.as_tensor(samples["rews"].reshape(-1, 1), dtype=torch.float32, device=device)
        done = torch.as_tensor(samples["done"].reshape(-1, 1), dtype=torch.float32, device=device)
        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        abs_to_state = self.abs_to_state
        next_state_feasible_action_set = torch.as_tensor(
            [padding(action_dim,
                     env.to_state(abs_to_state(next_obs_tmp.squeeze(0))).feasible_action_set
                     ) for next_obs_tmp in samples["next_obs"]],
            dtype=torch.int64, device=device)

        curr_q_value = self.dqn(state).gather(1, action)
        argmax_next_state_feasible_action = next_state_feasible_action_set.gather(1, self.dqn(next_state)
                                                                                  .gather(1,
                                                                                          next_state_feasible_action_set)
                                                                                  .argmax(dim=1, keepdim=True).detach())
        next_q_value = self.dqn_target(next_state).gather(  # Double DQN
            1, argmax_next_state_feasible_action).detach()
        mask = 1 - done
        target = (reward + gamma * next_q_value * mask).to(self.device)

        # calculate element-wise dqn loss
        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")

        return elementwise_loss

    def extract_path_from_replaybuffer_tmp_store(self):
        replaybuffer_tmp_store = self.replaybuffer_tmp_store
        return np.array([replaybuffer_tmp_store[0][0], *[i[-2] for i in replaybuffer_tmp_store]]), \
               [np.concatenate((i[0], i[-2])) for i in replaybuffer_tmp_store]

    def gen_bootstrapping_data(self):
        redundant_abs_state, redundant_abs_state_paired = self.extract_path_from_replaybuffer_tmp_store()
        redundant_path = redundant_abs_state[:, 5:10, :]
        goal_state_np = redundant_abs_state[0][:5]
        refined_transitions = self.refine_path_to_transitions(redundant_path, goal_state_np)
        if len(refined_transitions) > 0:
            self.logger.info(
                f"Add refined data into replay buffer: {len(refined_transitions) + 1}/{len(redundant_path)}")
        for t in refined_transitions:
            _t_paired = np.concatenate((t[0], t[-2]))
            if not any((_t_paired == x).all() for x in redundant_abs_state_paired):
                self.replaybuffer.store(*t)
                loss = self.update_model()
        self.replaybuffer_tmp_store = []

    def refine_path_to_transitions(self, redundant_path, goal_pattern, infeasible_dict=None):
        # refined_path = refine_redundant_action(redundant_path,
        #                                        self.env.goal_pattern.state,
        #                                        infeasible_dict)
        goal_state = RackState(goal_pattern)
        if infeasible_dict is None:
            infeasible_dict = {}
        refined_path = rm_ras_actions(redundant_path, h=3, infeasible_dict=infeasible_dict, )
        if len(refined_path) == len(redundant_path):
            return []
        env = self.env
        transitions = []
        for _i in np.arange(len(refined_path) - 1):
            s_current, s_next = refined_path[_i], refined_path[_i + 1]
            action = action_between_states(s_current, s_next)[0]
            is_finsihed = isdone(s_next, goal_pattern)
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

    def to_abs_state(self, state, env=None, goal_state=None):
        if env is not None:
            return np.vstack((env.goal_pattern.state, state.abs_state))
        elif goal_state is not None:
            return np.vstack((goal_state.state, state.abs_state))
        else:
            raise Exception

    def abs_to_state(self, abs_state):
        return abs_state[5:10, :]

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
        self.replaybuffer.store(*self.transition)
        return next_state, reward, done

    def init_replay_buffer(self, init_nums, reset_num):
        env = self.env.copy()
        env.is_curriculum_lr = False
        _step = 0
        state = env.reset()
        reset_cnt = 0
        score = 0
        while _step < init_nums:
            _step += 1
            action = self.select_action(state, env)  # step
            next_state, reward, done = self.step(action, env)  # next_state reward done
            state = next_state  # state = next_state
            score += reward  # reward
            reset_cnt += 1
            # if episode ends
            if done:  # done
                print(f":: Init replay buffer: done with score: {score}")
                print(f"the total length is: {len(env.reward_history)} ", env.reward_history)
                state = env.reset()
                score = 0
                reset_cnt = 0
            if reset_cnt >= reset_num:
                print(f":: Init replay buffer: action cannot find the solution within {reset_num} steps")
                state = env.reset()
                reset_cnt = 0
        del env

    def dqn_select_action(self, feasible_action_set, abs_state):
        feasible_action_set = torch.as_tensor(feasible_action_set, dtype=torch.int64, device=self.device)
        dqn_action_value = self.dqn(
            torch.as_tensor(abs_state, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)).detach()
        selected_action = feasible_action_set[dqn_action_value.squeeze()[feasible_action_set].argmax()].item()
        return selected_action

    def train(self, num_steps: int,
              reset_num: int = 800,
              astar_buffer_rate_episode=5,
              replay_init_steps: int = 5000,
              eval_interval=50):
        """Train the agent."""
        logger = self.logger
        _step = 0
        writer = self.writer
        min_epsilon = self.min_epsilon
        max_epsilon = self.max_epsilon
        env = self.env

        logger.info(f"Init Training. Data Saving in {self.path} ...")
        logger.info("Init Relay buffer ...")

        # init replay buffer
        self.init_replay_buffer(replay_init_steps, reset_num)

        # start training
        logger.info("Start Training")
        for i_episode in range(1, 500000000000000):
            if _step > num_steps:
                break
            # IS weight increasing
            fraction = min(_step / num_steps, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)
            writer.add_scalar("beta", self.beta, _step)
            # reset state and counter
            state = env.reset()
            self.replaybuffer_tmp_store = []
            _reset_cnt = 0
            _score = 0

            for _ in itertools.count():
                action = self.select_action(state, env)  # step
                next_state, reward, done = self.step(action, env, store_in_tmp=True)  # next_state reward done
                state = next_state  # state = next_state
                _score += reward  # reward
                _reset_cnt += 1
                _step += 1
                # update loss
                loss = self.update_model()
                # linearly decrease epsilon
                self.epsilon = max(min_epsilon, self.epsilon - (max_epsilon - min_epsilon) * self.epsilon_decay)
                # plot to tensorboard
                writer.add_scalar('loss-steps', loss, _step)
                writer.add_scalar('epsilon-steps', self.epsilon, _step)
                # if hard update is needed
                if _step % self.target_update_freq == 0:
                    self._target_hard_update()
                    # save the model
                    # self._save_check_point(self.model_path / f"model_{_step - self.target_update_freq}-{_step}.pth")
                    self._save_check_point(self.model_path / f"model.pth")
                # if episode ends
                if done:  # done
                    print(f":: Episode {i_episode}: done with score: {_score}")
                    print(f"the total length is: {len(env.reward_history)} ", env.reward_history)
                    writer.add_scalar('score-episode', _score, i_episode)
                    self.gen_bootstrapping_data()
                    break
                if _reset_cnt % reset_num == 0:
                    print(f":: Episode {i_episode}: action cannot find the solution within {reset_num} steps")
                    self.gen_bootstrapping_data()
                    break

            if i_episode % eval_interval == 0:
                logger.info("Start Evaluation ...")
                self.evaluate(writer, reset_num, _step)
                # self._save_replay_buffer(f"{train_setting}_replay_buffer_pri_cont")

            logger.info(f'Episode: {i_episode:<4}  '
                        f'Episode steps: {_ + 1:<4}  '
                        f'Return: {_score:<5.1f}')

        self._save_check_point(self.model_path + f"model_latest.pth")

    def evaluate(self, writer, reset_num, _step, eval_num=50):
        total_return = 0
        num_dones = 0
        env = self.env_test
        for i in range(eval_num):
            t_score = 0
            t_state = env.reset()
            for t in itertools.count(1):
                t_action = self.dqn_select_action(t_state.feasible_action_set, self.to_abs_state(t_state, env=env))
                t_next_state, t_reward, t_done, _ = env.step(t_action)
                t_reward = t_reward
                t_state = t_next_state  # state = next_state
                t_score += t_reward  # reward
                if t_done:
                    if t_reward > 0:
                        num_dones += 1
                    total_return += t_score
                    break
                if t % reset_num == 0:
                    break
        writer.add_scalar(
            'number_of_success/EVAL', num_dones, _step)
        if num_dones >= self.inc_diff_threshold:
            mean_return = total_return / num_dones
            if mean_return > self.best_eval_score:
                self.best_eval_score = mean_return
                self._save_check_point(os.path.join(self.model_path, 'model_best.pth'))
            writer.add_scalar(
                'reward/EVAL', mean_return, _step)
            print(f">>> average score: {mean_return}")
            self.env.increase_diffculty()
            self.env_test.increase_diffculty()
            self._target_hard_update()
            self.epsilon = 1
            # self.epsilon_decay = self.epsilon_decay / 1.2
        else:
            print(f">>> average score: {total_return}/{num_dones}")
        self.logger.info(f"Evaluation Result: number of success num {num_dones}/{eval_num}. "
                         f"New difficulty level: {self.env.difficulty}")
