from huri.learning.method.DQN.dqn import DQNAgent
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

    def extract_path_from_replaybuffer_tmp_store(self):
        replaybuffer_tmp_store = self.replaybuffer_tmp_store
        return [replaybuffer_tmp_store[0][0], *[i[-2] for i in replaybuffer_tmp_store]], [np.concatenate((i[0], i[-2]))
                                                                                          for i in
                                                                                          replaybuffer_tmp_store]

    def gen_bootstrapping_data(self):
        redundant_path, redundant_path_paired = self.extract_path_from_replaybuffer_tmp_store()
        refined_transitions = self.refine_path_to_transitions(redundant_path)
        if len(refined_transitions) > 0:
            self.logger.info("Add refined data into replay buffer")
        for t in refined_transitions:
            _t_paired = np.concatenate((t[0], t[-2]))
            if not any((_t_paired == x).all() for x in redundant_path_paired):
                self.replaybuffer.store(*t)
                loss = self.update_model()
        self.replaybuffer_tmp_store = []

    def refine_path_to_transitions(self, redundant_path, infeasible_dict={}):
        refined_path = refine_redundant_action(redundant_path,
                                               self.env.goal_pattern.state,
                                               infeasible_dict)
        if len(refined_path) == len(redundant_path):
            return []
        env = self.env
        transitions = []
        for _i in np.arange(len(refined_path) - 1):
            s_current, s_next = refined_path[_i], refined_path[_i + 1]
            action = action_between_states(s_current, s_next)[0]
            is_finsihed = isdone(s_next, env.goal_pattern.state)
            if _i == len(refined_path) - 2 and not is_finsihed:
                reward = -50
            else:
                reward = env._get_reward(is_finsihed, s_current, s_next, s_current)
            tran_tmp = [s_current, action, reward, s_next, is_finsihed]
            transitions.append(tran_tmp)
            if is_finsihed:
                break
        return transitions

    def step(self, action: np.ndarray, env=None, store_in_tmp=False) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        if env is None:
            env = self.env
        next_state, reward, done, _ = env.step(action)  # env step
        # reward clip:
        reward = reward
        self.transition += [reward, next_state.state, done]  #
        if store_in_tmp:
            self.replaybuffer_tmp_store.append(self.transition)
        self.replaybuffer.store(*self.transition)

        return next_state, reward, done

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
            state = self.env.reset()
            self.replaybuffer_tmp_store = []
            _reset_cnt = 0
            _score = 0

            for _ in itertools.count():
                action = self.select_action(state)  # step
                next_state, reward, done = self.step(action, store_in_tmp=True)  # next_state reward done
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
                    print(f"the total length is: {len(self.env.reward_history)} ", self.env.reward_history)
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
