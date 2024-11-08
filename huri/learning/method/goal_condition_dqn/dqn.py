import itertools
from typing import Tuple, Dict, List
from huri.learning.method.DQN.utils import PrioritizedReplayBuffer
import numpy as np
import torch
from tensorboardX import SummaryWriter
import time
import os
import copy
import torch.nn.functional as F
from pathlib import Path
import logging


# from file_sys import dump_pickle, load_pickle


def padding(size, array):
    if len(array) == 0:
        return np.zeros(size, dtype=int)
    max_v = max(array)
    pad_array = np.ones(size, dtype=int) * max_v
    pad_array[:len(array)] = array
    return pad_array


class DQNAgent:
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
        """Initialization.

        Args:
            replay_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update_freq (int): period for target model's hard update
            epsilon_decay (float): step size to decrease epsilon
            lr (float): learning rate
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            gamma (float): discount factor
        """
        # fundamental information
        self.obs_dim = env.observation_space_dim
        self.action_dim = env.action_space_dim
        self.num_classes = env.num_classes
        self.rack_size = env.rack_size
        self.gamma = gamma

        # batch size
        self.batch_size = batch_size

        # env for training and env for testing
        self.env = env
        self.env_test = copy.deepcopy(env)

        # setup the seed
        def seed_torch(seed):
            torch.manual_seed(seed)
            if torch.backends.cudnn.enabled:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

        np.random.seed(torch_seed)  # numpy seed
        seed_torch(torch_seed)  # torch seed
        self.env.seed(env_seed)  # training env seed
        self.env_test.seed(env_test_seed)  # test env seed

        # set up replay buffer size
        self.replaybuffer = PrioritizedReplayBuffer(self.obs_dim, replay_size, self.batch_size, alpha)
        self.beta = beta
        self.prior_eps = prior_eps

        # set up scheduler
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon

        # set up update freq
        self.target_update_freq = target_update_freq

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # networks: dqn, dqn_target
        self.dqn = net.to(self.device)
        self.dqn_target = copy.deepcopy(self.dqn)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)

        # transition to store in memory
        self.transition = list()  # state, selected_action， reward next state done

        # set training model path and tensorboard info path
        self.path = path
        self.model_path = path / "model"
        if not (self.model_path.is_dir()):
            self.model_path.mkdir()
        self.tensor_board_path = path / "tensor_board"
        self.writer = SummaryWriter(logdir=str(self.tensor_board_path))

        # logger
        self.logger = logger

        # update difficulty threshold
        self.best_eval_score = -np.inf
        self.inc_diff_threshold = 47

    def dummy_select_action(self, state, selected_action):
        self.transition = [state, selected_action]
        return selected_action

    def select_action(self, state, env=None) -> np.ndarray:
        """Select an action from the input state."""
        if env is None:
            env = self.env
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = env.sample(state)
        else:
            selected_action = self.dqn_select_action(state)
        self.transition = [state.state, selected_action]

        return selected_action

    def dqn_select_action(self, state):
        feasible_action_set = torch.as_tensor(
            state.feasible_action_set, dtype=torch.int64, device=self.device)
        dqn_action_value = self.dqn(
            torch.as_tensor(state.state, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)).detach()
        selected_action = feasible_action_set[dqn_action_value.squeeze()[feasible_action_set].argmax()].item()
        return selected_action

    def step(self, action: np.ndarray, env=None) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        if env is None:
            env = self.env
        next_state, reward, done, _ = env.step(action)  # env step
        # reward clip:
        reward = reward
        self.transition += [reward, next_state.state, done]  #
        self.replaybuffer.store(*self.transition)

        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.replaybuffer.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]

        elementwise_loss = self._compute_dqn_loss(samples)
        loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.replaybuffer.update_priorities(indices, new_priorities)

        return loss.item()

    def init_replay_buffer(self, init_nums, reset_num):
        env = self.env.copy()
        env.is_curriculum_lr = False
        _step = 0
        state = env.reset()
        reset_cnt = 0
        score = 0
        while _step < init_nums:
            _step += 1
            action = self.select_action(state)  # step
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

    def astar_replaybuffer(self, state):
        sol = self.env.gen_Astar_solution(state, max_iter_cnt=50)
        for action in sol:
            selected_action = self.dummy_select_action(state, action)
            state, reward, done = self.step(selected_action)
            if done:
                return None
        return state

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
            _reset_cnt = 0
            _score = 0
            # if i_episode % astar_buffer_rate_episode == 0:
            #     r = self.astar_replaybuffer(state)
            #     if r is None:  # if the Astar solved, return None
            #         print(f'Episode: {i_episode:<4}  ')
            #         continue
            #     state = r
            for _ in itertools.count():
                action = self.select_action(state)  # step
                next_state, reward, done = self.step(action)  # next_state reward done
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
                    self._save_check_point(self.model_path / f"model_{_step - self.target_update_freq}-{_step}.pth")
                # if episode ends
                if done:  # done
                    print(f":: Episode {i_episode}: done with score: {_score}")
                    print(f"the total length is: {len(self.env.reward_history)} ", self.env.reward_history)
                    writer.add_scalar('score-episode', _score, i_episode)
                    break
                if _reset_cnt % reset_num == 0:
                    print(f":: Episode {i_episode}: action cannot find the solution within {reset_num} steps")
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
        for i in range(eval_num):
            t_score = 0
            t_state = self.env_test.reset()
            for t in itertools.count(1):
                t_action = self.dqn_select_action(t_state)
                t_next_state, t_reward, t_done, _ = self.env_test.step(t_action)
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
        next_state_feasible_action_set = torch.as_tensor(
            [padding(action_dim,
                     env.to_state(next_obs_tmp.squeeze(0)).feasible_action_set
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

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.logger.info("Target Network Updated")
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _save_check_point(self, chkpt_file):
        chkpt_file = str(chkpt_file)
        logging.info(f"Saving the model check point in {chkpt_file}")
        torch.save(self.dqn.state_dict(), chkpt_file)

    def _load_checkpoint(self, chkpt_file):
        self.dqn.load_state_dict(torch.load(chkpt_file))
        self._target_hard_update()

    def continue_train(self, num_steps, reset_num, eval_interval, epsilon,
                       replay_buffer, chkpt_file, replay_init_steps=0):
        self.epsilon = epsilon
        self._load_checkpoint(chkpt_file)
        if self._load_replay_buffer(replay_buffer):
            print("buffer loadded successfully")
            print(f"buffer_size: {len(self.replaybuffer)}")
        else:
            print("load buffer failed")
        self.train(num_steps=num_steps,
                   reset_num=reset_num,
                   replay_init_steps=replay_init_steps,
                   eval_interval=eval_interval)

    def _save_replay_buffer(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        dump_pickle(self.replaybuffer, os.path.join(save_dir, "replaybuffer.pkl"))

    def _load_replay_buffer(self, buffer_dir):
        if not os.path.exists(buffer_dir):
            print("Load Failed!")
            return False
        self.replaybuffer = load_pickle(os.path.join(buffer_dir, "replaybuffer.pkl"))
        print("buffer loadded successfully")
        print(f"buffer_size: {len(self.replaybuffer)}")
        return True
