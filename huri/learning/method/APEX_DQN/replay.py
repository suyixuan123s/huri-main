import time
from typing import Dict, List, Callable, Tuple
import operator
import numpy as np
import os
import random
from multiprocessing.shared_memory import SharedMemory
import multiprocessing as mp
import torch


def gen_replay_buffer_shared_memo(obs_dim: list, size: int, batch_size: int = 32):
    obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)
    next_obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)
    acts_buf = np.zeros([size], dtype=np.float32)
    rews_buf = np.zeros([size], dtype=np.float32)
    done_buf = np.zeros(size, dtype=np.float32)

    def create_np_shared_mem(np_data_to_share):
        """
        https://stackoverflow.com/questions/7894791/use-numpy-array-in-shared-memory-for-multiprocessing
        :param np_data_to_share:
        :return:
        """
        SHARED_DATA_DTYPE = np_data_to_share.dtype
        SHARED_DATA_SHAPE = np_data_to_share.shape
        SHARED_DATA_NBYTES = np_data_to_share.nbytes
        shared_mem = SharedMemory(size=SHARED_DATA_NBYTES, create=True)
        arr = np.ndarray(SHARED_DATA_SHAPE, dtype=SHARED_DATA_DTYPE, buffer=shared_mem.buf)
        arr[:] = np_data_to_share[:]
        return shared_mem, SHARED_DATA_DTYPE, SHARED_DATA_SHAPE

    def create_int_shared_mem(data_to_share):
        val = mp.Value('i', 0)
        val.value = data_to_share
        return val

    # shared memory
    obs_buf_shared, obs_buf_shared_data_dtype, obs_buf_shared_data_shape = create_np_shared_mem(
        obs_buf)
    next_obs_buf_shared, next_obs_buf_shared_data_dtype, next_obs_buf_shared_data_shape = create_np_shared_mem(
        next_obs_buf)
    acts_buf_shared, acts_buf_shared_data_dtype, acts_buf_shared_data_shape = create_np_shared_mem(
        acts_buf)
    rews_buf_shared, rews_buf_shared_data_dtype, rews_buf_shared_data_shape = create_np_shared_mem(
        rews_buf)
    done_buf_shared, done_buf_shared_data_dtype, done_buf_shared_data_shape = create_np_shared_mem(
        done_buf)

    ptr_shared = create_int_shared_mem(0)
    size_shared = create_int_shared_mem(0)

    return {'obs_buf': [obs_buf_shared, obs_buf_shared_data_dtype, obs_buf_shared_data_shape],
            'next_obs_buf': [next_obs_buf_shared, next_obs_buf_shared_data_dtype, next_obs_buf_shared_data_shape],
            'acts_buf': [acts_buf_shared, acts_buf_shared_data_dtype, acts_buf_shared_data_shape],
            'rews_buf': [rews_buf_shared, rews_buf_shared_data_dtype, rews_buf_shared_data_shape],
            'done_buf': [done_buf_shared, done_buf_shared_data_dtype, done_buf_shared_data_shape],
            'ptr': ptr_shared,
            'size': size_shared,
            'max_size': size,
            'batch_size': batch_size}


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, shared_data):
        self.max_size, self.batch_size = shared_data['max_size'], shared_data['batch_size']

        # shared memory
        self.obs_buf_shared, self.obs_buf_shared_data_dtype, self.obs_buf_shared_data_shape = shared_data['obs_buf']
        self.next_obs_buf_shared, self.next_obs_buf_shared_data_dtype, self.next_obs_buf_shared_data_shape = \
            shared_data['next_obs_buf']
        self.acts_buf_shared, self.acts_buf_shared_data_dtype, self.acts_buf_shared_data_shape = shared_data['acts_buf']
        self.rews_buf_shared, self.rews_buf_shared_data_dtype, self.rews_buf_shared_data_shape = shared_data['rews_buf']
        self.done_buf_shared, self.done_buf_shared_data_dtype, self.done_buf_shared_data_shape = shared_data['done_buf']
        self._ptr_shared = shared_data['ptr']
        self._size_shared = shared_data['size']

    @staticmethod
    def create_np_array_from_shared_mem(
            shared_mem: SharedMemory, shared_data_dtype: np.dtype, shared_data_shape: Tuple[int, ...]
    ) -> np.ndarray:
        arr = np.ndarray(shared_data_shape, dtype=shared_data_dtype, buffer=shared_mem.buf)
        return arr

    @property
    def obs_buf(self):
        return self.create_np_array_from_shared_mem(shared_mem=self.obs_buf_shared,
                                                    shared_data_dtype=self.obs_buf_shared_data_dtype,
                                                    shared_data_shape=self.obs_buf_shared_data_shape)

    @property
    def next_obs_buf(self):
        return self.create_np_array_from_shared_mem(shared_mem=self.next_obs_buf_shared,
                                                    shared_data_dtype=self.next_obs_buf_shared_data_dtype,
                                                    shared_data_shape=self.next_obs_buf_shared_data_shape)

    @property
    def acts_buf(self):
        return self.create_np_array_from_shared_mem(shared_mem=self.acts_buf_shared,
                                                    shared_data_dtype=self.acts_buf_shared_data_dtype,
                                                    shared_data_shape=self.acts_buf_shared_data_shape)

    @property
    def rews_buf(self):
        return self.create_np_array_from_shared_mem(shared_mem=self.rews_buf_shared,
                                                    shared_data_dtype=self.rews_buf_shared_data_dtype,
                                                    shared_data_shape=self.rews_buf_shared_data_shape)

    @property
    def done_buf(self):
        return self.create_np_array_from_shared_mem(shared_mem=self.done_buf_shared,
                                                    shared_data_dtype=self.done_buf_shared_data_dtype,
                                                    shared_data_shape=self.done_buf_shared_data_shape)

    @property
    def ptr(self):
        return self._ptr_shared.value

    @property
    def size(self):
        return self._size_shared.value

    @ptr.setter
    def ptr(self, val):
        self._ptr_shared.value = val

    @size.setter
    def size(self, val):
        self._size_shared.value = val

    def store(
            self,
            obs: np.ndarray,
            act: int,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size


class SharedReplayBuffer(object):
    @staticmethod
    def perm_gpu(pop_size, num_samples, device):
        """Use torch.randperm to generate indices on a GPU tensor."""
        return torch.randperm(pop_size, device=device)[:num_samples]

    def __init__(self, obs_dim: list, size: int, batch_size: int = 32, device="cuda"):
        self.obs_buf = torch.zeros([size, *obs_dim], dtype=torch.float32, device=device)
        self.next_obs_buf = torch.zeros([size, *obs_dim], dtype=torch.float32, device=device)
        self.acts_buf = torch.zeros([size], dtype=torch.int64, device=device)
        self.rews_buf = torch.zeros([size], dtype=torch.float32, device=device)
        self.done_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.ptr = torch.tensor([0], dtype=torch.int64, device=device)
        self.size = torch.tensor([0], dtype=torch.int64, device=device)
        self.device = device
        self.max_size, self.batch_size = size, batch_size

        self.obs_buf.share_memory_()
        self.next_obs_buf.share_memory_()
        self.acts_buf.share_memory_()
        self.rews_buf.share_memory_()
        self.done_buf.share_memory_()
        self.ptr.share_memory_()
        self.size.share_memory_()

        self.lock = mp.Lock()

    def __len__(self) -> int:
        return self.size.detach().cpu().numpy().item()

    def sample_batch(self) -> Dict[str, torch.tensor]:
        """
        Randomly samples a number of experiences from the stored buffer
        Args:
            batch_size: number of experiences to sample
        Returns:
            a batch of tuple np arrays of state, action, reward, done, next_state
        """
        idxs = self.perm_gpu(pop_size=self.size[0], num_samples=self.batch_size,
                             device=self.device)  # equal to np.random.choice(self.size[0], size=self.batch_size, replace=False)
        with self.lock:
            return dict(obs=self.obs_buf[idxs],
                        next_obs=self.next_obs_buf[idxs],
                        acts=self.acts_buf[idxs],
                        rews=self.rews_buf[idxs],
                        done=self.done_buf[idxs])

    def store(
            self,
            obs: np.ndarray,
            act: int,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
    ):
        device = self.device
        with self.lock:
            self.obs_buf[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=device)
            self.next_obs_buf[self.ptr] = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            self.acts_buf[self.ptr] = torch.as_tensor(act, dtype=torch.int64, device=device)
            self.rews_buf[self.ptr] = torch.as_tensor(rew, dtype=torch.float32, device=device)
            self.done_buf[self.ptr] = torch.as_tensor(done, dtype=torch.float32, device=device)
            self.ptr[0] = (self.ptr[0] + 1) % self.max_size
            self.size[0] = min(self.size[0] + 1, self.max_size)


class SharedPERBuffer(SharedReplayBuffer):

    def __init__(self, obs_dim: list,
                 size: int,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_decay_steps: int = 100000,
                 batch_size: int = 32,
                 device="cuda"):
        super(SharedPERBuffer, self).__init__(obs_dim, size, batch_size, device)

        self.priorities = torch.zeros((size,), dtype=torch.float32, device=device)

        self.alpha = alpha
        self.beta_start = beta
        # ATTENTION: When using the multirpocessing, copy the weight to the gpu is a zero operation ( all weights are set to 0)
        self.beta = torch.tensor([0], dtype=torch.float32, device=device)  # please update beta in the subprocess!
        self.beta_decay_steps = beta_decay_steps

        self.priorities.share_memory_()
        self.beta.share_memory_()

    def update_beta(self, step):
        """Update the beta value which accounts for the bias in the PER.
                Args:
                    step: current global step
                Returns:
                    beta value for this indexed experience
        """
        beta_val = self.beta_start + step * (1.0 - self.beta_start) / self.beta_decay_steps
        with self.lock:
            self.beta[0] = min(1.0, beta_val)

        return self.beta

    def sample_batch(self) -> Dict[str, torch.tensor]:
        """
        Randomly samples a number of experiences from the stored buffer
        Args:
            batch_size: number of experiences to sample
        Returns:
            a batch of tuple np arrays of state, action, reward, done, next_state
        """
        if self.size[0] == self.max_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.size[0]]

        # probability to the power of alpha to weight how important that probability it, 0 = normal distirbution
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = probs.multinomial(num_samples=self.batch_size, replacement=False)

        weights = (self.size[0] * probs[indices]) ** (-self.beta[0])
        weights /= weights.max()

        with self.lock:
            return dict(obs=self.obs_buf[indices],
                        next_obs=self.next_obs_buf[indices],
                        acts=self.acts_buf[indices],
                        rews=self.rews_buf[indices],
                        done=self.done_buf[indices],
                        weights=weights,
                        indices=indices,
                        )

    def store(
            self,
            obs: np.ndarray,
            act: int,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
    ):
        device = self.device
        with self.lock:
            self.obs_buf[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=device)
            self.next_obs_buf[self.ptr] = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            self.acts_buf[self.ptr] = torch.as_tensor(act, dtype=torch.int64, device=device)
            self.rews_buf[self.ptr] = torch.as_tensor(rew, dtype=torch.float32, device=device)
            self.done_buf[self.ptr] = torch.as_tensor(done, dtype=torch.float32, device=device)

            # the priority for the latest sample is set to max priority so it will be resampled soon
            max_prio = self.priorities.max() if self.ptr[0] > 0 else 1.0
            self.priorities[self.ptr] = torch.as_tensor(max_prio, dtype=torch.float32, device=device)

            self.ptr[0] = (self.ptr[0] + 1) % self.max_size
            self.size[0] = min(self.size[0] + 1, self.max_size)

    def update_priorities(self, batch_indices: List, batch_priorities: List) -> None:
        """Update the priorities from the last batch, this should be called after the loss for this batch has been
        calculated.
        Args:
            batch_indices: index of each datum in the batch
            batch_priorities: priority of each datum in the batch
        """
        with self.lock:
            self.priorities[batch_indices] = batch_priorities


class HERSharedReplayBuffer(object):
    """
    Hindsight Experience replay
    """

    @staticmethod
    def perm_gpu(pop_size, num_samples, device):
        """Use torch.randperm to generate indices on a GPU tensor."""
        return torch.randperm(pop_size, device=device)[:num_samples]

    def __init__(self, obs_dim: list, size: int, batch_size: int = 32, device="cuda"):
        self.goal_buf = torch.zeros([size, *obs_dim], dtype=torch.float32, device=device)
        self.obs_buf = torch.zeros([size, *obs_dim], dtype=torch.float32, device=device)
        self.next_obs_buf = torch.zeros([size, *obs_dim], dtype=torch.float32, device=device)
        self.acts_buf = torch.zeros([size], dtype=torch.int64, device=device)
        self.rews_buf = torch.zeros([size], dtype=torch.float32, device=device)
        self.done_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.ptr = torch.tensor([0], dtype=torch.int64, device=device)
        self.size = torch.tensor([0], dtype=torch.int64, device=device)
        self.device = device
        self.max_size, self.batch_size = size, batch_size

        self.goal_buf.share_memory_()
        self.obs_buf.share_memory_()
        self.next_obs_buf.share_memory_()
        self.acts_buf.share_memory_()
        self.rews_buf.share_memory_()
        self.done_buf.share_memory_()
        self.ptr.share_memory_()
        self.size.share_memory_()

        self.lock = mp.Lock()

    def __len__(self) -> int:
        return self.size.detach().cpu().numpy().item()

    def sample_batch(self) -> Dict[str, torch.tensor]:
        """
        Randomly samples a number of experiences from the stored buffer
        Args:
            batch_size: number of experiences to sample
        Returns:
            a batch of tuple np arrays of state, action, reward, done, next_state
        """
        idxs = self.perm_gpu(pop_size=self.size[0], num_samples=self.batch_size,
                             device=self.device)  # equal to np.random.choice(self.size[0], size=self.batch_size, replace=False)
        with self.lock:
            return dict(goal=self.goal_buf[idxs],
                        obs=self.obs_buf[idxs],
                        next_obs=self.next_obs_buf[idxs],
                        acts=self.acts_buf[idxs],
                        rews=self.rews_buf[idxs],
                        done=self.done_buf[idxs])

    def store(self,
              goal: np.ndarray,
              obs: np.ndarray,
              act: int,
              rew: float,
              next_obs: np.ndarray,
              done: bool,
              ):
        device = self.device
        with self.lock:
            self.goal_buf[self.ptr] = torch.as_tensor(goal, dtype=torch.float32, device=device)
            self.obs_buf[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=device)
            self.next_obs_buf[self.ptr] = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            self.acts_buf[self.ptr] = torch.as_tensor(act, dtype=torch.int64, device=device)
            self.rews_buf[self.ptr] = torch.as_tensor(rew, dtype=torch.float32, device=device)
            self.done_buf[self.ptr] = torch.as_tensor(done, dtype=torch.float32, device=device)
            self.ptr[0] = (self.ptr[0] + 1) % self.max_size
            self.size[0] = min(self.size[0] + 1, self.max_size)


class SharedHERPERBuffer(HERSharedReplayBuffer):
    def __init__(self, obs_dim: list,
                 size: int,
                 alpha: float = 0.6,
                 beta: float = 0.6,
                 beta_decay_steps: int = 100000,
                 batch_size: int = 32,
                 device="cuda"):
        super(SharedHERPERBuffer, self).__init__(obs_dim, size, batch_size, device)

        self.priorities = torch.zeros((size,), dtype=torch.float32, device=device)

        self.alpha = alpha
        self.beta_start = beta
        # ATTENTION: When using the multirpocessing, copy the weight to the gpu is a zero operation ( all weights are set to 0)
        self.beta = torch.tensor([0], dtype=torch.float32, device=device)  # please update beta in the subprocess!
        self.beta_decay_steps = beta_decay_steps

        self.priorities.share_memory_()
        self.beta.share_memory_()

    def update_beta(self, step):
        """Update the beta value which accounts for the bias in the PER.
                Args:
                    step: current global step
                Returns:
                    beta value for this indexed experience
        """
        beta_val = self.beta_start + step * (1.0 - self.beta_start) / self.beta_decay_steps
        with self.lock:
            self.beta[0] = min(1.0, beta_val)

        return self.beta

    def sample_batch(self) -> Dict[str, torch.tensor]:
        """
        Randomly samples a number of experiences from the stored buffer
        Args:
            batch_size: number of experiences to sample
        Returns:
            a batch of tuple np arrays of state, action, reward, done, next_state
        """
        with self.lock:
            if self.size[0] == self.max_size:
                prios = self.priorities
            else:
                prios = self.priorities[:self.size[0]]
            # probability to the power of alpha to weight how important that probability it, 0 = normal distirbution
            probs = prios ** self.alpha
            probs /= probs.sum()
            indices = probs.multinomial(num_samples=self.batch_size, replacement=False)
            weights = (self.size[0] * probs[indices]) ** (-self.beta[0])
            weights /= weights.max()
            return dict(goal=self.goal_buf[indices],
                        obs=self.obs_buf[indices],
                        next_obs=self.next_obs_buf[indices],
                        acts=self.acts_buf[indices],
                        rews=self.rews_buf[indices],
                        done=self.done_buf[indices],
                        weights=weights,
                        indices=indices, )

    def store(
            self,
            goal: np.ndarray,
            obs: np.ndarray,
            act: int,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
            **kwargs,
    ):
        device = self.device
        with self.lock:
            self.goal_buf[self.ptr] = torch.as_tensor(goal, dtype=torch.float32, device=device)
            self.obs_buf[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=device)
            self.next_obs_buf[self.ptr] = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            self.acts_buf[self.ptr] = torch.as_tensor(act, dtype=torch.int64, device=device)
            self.rews_buf[self.ptr] = torch.as_tensor(rew, dtype=torch.float32, device=device)
            self.done_buf[self.ptr] = torch.as_tensor(done, dtype=torch.float32, device=device)

            # if rew > 3:
            #     print(goal)
            #     print(obs)
            #     print(next_obs)

            # the priority for the latest sample is set to max priority so it will be resampled soon
            max_prio = self.priorities.max() if self.ptr[0] > 0 else 1.0
            self.priorities[self.ptr] = torch.as_tensor(max_prio, dtype=torch.float32, device=device)

            self.ptr[0] = (self.ptr[0] + 1) % self.max_size
            self.size[0] = min(self.size[0] + 1, self.max_size)

    def update_priorities(self, batch_indices: List, batch_priorities: List) -> None:
        """Update the priorities from the last batch, this should be called after the loss for this batch has been
        calculated.
        Args:
            batch_indices: index of each datum in the batch
            batch_priorities: priority of each datum in the batch
        """
        with self.lock:
            self.priorities[batch_indices] = batch_priorities


class SharedHERPERBuffer2(SharedHERPERBuffer):
    def __init__(self, obs_dim: list,
                 size: int,
                 alpha: float = 0.6,
                 beta: float = 0.6,
                 beta_decay_steps: int = 100000,
                 batch_size: int = 32,
                 device="cuda"):
        super(SharedHERPERBuffer2, self).__init__(obs_dim, size, alpha, beta, beta_decay_steps, batch_size, device)

        action_dim = np.prod(obs_dim)
        compressed_abs_obs_dim = int((action_dim - 1) * action_dim / 2) + action_dim
        self.compressed_abs_obs_buf = torch.zeros((size, compressed_abs_obs_dim), dtype=torch.float32, device=device)
        self.compressed_abs_next_obs_buf = torch.zeros((size, compressed_abs_obs_dim), dtype=torch.float32,
                                                       device=device)

        self.compressed_abs_obs_buf.share_memory_()
        self.compressed_abs_next_obs_buf.share_memory_()

    def sample_batch(self) -> Dict[str, torch.tensor]:
        """
        Randomly samples a number of experiences from the stored buffer
        Args:
            batch_size: number of experiences to sample
        Returns:
            a batch of tuple np arrays of state, action, reward, done, next_state
        """
        with self.lock:
            if self.size[0] == self.max_size:
                prios = self.priorities
            else:
                prios = self.priorities[:self.size[0]]
            # probability to the power of alpha to weight how important that probability it, 0 = normal distirbution
            probs = prios ** self.alpha
            probs /= probs.sum()
            indices = probs.multinomial(num_samples=self.batch_size, replacement=False)
            weights = (self.size[0] * probs[indices]) ** (-self.beta[0])
            weights /= weights.max()
            return dict(goal=self.goal_buf[indices],
                        obs=self.obs_buf[indices],
                        next_obs=self.next_obs_buf[indices],
                        c_abs_obs=self.compressed_abs_obs_buf[indices],
                        c_abs_next_obs=self.compressed_abs_next_obs_buf[indices],
                        acts=self.acts_buf[indices],
                        rews=self.rews_buf[indices],
                        done=self.done_buf[indices],
                        weights=weights,
                        indices=indices, )

    def store(
            self,
            goal: np.ndarray,
            obs: np.ndarray,
            c_abs_obs: np.ndarray,
            act: int,
            rew: float,
            next_obs: np.ndarray,
            c_abs_next_obs: np.ndarray,
            done: bool,
    ):
        device = self.device
        with self.lock:
            self.goal_buf[self.ptr] = torch.as_tensor(goal, dtype=torch.float32, device=device)
            self.obs_buf[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=device)
            self.next_obs_buf[self.ptr] = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            self.acts_buf[self.ptr] = torch.as_tensor(act, dtype=torch.int64, device=device)
            self.rews_buf[self.ptr] = torch.as_tensor(rew, dtype=torch.float32, device=device)
            self.done_buf[self.ptr] = torch.as_tensor(done, dtype=torch.float32, device=device)
            self.compressed_abs_obs_buf[self.ptr] = torch.as_tensor(c_abs_obs, dtype=torch.float32, device=device)
            self.compressed_abs_next_obs_buf[self.ptr] = torch.as_tensor(c_abs_next_obs, dtype=torch.float32,
                                                                         device=device)

            # if rew > 3:
            #     print(goal)
            #     print(obs)
            #     print(next_obs)

            # the priority for the latest sample is set to max priority so it will be resampled soon
            max_prio = self.priorities.max() if self.ptr[0] > 0 else 1.0
            self.priorities[self.ptr] = torch.as_tensor(max_prio, dtype=torch.float32, device=device)

            self.ptr[0] = (self.ptr[0] + 1) % self.max_size
            self.size[0] = min(self.size[0] + 1, self.max_size)


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.

    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight

    """

    def __init__(
            self,
            shared_data,
            alpha: float = 0.6
    ):
        """Initialization."""
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(shared_data=shared_data)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(
            self,
            obs: np.ndarray,
            act: int,
            rew: float,
            next_obs: np.ndarray,
            done: bool
    ):
        """Store experience and priority."""
        super().store(obs, act, rew, next_obs, done)

        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight


class SegmentTree:
    """ Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    Attributes:
        capacity (int)
        tree (list)
        operation (function)

    """

    def __init__(self, capacity: int, operation: Callable, init_value: float):
        """Initialization.

        Args:
            capacity (int)
            operation (function)
            init_value (float)

        """
        assert (
                capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(
            self, start: int, end: int, node: int, node_start: int, node_end: int
    ) -> float:
        """Returns result of operation in segment."""
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def operate(self, start: int = 0, end: int = 0) -> float:
        """Returns result of applying `self.operation`."""
        if end <= 0:
            end += self.capacity
        end -= 1

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float):
        """Set value in tree."""
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    """ Create SumSegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialization.

        Args:
            capacity (int)

        """
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add, init_value=0.0
        )

    def sum(self, start: int = 0, end: int = 0) -> float:
        """Returns arr[start] + ... + arr[end]."""
        return super(SumSegmentTree, self).operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        """Find the highest index `i` about upper bound in the tree"""
        # TODO: Check assert case and fix bug
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1

        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    """ Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialization.

        Args:
            capacity (int)

        """
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=min, init_value=float("inf")
        )

    def min(self, start: int = 0, end: int = 0) -> float:
        """Returns min(arr[start], ...,  arr[end])."""
        return super(MinSegmentTree, self).operate(start, end)


def test_p(replay):
    """Logic to be executed by the child process"""
    for i in range(1):
        replay.store(np.random.random((5, 10)), np.random.randint(2500), rew=np.random.random() * 50,
                     next_obs=np.random.random((5, 10)), done=np.random.randint(2))
        time.sleep(.5)
    print(replay.obs_buf[0][0][:3])


if __name__ == "__main__":
    shared_data = gen_replay_buffer_shared_memo(obs_dim=(5, 10), size=50)

    replay = ReplayBuffer(shared_data)
    pp = []
    for _ in range(10):
        p = mp.Process(target=test_p, args=(replay,))
        pp.append(p)
        p.start()
    b = time.time()

    while True:
        replay.store(np.random.random((5, 10)), np.random.randint(2500), rew=np.random.random() * 50,
                     next_obs=np.random.random((5, 10)), done=np.random.randint(2))
        time.sleep(.5)
        # print(np.ndarray(shared_data['obs_buf'][0], dtype=shared_data['obs_buf'][1], buffer=shared_data['obs_buf'][2]))
        if replay.obs_buf[-1][0][0] != 0:
            a = time.time()
            print("FINISHED", a - b)
            exit(0)
# 89.65514898300171
