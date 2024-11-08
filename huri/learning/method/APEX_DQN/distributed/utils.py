""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20231026osaka

"""
import numpy as np
import torch
import copy
from ray.rllib.policy.sample_batch import SampleBatch
from huri.learning.env.rack_v3.env import RackState, RackStatePlot


# import psutil, gc


# def abs_state(state: torch.tensor, goal: torch.tensor, env_classes: int, feasible_category):
#     arrange = state[:, None, ...].clone()
#     arrange[arrange > 0] = 1
#     state_onehot = to_onehot(state[:, None, ...], env_classes)
#     goal_onehot = to_onehot(goal[:, None, ...], env_classes)
#     state_f1 = torch.cat((state_onehot[:, [0]], goal_onehot[:, [0]], goal_onehot[:, [1]], arrange), axis=1)
#     state_f2 = torch.cat((state_onehot[:, [1]], goal_onehot[:, [1]], goal_onehot[:, [0]], arrange), axis=1)
#     return state_f1, state_f2, feasible_category

# def auto_garbage_collect(pct=10.0):
#     """
#     auto_garbage_collection - Call the garbage collection if memory used is greater than 80% of total available memory.
#                               This is called to deal with an issue in Ray not freeing up used memory.
#
#         pct - Default value of 80%.  Amount of memory in use that triggers the garbage collection call.
#     """
#     if psutil.virtual_memory().percent >= pct:
#         gc.collect()
#         print("Clear memory")
#         print("--------:", psutil.virtual_memory().percent)
#     return
#

def dummy_abs_state(state: np.ndarray, feasible_category, device):
    state_f = torch.as_tensor(state, dtype=torch.float32, device=device)
    # state_f1, state_f2 = state_f[:, 0], state_f[:, 1]
    # state_f1 = state_f[:, 0]

    return [state_f[:, i][:, :2] for i in range(state_f.shape[1])], feasible_category
    # return state_f1[:, :2], state_f1[:, 2:], feasible_category
    # return state_f1, state_f2, feasible_category


def abs_state_np(state: np.ndarray, goal: np.ndarray, env_classes: int):
    """
    Suppose we have a state and a goal, we want to convert them into a 4xNxM tensor.
    The first two channels are the one-hot representation of the state of a category and its target.
    The third channel is the one-hot representation of the goal of other categories.
    The fourth channel is the arrangement of the state.
    :param state: NxM numpy array or 1xNxM numpy array
    :param goal: NxM numpy array or 1xNxM numpy array
    :param env_classes: number of classes in the environment
    :return: 4xNxM numpy array
    """
    state, goal = np.array(state), np.array(goal)
    if len(state.shape) == 2:
        state = state.reshape((1, *state.shape))
    if len(goal.shape) == 2:
        goal = goal.reshape((1, *goal.shape))
    arrange = state.copy()
    arrange[arrange > 0] = 1
    arrange_goal = goal.copy()
    arrange_goal[arrange_goal > 0] = 1
    state_onehot = to_onehot_np(state, env_classes)
    goal_onehot = to_onehot_np(goal, env_classes)
    features = []
    for i in range(env_classes):
        # goal of other categories
        # other_goal = np.sum(np.concatenate((goal_onehot[:i],
        #                                     goal_onehot[i + 1:]), axis=0), axis=0)[None, ...]
        # features.append(np.concatenate((state_onehot[[i]],
        #                                 goal_onehot[[i]],
        #                                 other_goal,
        #                                 arrange), axis=0))
        # features.append(np.concatenate((state_onehot[[i]],
        #                                 goal_onehot[[i]],
        #                                 arrange,
        #                                 arrange_goal,), axis=0))
        features.append(np.concatenate((state_onehot[[i]],
                                        goal_onehot[[i]],), axis=0))

    return features


def to_onehot(tensor, num_categories=2):
    """
    Convert a batch of 2D tensors with integer categories to one-hot representation.

    Parameters:
    - tensor: A (batch, 1, N, M) torch Tensor.
    - num_categories: Total number of categories.

    Returns:
    - A (batch, N, M, num_categories) torch Tensor in one-hot format.
    """

    # Get the shape of the input tensor
    batch_size, _, N, M = tensor.shape

    # Create a one-hot tensor of zeros of appropriate size
    # one_hot = torch.zeros(batch_size, N, M, num_categories, device=tensor.device)
    one_hot = torch.zeros(batch_size, num_categories + 1, N, M, dtype=tensor.dtype, device=tensor.device)

    # Fill in the ones at the right indices
    one_hot.scatter_(1, tensor.long(), 1)

    return one_hot[:, 1:]


def to_onehot_np(x, num_categories=2):
    # Create a one-hot array of zeros of appropriate size
    one_hot = np.zeros((num_categories + 1, x.shape[-2], x.shape[-1]))
    # Use fancy indexing to fill the right spots with ones
    one_hot[x, np.arange(x.shape[-2])[:, None], np.arange(x.shape[-1])] = 1
    return one_hot[1:]


def swap_values(arr, k):
    """
    Swap all entries of a batchxmxn size numpy array with values from 1 to k.

    Parameters:
    - arr: numpy array of size batchxmxn
    - k: maximum possible value in arr (exclusive of 0)

    Returns:
    - A numpy array with swapped values
    """

    # generate a permutation of values from 1 to k
    original = np.arange(1, k + 1)
    perm = np.random.permutation(original)
    while np.all(perm == original):
        perm = np.random.permutation(original)

    # create a direct mapping array for replacement
    mapping = np.arange(k + 1)  # include 0 to k
    mapping[1:k + 1] = perm

    # swap values efficiently using advanced indexing
    return mapping[arr]


def swap_values_torch(arr, k):
    """
    Swap all entries of a batchxmxn size torch tensor with values from 1 to k.

    Parameters:
    - arr: torch tensor of size batchxmxn
    - k: maximum possible value in arr (exclusive of 0)

    Returns:
    - A torch tensor with swapped values
    """

    # generate a permutation of values from 1 to k
    original = torch.arange(1, k + 1)
    perm = original[torch.randperm(k)]
    while torch.all(perm == original):
        perm = original[torch.randperm(k)]

    # create a direct mapping tensor for replacement
    mapping = torch.arange(k + 1)  # include 0 to k
    mapping[1:k + 1] = perm

    # swap values efficiently using advanced indexing
    return mapping[arr.long()]  # using .long() to ensure the indices are long integers


def synthetic_traj_category(states_np, goal_np, toggle_debug=False):
    uniq_class = np.unique(states_np)
    uniq_class = uniq_class[uniq_class > 0]
    if len(uniq_class) <= 1:
        return None
    state_list = swap_values(np.concatenate((states_np, goal_np[None,]), axis=0), k=len(uniq_class))
    states_swap = state_list[:-1]
    goal_swap = state_list[-1]
    if toggle_debug:
        rsp = RackStatePlot(goal_swap, )
        plot = rsp.plot_states(states_swap, row=15).get_img()
        cv2.imshow(f"plot_swap", plot)

        rsp = RackStatePlot(goal_np, )
        plot = rsp.plot_states(states_np, row=15).get_img()
        cv2.imshow(f"plot_no_swap", plot)
        cv2.waitKey(0)
    return (states_swap, goal_swap)


def synthetic_traj_direction(states_np, goal_np, toggle_debug=False):
    states_swap = states_np[::-1]
    goal_swap = np.asarray(states_np[0])
    if toggle_debug:
        rsp = RackStatePlot(goal_swap, )
        plot = rsp.plot_states(states_swap, row=15).get_img()
        cv2.imshow(f"plot_swap", plot)

        rsp = RackStatePlot(goal_np, )
        plot = rsp.plot_states(states_np, row=15).get_img()
        cv2.imshow(f"plot_no_swap", plot)
        cv2.waitKey(0)
    return (states_swap, goal_swap)


def synthetic_traj_reflection_ud(states_np, goal_np, toggle_debug=False):
    states_swap = np.flip(states_np, axis=1)
    goal_swap = np.flipud(goal_np)

    if toggle_debug:
        rsp = RackStatePlot(goal_swap, )
        plot = rsp.plot_states(states_swap, row=15).get_img()
        cv2.imshow(f"plot_swap", plot)

        rsp = RackStatePlot(goal_np, )
        plot = rsp.plot_states(states_np, row=15).get_img()
        cv2.imshow(f"plot_no_swap", plot)
        cv2.waitKey(0)
    return (states_swap, goal_swap)


def synthetic_traj_reflection_lr(states_np, goal_np, toggle_debug=False):
    states_swap = np.flip(states_np, axis=2)
    goal_swap = np.fliplr(goal_np)
    if toggle_debug:
        rsp = RackStatePlot(goal_swap, )
        plot = rsp.plot_states(states_swap, row=15).get_img()
        cv2.imshow(f"plot_swap", plot)

        rsp = RackStatePlot(goal_np, )
        plot = rsp.plot_states(states_np, row=15).get_img()
        cv2.imshow(f"plot_no_swap", plot)
        cv2.waitKey(0)
    return (states_swap, goal_swap)


def synthetic_traj_reflection_lrud(states_np, goal_np, toggle_debug=False):
    states_swap = np.flip(states_np, axis=(1, 2))  # flip up-down and left-right
    goal_swap = np.flip(goal_np, axis=(0, 1))  # flip along both axes
    if toggle_debug:
        rsp = RackStatePlot(goal_swap, )
        plot = rsp.plot_states(states_swap, row=15).get_img()
        cv2.imshow(f"plot_swap", plot)

        rsp = RackStatePlot(goal_np, )
        plot = rsp.plot_states(states_np, row=15).get_img()
        cv2.imshow(f"plot_no_swap", plot)
        cv2.waitKey(0)
    return (states_swap, goal_swap)


def padding(size, array):
    if len(array) == 0:
        return np.zeros(size, dtype=int)
    max_v = max(array)
    pad_array = np.ones(size, dtype=int) * max_v
    pad_array[:len(array)] = array
    return pad_array


def one_hot_vector(action_dim, indices):
    vector = np.zeros(action_dim, dtype=float)
    if len(indices) > 0:
        vector[indices] = 1
    return vector


def transform(x, num_categories=2):
    # Create a one-hot array of zeros of appropriate size
    one_hot = np.zeros((num_categories + 1, x.shape[0], x.shape[1]))
    # Use fancy indexing to fill the right spots with ones
    one_hot[x, np.arange(x.shape[0])[:, None], np.arange(x.shape[1])] = 1
    return one_hot[1:]


def category_feasible_action(state, action_dim, num_category):
    state_onehot = to_onehot_np(np.array(state), num_category)
    possible_action_set = RackState(state).feasible_action_set
    c_possible_action_set_list = []
    for state_onehot_c in state_onehot:
        c_possible_action_set = RackState(state_onehot_c).feasible_action_set
        c_possible_action_set = c_possible_action_set[np.in1d(c_possible_action_set, possible_action_set)]
        c_possible_action_set_list.append(one_hot_vector(action_dim, c_possible_action_set))
    return c_possible_action_set_list


class Trajectory(object):

    def __init__(self, goal,
                 action_dim,
                 n_step=1,
                 gamma=1.0,
                 num_categories=2):
        self.goal = goal
        self.n_step = n_step  # Number of steps for multi-step return
        self.action_dim = action_dim
        self.gamma = gamma  # Discount factor
        self.num_categories = num_categories
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        # --
        # self.abs_states = []
        # self.abs_next_states = []
        # self.next_state_feasible_actions = []
        # self.state_feasible_categories = []
        # self.next_state_feasible_categories = []

    def add_transition(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        # ----
        # if len(self.abs_next_states) > 0:
        #     self.abs_states.append(self.abs_next_states[-1])
        #     self.state_feasible_categories.append(self.next_state_feasible_categories[-1])
        # else:
        #     self.abs_states.append(abs_state_np(state, self.goal, self.num_categories))
        #     self.state_feasible_categories.append(
        #         category_feasible_action(state, self.action_dim, self.num_categories))
        # self.abs_next_states.append(abs_state_np(next_state, self.goal, self.num_categories))
        # self.next_state_feasible_actions.append(
        #     padding(self.action_dim, RackState(next_state).feasible_action_set))
        # self.next_state_feasible_categories.append(
        #     category_feasible_action(next_state, self.action_dim, self.num_categories))

    def compute_n_step_return(self, start_index, end_index):
        """
        Compute the n-step return for a given range of indices.
        :param start_index:
        :param end_index:
        :return:
        """
        discount = 1.0  # Initial discount factor
        # rewards = self.rewards[start_index:end_index]
        # G = sum(rewards)
        G = .0
        for i in range(start_index, end_index):
            G += self.rewards[i] * discount
            discount *= self.gamma  # Apply gamma for the next step
            if self.dones[i]:  # If episode ended, no need to look further
                break
        # Determine the next state based on the range of indices
        next_state_index = end_index - 1  # Default to last index in range
        if end_index < len(self.states):
            next_state = self.next_states[next_state_index]
        else:
            next_state = self.next_states[-1]
        # Check if done signal occurs in the range
        done = any(self.dones[start_index:end_index])
        return G, next_state, done

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        if isinstance(index, slice):
            # start, stop, step = index.indices(len(self.states))
            # if len(self.abs_states) > 0:
            #     states = self.abs_states[index]
            #     next_states = self.abs_next_states[index]
            #     next_state_feasible_actions = self.next_state_feasible_actions[index]
            #     state_feasible_categories = self.state_feasible_categories[index]
            #     next_state_feasible_categories = self.next_state_feasible_categories[index]
            #     actions = self.actions[index]
            #     rewards = self.rewards[index]
            #     dones = self.dones[index]
            # else:
            start, stop, step = index.indices(len(self.states))
            states = []
            next_states = []
            next_state_feasible_actions = []
            state_feasible_categories = []
            next_state_feasible_categories = []
            actions = self.actions[index]
            rewards = self.rewards[index]
            dones = self.dones[index]
            # goals = []
            for i in range(start, stop, step if step else 1):
                # end_idx = min(i + self.n_step, len(self.states))
                # reward, next_state, done = self.compute_n_step_return(i, end_idx)
                # for i in range(len(self.states)):
                if len(next_states) > 0:
                    states.append(next_states[-1])
                    state_feasible_categories.append(next_state_feasible_categories[-1])
                else:
                    states.append(abs_state_np(self.states[i], self.goal, self.num_categories))
                    state_feasible_categories.append(
                        category_feasible_action(self.states[i], self.action_dim, self.num_categories))
                next_states.append(abs_state_np(self.next_states[i], self.goal, self.num_categories))
                next_state_feasible_actions.append(
                    padding(int(self.action_dim*0.575), RackState(self.next_states[i]).feasible_action_set))
                next_state_feasible_categories.append(
                    category_feasible_action(self.next_states[i], self.action_dim, self.num_categories))
                # actions.append(self.actions[i])
                # rewards.append(self.rewards[i])
                # dones.append(self.dones[i])
                # goals.append(self.goal)
            # states = self.states[index]
            # actions = self.actions[index]
            # rewards = self.rewards[index]
            # next_states = self.next_states[index]
            # dones = self.dones[index]
            return SampleBatch({'state': states,
                                'action': actions,
                                'reward': rewards,
                                'next_state': next_states,
                                'next_state_feasible_action': next_state_feasible_actions,
                                'state_feasible_category': state_feasible_categories,
                                'next_state_feasible_category': next_state_feasible_categories,
                                'done': dones,
                                # 'goal': goals,
                                })
        else:
            # state = self.states[index]
            # action = self.actions[index]
            # reward = self.rewards[index]
            # next_state = self.next_states[index]
            # done = self.dones[index]
            # end_idx = min(index + self.n_step, len(self.states))
            # reward, next_state, done = self.compute_n_step_return(index, end_idx)
            state = abs_state_np(self.states[index], self.goal, self.num_categories)
            next_state = abs_state_np(self.next_states[index], self.goal, self.num_categories)
            reward = self.rewards[index]
            action = self.actions[index]
            done = self.dones[index]
            state_feasible_category = category_feasible_action(state, self.action_dim, self.num_categories)
            next_state_feasible_category = category_feasible_action(next_state, self.action_dim, self.num_categories)

            return SampleBatch({'state': [state],
                                'action': [action],
                                'reward': [reward],
                                'next_state': [next_state],
                                'next_state_feasible_action': [
                                    padding(int(self.action_dim*0.575), RackState(self.next_states[index]).feasible_action_set)],
                                'state_feasible_category': [state_feasible_category],
                                'next_state_feasible_category': [next_state_feasible_category],
                                'done': [done],
                                # 'goal': [self.goal]
                                })

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return 'Trajectory(len={})'.format(len(self))


if __name__ == '__main__':
    tensor = torch.tensor([[[[1, 2], [0, 1]]]], device="cuda:0", dtype=torch.float32)
    print(tensor)
    print(tensor.shape)
    num_categories = 2
    print(to_onehot(tensor, num_categories))
    import huri.core.file_sys as fs
    from huri.learning.env.rack_v3.env import RackArrangementEnv
    from huri.learning.method.APEX_DQN.distributed.reanalyzer import refined_path_to_transitions_recur
    import cv2

    traj = fs.load_pickle('./test/trajectory.pkl')
    print(synthetic_traj_category(np.array([*traj.states, traj.next_states[-1]]), traj.goal, toggle_debug=True))
