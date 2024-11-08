import logging
import collections

import matplotlib.pyplot as plt
import torch
import numpy as np

from huri.components.utils.matlibplot_utils import Plot
from huri.learning.network.d3qn import DuelingDQN


class Node():
    env = None
    lr = None
    net = None
    data = {}
    is_debug = None

    def __init__(self, state,
                 move,
                 reward=0,
                 parent=None,
                 logger=logging.getLogger(__name__)):
        # node
        Node.data[str(state.state)] = self
        self.state = state  # state s
        self.parent = parent

        # edge resulting to the node
        self.reward = reward
        self.move = move  # action index
        self.is_expanded = False

        # children node information
        if Node.env is not None:
            action_space_dim = Node.env.action_space_dim
        else:
            raise Exception("Setup the env first")
        if Node.net is None:
            raise Exception("Setup the env net first")

        self.children = {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t_state = torch.as_tensor(self.state.state, dtype=torch.float32, device=device)[
            None, None]
        self.estimated_heuristic = Node.net(t_state).detach().cpu().numpy().reshape(-1)
        self.child_priors = np.zeros([action_space_dim], dtype=np.float32)
        # self.child_Q = np.zeros([action_space_dim], dtype=np.float32)
        # self.child_Q = (self.estimated_heuristic - self.estimated_heuristic.min() + 1) / (
        #         self.estimated_heuristic.max() - self.estimated_heuristic.min() + 1
        # )
        self.child_Q = self.estimated_heuristic
        self.child_number_visits = np.zeros([action_space_dim], dtype=np.float32)
        self.action_idxes = state.feasible_action_set.copy()

        # logger
        self.logger = logger

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value

    def check_repeat(self):
        top_of_chain = self.parent
        current_state = self.state
        while top_of_chain.parent is not None:
            if current_state == top_of_chain.state:
                return True
            top_of_chain = top_of_chain.parent
        return False

    @property
    def V_s(self):
        """
        V(s): V_s of the current state
        V(s) = argmax_a'{Q(s,a')}
        """
        return np.max(self.child_Q[self.action_idxes])

    def set_V(self, v):
        """
        Q(s,a) = r + yV(s), s is the current state, a is the action
        :param v : children state V(s)
        """
        self.parent.child_Q[self.move] = self.lr * v + self.reward

    def set_V_end(self):
        """
        Q(s,a) = r, if done
        """
        self.parent.child_Q[self.move] = self.reward

    def child_U(self, c_puct=.0):
        # return c_puct * np.sqrt(self.number_visits) * (
        #         abs(self.child_priors) / (1 + self.child_number_visits))
        return c_puct * np.sqrt(self.number_visits) / (1 + self.child_number_visits)

    def best_child(self):
        if len(self.action_idxes) != 0:
            bestmove = self.child_Q + self.child_U()
            if self.is_debug:
                fig = plt.figure()
                ax = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)

                ax.plot(np.arange(len(bestmove)), self.child_Q)
                ax2.plot(np.arange(len(bestmove)), self.child_U())
                # ax = fig.add_subplot(111, projection='3d')
                # coord = np.array([np.array(self.env._expr_action(i)) for i in self.action_idxes]).reshape(-1, 4)
                # ax.scatter(coord[:, 0], coord[:, 1], bestmove[self.action_idxes])
                ax.scatter(self.action_idxes, bestmove[self.action_idxes], c='r')
                for i, txt in enumerate(self.action_idxes):
                    ax.annotate(txt, (self.action_idxes[i], bestmove[self.action_idxes][i]))

                plt.title(self.move)
                plt.show()
            bestmove = self.action_idxes[np.argmax(bestmove[self.action_idxes])]
        else:
            bestmove = np.argmax(self.child_Q + self.child_U())
        return bestmove

    def maybe_add_child(self, move):
        if move not in self.children:
            self.env.reset_state(self.state)
            state_moved, reward, is_finished, _ = self.env.step(move)
            self.children[move] = Node(state=state_moved, move=move, reward=reward, parent=self)
            if self.children[move].check_repeat():
                del self.children[move]
                # remove the action that can cause loop
                self.action_idxes = np.delete(self.action_idxes, np.argwhere(self.action_idxes == move))
                return self
        return self.children[move]

    def select_leaf(self):
        current = self
        # print('root', end='')
        while current.is_expanded:
            best_move = current.best_child()
            # print(f"->{best_move}", end='')
            current = current.maybe_add_child(best_move)
        # print()
        return current

    def expand(self):
        """
        To figure out 1. feasible actions for the current leaf node
                      2. move probabilities for each feasible action
        """

        self.is_expanded = True
        action_inds = self.action_idxes
        if len(action_inds) == 0:
            self.is_expanded = False

        # m = torch.nn.Softmax(dim=0)
        # c_p = m(torch.tensor(self.estimated_heuristic[action_inds])).numpy()
        # _mask = np.zeros_like(self.child_priors)
        # _mask[action_inds] = c_p
        # c_p = _mask
        # # TODO should we make the sum of the probability to 1
        # self.child_priors = c_p

    def backup(self, is_done: bool = False):
        """
        Backpropagate the 1. N: visit times
                          2. W: total values
        """
        current = self
        while current.parent is not None:
            current.number_visits += 1
            current = current.parent
        if is_done:
            self.parent.action_idxes = np.delete(self.parent.action_idxes,
                                                 np.argwhere(self.parent.action_idxes == self.move))

            path = []
            current = self
            while current.parent is not None:
                path.append(current.state)
                current = current.parent
            return path

        # if not is_done:
        #     current.set_V()
        # else:
        #     current.set_V_end()
        # # print(f'r:{current.reward}, m: {current.move}', end='->')
        # # try:
        # #     print(current.parent.child_Q[[current.parent.action_idxes]])
        # # except:
        # #     pass

        #     current.set_V(current.V_s)
        #     # print(f'r:{current.reward}, m: {current.move}, v: {current.V_s}', end='->')
        #     # try:
        #     #     print(current.parent.child_Q[[current.parent.action_idxes]])
        #     # except:
        #     #     pass

        # # print()
        # # print("-" * 20)


class DummyNode(object):
    def __init__(self, lr):
        self.parent = None
        self.child_Q = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)
        self.lr = lr

    def set_Q(self, a, child_v, rew):
        """
        Q(s,a) = r + yV(s), s is the current state, a is the action
        :param a : action
        :param child_v : children state V(s)
        :param rew: reward of action a
        """
        self.child_Q[a] = self.lr * child_v + rew


def MCTS(state,
         env,
         iter_num: int,
         net: DuelingDQN,
         lr=.95,
         infeasible_dict=None,
         infeasible_local_pattern=None,
         is_debug=False):
    if infeasible_dict is None:
        infeasible_dict = {}
    if infeasible_local_pattern is None:
        infeasible_local_pattern = {}
    # infomation
    generated_path = []
    Node.lr = lr
    Node.env = env.copy()
    Node.net = net
    Node.is_debug = is_debug
    root = Node(state, move=None, parent=DummyNode(lr))
    for i in range(iter_num):
        leaf = root.select_leaf()
        if len(infeasible_dict) > 0:
            if str(leaf.state.state) in infeasible_dict:
                leaf.action_idxes = np.delete(leaf.action_idxes,
                                              np.argwhere(np.in1d(leaf.action_idxes,
                                                                  infeasible_dict[str(leaf.state.state)])))
        if len(infeasible_local_pattern) > 0:
            for slot_id in infeasible_local_pattern.keys():
                local_pattern = str(np.pad(leaf.state.state, 1)[slot_id[0]:slot_id[0] + 3,
                    slot_id[1]:slot_id[1] + 3])
                if local_pattern in infeasible_local_pattern[slot_id]:
                    leaf.action_idxes = np.delete(leaf.action_idxes, np.argwhere(np.in1d(leaf.action_idxes,
                                                                  infeasible_local_pattern[slot_id][local_pattern])))

        # child_priors = child_priors.detach().cpu().numpy().reshape(-1)
        # value_estimate = value_estimate.item()
        if env.is_finished(leaf.state) or len(leaf.state.feasible_action_set) == 0:
            path = leaf.backup(is_done=True)
            if len(leaf.state.feasible_action_set) > 0:
                generated_path.append(path)
            continue
        leaf.expand()  # need to make sure valid moves
        leaf.backup()
    # print([len(p) for p in generated_path])
    return root, generated_path


def MCTS_Continue(infeasible_state_p_1, infeasible_state_p_2, root, infeasible_dict={}, iter=300):
    def _to_action(pick_id, place_id, rack_size):
        pick_id_int = pick_id[0] * rack_size[1] + pick_id[1]
        place_id_int = place_id[0] * rack_size[1] + place_id[1]
        return pick_id_int * np.prod(rack_size) + place_id_int

    move = infeasible_state_p_2 - infeasible_state_p_1
    move_to_idx = np.where(move > 0)
    move_from_idx = np.where(move < 0)
    if len(move_to_idx[0]) != 1 or len(move_from_idx[0]) != 1:
        return None, None
    action_id = _to_action(move_to_idx, move_from_idx, infeasible_state_p_1.shape)
    if str(infeasible_state_p_1) in infeasible_dict:
        infeasible_dict[str(infeasible_state_p_1)].append(action_id)
    else:
        infeasible_dict[str(infeasible_state_p_1)] = [action_id]
    node = Node.data.get(str(infeasible_state_p_1), None)
    if node is not None:
        return (*MCTS(root.state, None, iter, None, None,
                      infeasible_dict=infeasible_dict), infeasible_dict)
    else:
        return None, None, None


if __name__ == "__main__":
    import numpy as np
    import huri.core.file_sys as fs
    from huri.learning.env.arrangement_planning_rack.env import RackArrangementEnv, RackState, RackStatePlot

    # initialize the environment
    num_tube_classes = 3
    rack_size = (5, 10)
    action_space_dim = np.prod(rack_size) ** 2
    observation_space_dim = (1, *rack_size)
    env = RackArrangementEnv(rack_size=rack_size,
                             num_classes=num_tube_classes,
                             observation_space_dim=observation_space_dim,
                             action_space_dim=action_space_dim,
                             is_curriculum_lr=True,
                             is_goalpattern_fixed=True,
                             seed=1988,
                             is_evl_mode=True,
                             difficulty=26)

    env.goal_pattern = np.array([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                                 [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                                 [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                                 [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                                 [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]])

    data_path = fs.workdir_learning / "run" / f"dqn_2022_01_08_21_13_41"
    model_name = "model_1862000-1864000.pth"
    model_path = data_path / "model" / model_name

    # load neural network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DuelingDQN(obs_dim=observation_space_dim, action_dim=action_space_dim).to(device)
    net.load_state_dict(torch.load(model_path))

    success = 0
    fail = 0
    num_trial = 100
    total_length = 0
    for i in range(num_trial):
        state = env.reset()
        root, paths = MCTS(state=state,
                           env=env,
                           net=net,
                           iter_num=200,
                           lr=.95)
        if len(paths) > 0:
            success += 1
        else:
            fail += 1
        if len(paths) > 0:
            shortest_path = paths[np.argmin([len(_) for _ in paths])][::-1]
            total_length += len(shortest_path)
            # print([_.state for _ in shortest_path])
            # drawer = RackStateVisualizer(env.goal_pattern)
            # drawer.plot_states(path, row=6)
        # if len(paths) > 0:
        #     for p in [paths[0]]:
        #         drawer = RackStateVisualizer(env.goal_pattern)
        #         drawer.plot_states(p[::-1], row=6)
    print("num of success trival is ", success)
    print("num of failed trival is ", fail)
    print("average length of the path is  ", total_length / num_trial)
    # print(len(paths))
