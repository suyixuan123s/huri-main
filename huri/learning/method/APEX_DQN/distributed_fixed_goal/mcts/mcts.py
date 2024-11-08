import logging
import math
import time

import numpy as np
from typing import Callable, Tuple, Mapping, Union, Any
import collections

import torch

from huri.learning.env.rack_v3.env_mix import RackArrangementEnv
import copy

log = logging.getLogger(__name__)

"""
Mastering the game of Go without human knowledge
MCTS:
Each node (s) in the search tree stores:

0. the edges (a) stemmed from the node (s) 
1. prior probability P(s,a)
2. a visit count N(s,a)
3. total action value W(s,a)
4. mean action value Q(s,a)
-----
selection policy
    a* = argmax_a { Q(s,a) + U(s,a) }
    U(s,a) \propto P(s,a)/(1+N(s,a))
    U(s,a) := C(s) * P(s,a) * sqrt[sum_b(N(s,b)] / (1+N(s,a))
    Q(s,a) := W(s,a)/N(s,a) 
    C(s) := log((1+N(s)+c_base)/c_base) + c_init          => grows slowly with search time     
"""


class Node:
    """Node in the MCTS search tree."""

    def __init__(self, to_play: int, num_actions: np.ndarray, move: int = None, parent: Any = None) -> None:
        """
        Args:
            to_play: the id of the current player.
            num_actions: number of total actions, including illegal move.
            prior: a prior probability of the node for a specific action, could be empty in case of root node.
            move: the action associated with the prior probability.
            parent: the parent node, could be `None` if this is the root node.
        """

        self.to_play = to_play
        self.move = move
        self.parent = parent
        self.num_actions = num_actions
        self.is_expanded = False

        self.child_total_value = np.zeros(num_actions, dtype=np.float32)
        self.child_number_visits = np.zeros(num_actions, dtype=np.float32)
        self.child_priors = np.zeros(num_actions, dtype=np.float32)

        self.children: Mapping[int, Node] = {}

        # Number of virtual losses on this node, only used in 'parallel_uct_search'
        self.losses_applied = 0

    def child_U(self, c_puct_base: float, c_puct_init: float) -> np.ndarray:
        """Returns a 1D numpy.array contains prior score for all child."""
        pb_c = math.log((1 + self.number_visits + c_puct_base) / c_puct_base) + c_puct_init
        return pb_c * self.child_priors * (math.sqrt(self.number_visits) / (1 + self.child_number_visits))

    def child_Q(self):
        """Returns a 1D numpy.array contains mean action value for all child."""
        return self.child_total_value / (1 + self.child_number_visits)

    @property
    def number_visits(self):
        """The number of visits for current node is stored at parent's level."""
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value

    @property
    def total_value(self):
        """The total value for current node is stored at parent's level."""
        return self.parent.child_total_value[self.move]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value

    @property
    def Q(self):
        """Returns the mean action value Q(s, a)."""
        return self.parent.child_total_value[self.move] / (1 + self.parent.child_total_value[self.move])

    @property
    def has_parent(self) -> bool:
        return isinstance(self.parent, Node)


class DummyNode(object):
    """A place holder to make computation possible for the root node."""

    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)


def best_child(node: Node, legal_actions: np.ndarray, c_puct_base: float, c_puct_init: float,
               child_to_play: int) -> Node:
    """ Returns best child node with maximum action value Q plus an upper confidence bound U.
    And creates the selected best child node if not already exists.

    Args:
        node: the current node in the search tree.
        legal_actions: a 1D bool numpy.array mask for all actions,
                where `1` represents legal move and `0` represents illegal move.
        c_puct_base: a float constant determining the level of exploration.
        c_puct_init: a float constant determining the level of exploration.
        child_to_play: the player id for children nodes.

    Returns:
        The best child node corresponding to the UCT score.

    Raises:
        ValueError:
            if the node instance itself is a leaf node.
    """
    if not node.is_expanded:
        raise ValueError('Expand leaf node first.')

    # The child Q value is evaluated from the opponent perspective.
    # when we select the best child for node, we want to do so from node.to_play's perspective,
    # so we always switch the sign for node.child_Q values, this is required since we're talking about two-player, zero-sum games.
    ucb_scores = node.child_Q() + node.child_U(c_puct_base, c_puct_init)

    # Exclude illegal actions, note in some cases, the max ucb_scores may be zero.
    ucb_scores = np.where(legal_actions, ucb_scores, -1000)

    if np.any(np.isnan(ucb_scores)):
        print("A!")
    # Break ties if we have multiple 'maximum' values.
    move = np.random.choice(np.where(ucb_scores == ucb_scores.max())[0])

    assert legal_actions[move]

    if move not in node.children:
        node.children[move] = Node(to_play=child_to_play, num_actions=node.num_actions, move=move, parent=node)

    return node.children[move]


def expand(node: Node, ) -> None:
    """Expand all actions, including illegal actions.

    Args:
        node: current leaf node in the search tree.
        prior_prob: 1D numpy.array contains prior probabilities of the state for all actions.

    Raises:
        ValueError:
            if node instance already expanded.
            if input argument `prior` is not a valid 1D float numpy.array.
    """
    if node.is_expanded:
        raise RuntimeError('Node already expanded.')

    node.is_expanded = True


def backup(node: Node, value: float) -> None:
    """Update statistics of the this node and all traversed parent nodes.

    Args:
        node: current leaf node in the search tree.
        value: the evaluation value (from last player's perspective).

    Raises:
        ValueError:
            if input argument `value` is not float data type.
    """

    if not isinstance(value, float):
        raise ValueError(f'Expect `value` to be a float type, got {type(value)}')

    while node is not None and isinstance(node, Node):
        node.number_visits += 1
        node.total_value += value
        node = node.parent


def add_dirichlet_noise(node: Node, legal_actions: np.ndarray, eps: float = 0.25, alpha: float = 0.03) -> None:
    """Add dirichlet noise to a given node.

    Args:
        node: the root node we want to add noise to.
        legal_actions: a 1D bool numpy.array mask for all actions,
            where `1` represents legal move and `0` represents illegal move.
        eps: epsilon constant to weight the priors vs. dirichlet noise.
        alpha: parameter of the dirichlet noise distribution.

    Raises:
        ValueError:
            if input argument `node` is not expanded.
            if input argument `eps` or `alpha` is not float type
                or not in the range of [0.0, 1.0].
    """

    if not isinstance(node, Node) or not node.is_expanded:
        raise ValueError('Expect `node` to be expanded')
    if not isinstance(eps, float) or not 0.0 <= eps <= 1.0:
        raise ValueError(f'Expect `eps` to be a float in the range [0.0, 1.0], got {eps}')
    if not isinstance(alpha, float) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f'Expect `alpha` to be a float in the range [0.0, 1.0], got {alpha}')

    alphas = np.ones_like(legal_actions) * alpha
    noise = np.random.dirichlet(alphas)

    node.child_priors = node.child_priors * (1 - eps) + noise * eps


def generate_search_policy(visit_counts: np.ndarray, temperature: float) -> np.ndarray:
    """Returns a policy action probabilities after MCTS search,
    proportional to its exponentialted visit count.

    Args:
        visit_counts: the visit number of the children nodes from the root node of the search tree.
        temperature: a parameter controls the level of exploration.

    Returns:
        a 1D numpy.array contains the action probabilities after MCTS search.

    Raises:
        ValueError:
            if input argument `temperature` is not float type or not in range [0.0, 1.0].
    """
    if not isinstance(temperature, float) or not 0 <= temperature <= 1.0:
        raise ValueError(f'Expect `temperature` to be float type in the range [0.0, 1.0], got {temperature}')

    if temperature > 0.0:
        # Hack to avoid overflow when doing power operation over large numbers
        exp = min(10.0, 1.0 / temperature)
        visit_counts = np.power(visit_counts, exp)

    return visit_counts / np.sum(visit_counts)


def add_virtual_loss(node: Node) -> None:
    """Propagate a virtual loss to the traversed path.

    Args:
        node: current leaf node in the search tree.

    """
    # This is a loss for both players in the traversed path,
    # since we want to avoid multiple threads to select the same path.
    # However since we'll switching the sign for child_Q when selecting best child,
    # here we use +1 instead of -1.
    vloss = +1
    while node.parent is not None:
        node.losses_applied += 1
        node.total_value += vloss
        node = node.parent


def revert_virtual_loss(node: Node) -> None:
    """Undo virtual loss to the traversed path.

    Args:
        node: current leaf node in the search tree.
    """

    vloss = -1
    while node.parent is not None:
        if node.losses_applied > 0:
            node.losses_applied -= 1
            node.total_value += vloss
        node = node.parent


def uct_search(
        env: RackArrangementEnv,
        eval_func: Callable[[np.ndarray, bool], Tuple[np.ndarray, Union[np.ndarray, float]]],
        root_node: Node,
        c_puct_base: float,
        c_puct_init: float,
        temperature: float,
        num_simulations: int = 800,
        root_noise: bool = False,
        deterministic: bool = False,
) -> Tuple[int, np.ndarray, Node]:
    """Single-threaded Upper Confidence Bound (UCB) for Trees (UCT) search without any rollout.

    It follows the following general UCT search algorithm, except here we don't do rollout.
    ```
    function UCTSEARCH(r,m)
      i←1
      for i ≤ m do
          n ← select(r)
          n ← expand(n)
          ∆ ← rollout(n)
          backup(n,∆)
      end for
      return end function
    ```

    Args:
        env: a gym like custom BoardGameEnv environment.
        eval_func: a evaluation function when called returns the
            action probabilities and winning probability from
            current player's perspective.
        root_node: root node of the search tree, this comes from reuse sub-tree.
        c_puct_base: a float constant determining the level of exploration.
        c_puct_init: a float constant determining the level of exploration.
        temperature: a parameter controls the level of exploration
            when generate policy action probabilities after MCTS search.
        num_simulations: number of simulations to run, default 800.
        root_noise: whether add dirichlet noise to root node to encourage exploration, default off.
        deterministic: after the MCTS search, choose the child node with most visits number to play in the game,
            instead of sample through a probability distribution, default off.

    Returns:
        tuple contains:
            a integer indicate the sampled action to play in the environment.
            a 1D numpy.array search policy action probabilities from the MCTS search result.
            a Node instance represent subtree of this MCTS search, which can be used as next root node for MCTS search.

    Raises:
        ValueError:
            if input argument `env` is not valid BoardGameEnv instance.
            if input argument `num_simulations` is not a positive integer.
        RuntimeError:
            if the game is over.
    """
    if not isinstance(env, RackArrangementEnv):
        raise ValueError(f'Expect `env` to be a valid BoardGameEnv instance, got {env}')
    if not 1 <= num_simulations:
        raise ValueError(f'Expect `num_simulations` to a positive integer, got {num_simulations}')
    if env.is_finished():
        raise RuntimeError('Game is over.')

    # Create root node
    if root_node is None:
        root_node = Node(to_play=0, num_actions=env.action_space_dim, parent=DummyNode())
        prior_prob, value = eval_func(env.observation(), False)
        expand(root_node)
        backup(root_node, value)

    # Add dirichlet noise to the prior probabilities to root node.
    if root_noise:
        add_dirichlet_noise(root_node, env.legal_actions())
    while root_node.number_visits < num_simulations:
        node = root_node
        # Make sure do not touch the actual environment.
        sim_env = env.copy(toggle_reset=False)
        obs = sim_env.observation()
        done = sim_env.is_finished()
        # Phase 1 - Select
        # Select best child node until one of the following is true:
        # - reach a leaf node.
        # - game is over.
        while node.is_expanded:
            # Select the best move and create the child node on demand
            node = best_child(node, sim_env.legal_actions(), c_puct_base, c_puct_init, 0)
            # Make move on the simulation environment.
            _obs, reward, done, _ = sim_env.step(node.move)
            obs = sim_env.observation()
            if done:
                break
        # Special case - If game is over, using the actual reward from the game to update statistics
        if done:
            # The reward is for the last player who made the move won/loss the game.
            # So when backing up value from node represents current player, we add a minus sign
            backup(node, float(reward))
            continue
        # Phase 2 - Expand and evaluation
        prior_prob, value = eval_func(obs, False)
        expand(node, prior_prob)
        # Phase 3 - Backup statistics
        backup(node, value)
    # Play - generate action probability from the root node.
    pi_probs = generate_search_policy(root_node.child_number_visits, temperature)

    move = None
    if deterministic:
        # Choose the action with most visit count.
        move = np.argmax(root_node.child_number_visits)
    else:
        # Sample an action.
        while move is None or not env.legal_actions()[move]:
            move = np.random.choice(np.arange(pi_probs.shape[0]), p=pi_probs)

    # import huri.core.file_sys as fs
    # import time
    # fs.dump_pickle(root_node, f"./root_node_{time.strftime('%Y%m%d-%H%M%S')}.pkl", reminder=False)
    # exit(0)

    next_root_node = None
    if move in root_node.children:
        next_root_node = root_node.children[move]
        next_root_node.parent = DummyNode()

    return (move, pi_probs, next_root_node)


def parallel_uct_search(
        env: RackArrangementEnv,
        eval_func: Callable[[np.ndarray, bool], Tuple[np.ndarray, Union[np.ndarray, float]]],
        root_node: Node,
        c_puct_base: float,
        c_puct_init: float,
        temperature: float,
        num_simulations: int = 800,
        num_parallel: int = 8,
        root_noise: bool = False,
        deterministic: bool = False,
) -> Tuple[int, np.ndarray, Node]:
    """Single-threaded Upper Confidence Bound (UCB) for Trees (UCT) search without any rollout.

    This implementation uses tree parallel search and batched evaluation.

    It follows the following general UCT search algorithm, except here we don't do rollout.
    ```
    function UCTSEARCH(r,m)
      i←1
      for i ≤ m do
          n ← select(r)
          n ← expand(n)
          ∆ ← rollout(n)
          backup(n,∆)
      end for
      return end function
    ```

    Args:
        env: a gym like custom BoardGameEnv environment.
        eval_func: a evaluation function when called returns the
            action probabilities and winning probability from
            current player's perspective.
        root_node: root node of the search tree, this comes from reuse sub-tree.
        c_puct_base: a float constant determining the level of exploration.
        c_puct_init: a float constant determining the level of exploration.
        temperature: a parameter controls the level of exploration
            when generate policy action probabilities after MCTS search.
        num_simulations: number of simulations to run, default 800.
        num_parallel: Number of parallel leaves for MCTS search. This is also the batch size for neural network evaluation.
        root_noise: whether add dirichlet noise to root node to encourage exploration,
            default off.
        deterministic: after the MCTS search, choose the child node with most visits number to play in the game,
            instead of sample through a probability distribution, default off.

    Returns:
        tuple contains:
            a integer indicate the sampled action to play in the environment.
            a 1D numpy.array search policy action probabilities from the MCTS search result.
            a Node instance represent subtree of this MCTS search, which can be used as next root node for MCTS search.

    Raises:
        ValueError:
            if input argument `env` is not valid BoardGameEnv instance.
            if input argument `num_simulations` is not a positive integer.
        RuntimeError:
            if the game is over.
    """
    if not isinstance(env, RackArrangementEnv):
        raise ValueError(f'Expect `env` to be a valid RackArrangementEnv instance, got {type(env)}')
    if not 1 <= num_simulations:
        raise ValueError(f'Expect `num_simulations` to a positive integer, got {num_simulations}')
    if env.is_finished():
        raise RuntimeError('Game is over.')

    # Create root node
    if root_node is None:
        root_node = Node(to_play=0, num_actions=env.action_space_dim, parent=DummyNode())
        expand(root_node, prior_prob)
        backup(root_node, value)

    # Add dirichlet noise to the prior probabilities to root node.
    if root_noise:
        add_dirichlet_noise(root_node, env.legal_actions())
    while root_node.number_visits < num_simulations + num_parallel:
        leaves = []
        failsafe = 0
        while len(leaves) < num_parallel and failsafe < num_parallel * 2:
            # This is necessary as when a game is over no leaf is added to leaves,
            # as we use the actual game results to update statistic
            failsafe += 1
            node = root_node
            # Make sure do not touch the actual environment.
            sim_env = env.copy(toggle_reset=False)
            obs = sim_env.observation()
            done = sim_env.is_finished()

            # Phase 1 - Select
            # Select best child node until one of the following is true:
            # - reach a leaf node.
            # - game is over.
            while node.is_expanded:
                # Select the best move and create the child node on demand
                node = best_child(node, sim_env.legal_actions(), c_puct_base, c_puct_init, 0)
                # Make move on the simulation environment.
                _obs, reward, done, _ = sim_env.step(node.move)
                obs = sim_env.observation()
                if done:
                    break
            # Special case - If game is over, using the actual reward from the game to update statistics.
            if done:
                # The reward is for the last player who made the move won/loss the game.
                # So when backing up value from node represents current player, we add a minus sign
                backup(node, float(reward))
                continue
            else:
                add_virtual_loss(node)
                leaves.append((node, obs))
        if leaves:
            batched_nodes, batched_obs = map(list, zip(*leaves))
            prior_probs, values = eval_func(np.stack(batched_obs, axis=0), True)

            for leaf, prior_prob, value in zip(batched_nodes, prior_probs, values):
                revert_virtual_loss(leaf)

                # If a node was picked multiple times (despite virtual losses), we shouldn't
                # expand it more than once.
                if leaf.is_expanded:
                    continue

                expand(leaf, prior_prob)
                backup(leaf, value.item())

    # Play - generate action probability from the root node.
    pi_probs = generate_search_policy(root_node.child_number_visits, temperature)

    move = None
    if deterministic:
        # Choose the action with most visit count.
        move = np.argmax(root_node.child_number_visits)
    else:
        # Sample an action.
        while move is None or not env.legal_actions()[move]:
            # print(pi_probs)
            try:
                move = np.random.choice(np.arange(pi_probs.shape[0]), p=pi_probs)
            except Exception as e:
                print(e)
                import pdb
                pdb.set_trace()

    import huri.core.file_sys as fs
    import time
    fs.dump_pickle(root_node, f"./root_node_parallel_{time.strftime('%Y%m%d-%H%M%S')}.pkl", reminder=False)
    exit(0)

    next_root_node = None
    if move in root_node.children:
        next_root_node = root_node.children[move]
        next_root_node.parent = DummyNode()

    return (move, pi_probs, next_root_node)


import hydra
from huri.learning.env.rack_v3 import create_env
from huri.learning.method.APEX_DQN.distributed.network_old import DDQN as DDQN2


@hydra.main(config_path='../params', config_name='20230517_3x6_2.yaml', version_base='1.3')
def create_env_3x3(cfg):
    env_meta = create_env(rack_sz=cfg['env']['rack_sz'],
                          num_tube_class=cfg['env']['num_tube_class'],
                          seed=cfg['env']['seed'],
                          toggle_curriculum=cfg['env']['toggle_curriculum'],
                          toggle_goal_fixed=cfg['env']['toggle_goal_fixed'],
                          scheduler='GoalRackStateScheduler3',
                          num_history=1)
    input_shape = env_meta.observation_space_dim
    num_actions = env_meta.action_space_dim
    network = DDQN2(input_shape, num_actions,
                    num_category=2,
                    num_filters=10,
                    num_res_block=19,
                    num_fc_units=128)
    # torch.load('')

    parallel_uct_search(
        env=env,
        eval_func=network,
        root_node=None,
        c_puct_base=19652,
        c_puct_init=1.25,
        temperature=1,
        num_simulations=100,
        num_parallel=1,
        root_noise=True,
        deterministic=True,
    )

    return network


# Tree Compression
if __name__ == "__main__":
    from huri.learning.env.rack_v3.env_mix import RackState, RackStatePlot
    from huri.learning.method.APEX_DQN.distributed.network import DDQN

    class_num = 1
    rack_size = (9, 9)
    obs_dim, act_dim = RackState.get_obs_act_dim_by_size(rack_size)
    env = RackArrangementEnv(rack_size=rack_size,
                             num_classes=class_num,
                             observation_space_dim=obs_dim,
                             is_goalpattern_fixed=False,
                             is_curriculum_lr=False, num_history=1)

    nn = create_env_3x3()
