import numpy as np
import copy

from shapely.geometry.polygon import Point
from torch.distributions.utils import lazy_property

from huri.learning.env.arrangement_planning_rack.env import RackArrangementEnv, RackStatePlot
from huri.learning.env.arrangement_planning_two_rack.utils import get_random_states, get_fillable_movable


class RackState(object):
    """
    The state of rack. It provides convenient API
    """
    _cache = {}
    _from_rack_ind = None
    _to_rack_ind = None
    _pad_rack_ind = None

    def __init__(self, state):
        self.state = np.asarray(state, dtype=int)

    def _cal_fillable_movable_slots(self):
        """
        An internal function, do not explicitly call it.
        Calculate movable and fillable slots; and cache the results.

        Returns
        -------

        """
        fillable_slots, movable_slots = get_fillable_movable(self.state)
        fillable_slots[self._pad_rack_ind] = np.zeros_like(fillable_slots[self._pad_rack_ind])
        state_str = str(self.state)
        RackState._cache[state_str] = {
            "fillable_slots": fillable_slots,
            "movable_slots": movable_slots,
        }

    def swap(self, ind_a, ind_b):
        """
        Swap two positions in the rack and return a new rack state

        Parameters
        ----------
        ind_a: position A index
        ind_b: position B index

        Returns RackState
        -------

        """
        new_state = self.state.copy()
        new_state[ind_a], new_state[ind_b] = self.state[ind_b], self.state[ind_a]
        return RackState(new_state)

    @lazy_property
    def feasible_action_set(self):
        """
        Get feasible action set

        Returns np.array
        -------

        """
        state = self.state.copy()
        rack_size = state.shape
        fillable_slots_ind = np.where(self.fillable_slots.ravel() == 1)[0]
        movable_slots_ind = np.where(self.movable_slots.ravel() == 1)[0]
        if len(fillable_slots_ind) == 0 or len(movable_slots_ind) == 0:
            return np.array([])
        fillable_movable_ind_comb = np.array(np.meshgrid(fillable_slots_ind, movable_slots_ind)).T.reshape(-1, 2)
        return fillable_movable_ind_comb[:, 0] * np.prod(rack_size) + fillable_movable_ind_comb[:, 1]

    @property
    def fillable_slots(self):
        """
        Return feasible slots indices

        Returns np.array
        -------

        """
        state_str = str(self.state)
        if state_str not in RackState._cache:
            self._cal_fillable_movable_slots()
        return RackState._cache[state_str]["fillable_slots"]

    @property
    def movable_slots(self):
        """
        Return movable slots indices

        Returns np.array
        -------

        """
        state_str = str(self.state)
        if state_str not in RackState._cache:
            self._cal_fillable_movable_slots()
        return RackState._cache[state_str]["movable_slots"]

    def __sub__(self, other) -> np.ndarray:
        assert type(self) == type(other)
        return self.copy(self.state - other.state)

    def __getitem__(self, item):
        return self.state[item]

    def __eq__(self, other):
        assert type(self) == type(other)
        return np.array_equal(self.state, other.state)

    def __lt__(self, other):
        return self.state < other

    def __gt__(self, other):
        return self.state > other

    def __le__(self, other):
        return self.state <= other

    def __ge__(self, other):
        return self.state >= other

    def __repr__(self):
        return np.array2string(self.state)

    def copy(self, state=None):
        if state is None:
            state = self.state.copy()
        else:
            state = copy.deepcopy(state)
        return RackState(state=state)


class TwoRackArrangementEnv(RackArrangementEnv):
    def __init__(self,
                 from_rack_size=(5, 10),
                 to_rack_size=(5, 10),
                 num_classes=5,
                 observation_space_dim=(5, 10),
                 action_space_dim=10,
                 action_seed=777,
                 is_curriculum_lr=True,
                 is_goalpattern_fixed=True,
                 difficulty=2,
                 seed=777,
                 is_evl_mode=True):

        # rack size
        self.from_rack_size = from_rack_size
        self.to_rack_size = from_rack_size
        self.two_rack_size = (from_rack_size[0] + to_rack_size[0] + 1, max(from_rack_size[1], to_rack_size[1]))
        self.from_rack_ind = (slice(0, from_rack_size[0]), slice(0, self.two_rack_size[1]))
        self.to_rack_ind = (slice(from_rack_size[0] + 1, from_rack_size[0] + 1 + to_rack_size[0]),
                            slice(0, self.two_rack_size[1]))
        RackState._from_rack_ind = self.from_rack_ind
        RackState._to_rack_ind = self.to_rack_ind
        RackState._pad_rack_ind = (slice(0, from_rack_size[0] + 1), slice(0, self.two_rack_size[1]))

        super().__init__(rack_size=self.two_rack_size, num_classes=num_classes,
                         observation_space_dim=observation_space_dim,
                         action_space_dim=action_space_dim, action_seed=action_seed, is_curriculum_lr=is_curriculum_lr,
                         is_goalpattern_fixed=is_goalpattern_fixed, difficulty=difficulty, seed=seed,
                         is_evl_mode=is_evl_mode, )

    def reset_state_goal(self, init_state, goal_pattern):
        self.goal_pattern = RackState(goal_pattern)
        self.state = self.reset_state(init_state)
        return self.state

    def reset_state(self, init_state):
        if isinstance(init_state, np.ndarray):
            self.state = RackState(init_state.copy())
        elif isinstance(init_state, RackState):
            self.state = init_state.copy()
        else:
            raise Exception("Undefined input type")
        self.rack_state_history = []
        self.reward_history = []
        return self.state

    def reset(self):
        if not self.is_goalpattern_fixed:
            goal_pattern = self._np_random.randint(1, self.num_classes + 1, size=self.rack_size)
        else:
            if isinstance(self.goal_pattern, RackState):
                goal_pattern = self.goal_pattern.state
            elif isinstance(self.goal_pattern, np.ndarray):
                goal_pattern = self.goal_pattern
            else:
                raise Exception("No support goal state")
        if not self.is_curriculum_lr:
            init_state = get_random_states(self.from_rack_size, goal_pattern[self.to_rack_ind],
                                           np_random=self._np_random)
        else:
            goalpattern_ravel = goal_pattern.ravel()
            max_len = len(np.where(goalpattern_ravel > 0)[0]) + 1
            feasible_diff = min(max_len, self.difficulty)
            min_len = np.arange(feasible_diff - 1) + 1
            tmp = (min_len) * 2
            distribution = tmp / np.sum(tmp)
            if self.is_evl_mode:
                min_obj_num = feasible_diff - 1
            else:
                min_obj_num = self._np_random.choice(min_len, p=distribution)
            init_state = get_random_states(self.from_rack_size, goal_pattern[self.to_rack_ind],
                                           min_obj_num=min_obj_num,
                                           obj_num=feasible_diff,
                                           np_random=self._np_random)
        init_state_two_rack = np.zeros(self.two_rack_size)
        init_state_two_rack[:self.from_rack_size[0], :self.from_rack_size[1]] = init_state

        return self.reset_state_goal(init_state=init_state_two_rack, goal_pattern=goal_pattern)

    def _get_reward(self, is_finished, state_current, state_next, goal_pattern):
        if is_finished:
            return 50
        reward = 0  # every move will cause 1 penalty
        move_map = state_next - state_current
        move_to_idx = np.where(move_map > 0)
        move_from_idx = np.where(move_map < 0)
        is_move_to_pattern = goal_pattern[move_to_idx] == state_next[move_to_idx]
        is_move_out_pattern = goal_pattern[move_from_idx] == state_current[move_from_idx]
        if is_move_to_pattern and not is_move_out_pattern:
            reward += 1
        if is_move_out_pattern:
            reward -= 2
        # if not is_move_to_pattern:
        #     reward -= 1
        reward -= np.linalg.norm(np.array(move_to_idx) - np.array(move_from_idx)) / \
                  np.linalg.norm(np.array(self.two_rack_size))
        return reward

    def set_goal_pattern(self, goal_pattern):
        if isinstance(goal_pattern, np.ndarray):
            self.goal_pattern = RackState(goal_pattern.copy())
        elif isinstance(goal_pattern, RackState):
            self.goal_pattern = goal_pattern.copy()
        else:
            raise Exception("Undefined input type")

    def to_state(self, state_np: np.ndarray):
        return RackState(state_np)


if __name__ == "__main__":
    num_classes = 3
    rack_size_1 = (5, 10)
    rack_size_2 = (5, 10)
    two_rack_size = (rack_size_1[0] + rack_size_2[0] + 1,
                     max(rack_size_1[1], rack_size_2[1]))
    action_space_dim = np.prod(two_rack_size) ** 2
    observation_space_dim = (1, *two_rack_size)
    env = TwoRackArrangementEnv(from_rack_size=rack_size_1,
                                to_rack_size=rack_size_2,
                                num_classes=num_classes,
                                observation_space_dim=observation_space_dim,
                                action_space_dim=action_space_dim,
                                is_goalpattern_fixed=True,
                                is_curriculum_lr=False)
    env.goal_pattern = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 2, 3, 3, 3], ])
    s = env.reset()
    act = env.sample()
    print(s)
    print(act)
    state, reward, is_done, _ = env.step(act)
    print(reward)
    print(state)
    print(state.fillable_slots)
