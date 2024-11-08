import numpy as np
import copy

from shapely.geometry.polygon import Point
from torch.distributions.utils import lazy_property

from huri.components.utils.matlibplot_utils import Plot
from huri.learning.env.env_meta import Gym_Proto, spaces
from huri.definitions.rack_def import rectangle_polygon
from huri.learning.env.arrangement_planning_rack.utils import (get_fillable_movable,
                                                               isdone,
                                                               get_random_states,
                                                               get_random_goal_pattern)
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


class RackState:
    """
    The state of rack. It provides convenient API
    """
    _cache = {}

    def __init__(self, state):
        self._state = np.asarray(state, dtype=int)

    @property
    # (1-5 row: current state, 6-10: fillable slots, 11-15: movable slots)
    def state(self):
        return self._state

    @property
    def abs_state(self):
        """
        Abstract state
        :return:
        """
        return np.vstack((self._state, self.fillable_slots, self.movable_slots))

    def _cal_fillable_movable_slots(self):
        """
        An internal function, do not explicitly call it.
        Calculate movable and fillable slots; and cache the results.

        Returns
        -------

        """
        fillable_slots, movable_slots = get_fillable_movable(self._state)
        state_str = str(self._state)
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
        new_state = self._state.copy()
        new_state[ind_a], new_state[ind_b] = self._state[ind_b], self._state[ind_a]
        return RackState(new_state)

    @lazy_property
    def feasible_action_set(self):
        """
        Get feasible action set

        Returns np.array
        -------

        """
        state = self._state.copy()
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
        state_str = str(self._state)
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
        state_str = str(self._state)
        if state_str not in RackState._cache:
            self._cal_fillable_movable_slots()
        return RackState._cache[state_str]["movable_slots"]

    def __sub__(self, other) -> np.ndarray:
        assert type(self) == type(other)
        return self.copy(self._state - other._state)

    def __getitem__(self, item):
        return self._state[item]

    def __eq__(self, other):
        assert type(self) == type(other)
        return np.array_equal(self._state, other._state)

    def __lt__(self, other):
        return self._state < other

    def __gt__(self, other):
        return self._state > other

    def __le__(self, other):
        return self._state <= other

    def __ge__(self, other):
        return self._state >= other

    def __repr__(self):
        return np.array2string(self._state)

    def copy(self, state=None):
        if state is None:
            state = self._state.copy()
        else:
            state = copy.deepcopy(state)
        return RackState(state=state)


class RackStatePlot:
    COLOR = {1: "mediumorchid",
             2: "royalblue",
             3: "gray",
             4: "lightgreen",
             "alpha": .2,
             "line_color": "silver"
             }

    def __init__(self, goal_pattern: RackState,
                 color: dict = None,
                 plot: Plot = None):
        if isinstance(goal_pattern, np.ndarray):
            self._goal_pattern = RackState(goal_pattern.copy())
        elif isinstance(goal_pattern, RackState):
            self._goal_pattern = goal_pattern.copy()
        else:
            raise Exception("Undefined input type")

        # setup color
        if color is None or not isinstance(color, dict):
            self.color = RackStatePlot.COLOR
        else:
            self.color = color
        if "alpha" not in self.color:
            self.color["alpha"] = RackStatePlot.COLOR["alpha"]
        if "line_color" not in self.color:
            self.color["line_color"] = RackStatePlot.COLOR["line_color"]

        # rack state shape
        self.state_shape = self._goal_pattern._state.shape
        self.slot_dim = np.array([6, 6])
        self.slot_centers = np.zeros((*self.state_shape, 2))
        self.slot_polygons = np.ndarray(shape=self.state_shape, dtype=object)
        self.num_classes = np.max(self._goal_pattern._state)
        for _x_ind, _x in enumerate(np.arange(self.state_shape[0]) * self.slot_dim[0]):
            for _y_ind, _y in enumerate(np.arange(self.state_shape[1]) * self.slot_dim[1]):
                self.slot_centers[_x_ind, _y_ind] = np.array([_x, _y])
                self.slot_polygons[_x_ind, _y_ind] = rectangle_polygon(np.array([_x, _y]), self.slot_dim)

        # init plot
        if plot is None:
            self.plot = Plot(is_axis=False, dpi=300)
        else:
            self.plot = plot

    def _plot_state(self, state: RackState, axe):
        tube_indices = np.asarray(np.where(state > 0)).T
        for ind in tube_indices:
            tube_type = state[ind[0], ind[1]]
            tube_color = self.color[tube_type]
            tube_pos = self.slot_centers[ind[0], ind[1]]
            tube_polygon = Point(tube_pos[0], tube_pos[1]).buffer(
                np.min(self.slot_dim) / np.sqrt(2) / 2
            )
            axe.plot(*tube_polygon.exterior.xy, color=tube_color)

    def _plot_rack(self, axe):
        goal_pattern_np = self._goal_pattern._state
        for ind in np.array(np.meshgrid(np.arange(self.state_shape[0]), np.arange(self.state_shape[1]))).T.reshape(-1,
                                                                                                                   2):
            tube_type = goal_pattern_np[ind[0], ind[1]]
            if tube_type > 0:
                tube_color = self.color[tube_type]
                axe.fill(*self.slot_polygons[ind[0], ind[1]].exterior.xy, alpha=self.color["alpha"], fc=tube_color,
                         ec='none')
            axe.plot(*self.slot_polygons[ind[0], ind[1]].exterior.xy, color=self.color["line_color"])

    def _plot_arrow_between_2states(self, state_pr, state, axe):
        movemap = state - state_pr
        if isinstance(movemap, RackState):
            movemap = movemap._state
        move_to_ind = np.concatenate(np.where(movemap > 0))
        move_from_ind = np.concatenate(np.where(movemap < 0))
        if len(move_to_ind) < 1 or len(move_from_ind) < 1:
            return
        goal_pos = self.slot_centers[move_to_ind[0], move_to_ind[1]]
        start_pos = self.slot_centers[move_from_ind[0], move_from_ind[1]]
        direction = np.array(goal_pos) - np.array(start_pos)
        direction = direction - 3 * direction / np.linalg.norm(direction)
        axe.arrow(*start_pos,
                  *direction,
                  width=0.5,
                  head_width=2,
                  linestyle='dashed',
                  head_length=3, fc='k', ec='k', zorder=100)

    def plot_bg(self):
        self._plot_rack(axe=self.plot.axes[-1])
        return self.plot.get_img()

    def plot_states(self, rack_states, row=6):
        # length of the eack
        state_num = len(rack_states)
        # set layout
        fig, axes = plt.subplots(int(np.ceil(state_num / (row))), min(state_num, row),
                                 figsize=(min(state_num, row) * 2, 3 * int(np.ceil(state_num / (row)))))
        if state_num == 1:
            axes = np.array([axes])
        # do not show the axis in the figure
        [i.axis('off') for i in axes.ravel()]
        # reshape the axs into min(state_num or row)
        axes = axes.reshape(-1, min(state_num, row))
        # plot
        for state_ind, state in enumerate(rack_states):
            axe_tmp = axes[state_ind // row, state_ind % row]
            # plot each rack state
            self._plot_rack(axe=axe_tmp)
            # plot tube
            self._plot_state(state, axe=axe_tmp)
            # plot arrow
            state_ind_pr = state_ind - 1
            if state_ind_pr >= 0:
                state_pr = rack_states[state_ind_pr]
                self._plot_arrow_between_2states(state_pr=state_pr, state=state, axe=axe_tmp)
        return fig


class RackArrangementEnv(Gym_Proto):
    def __init__(self,
                 rack_size=(5, 10),
                 num_classes=5,
                 observation_space_dim=(5, 10),
                 action_space_dim=10,
                 is_curriculum_lr=True,
                 is_goalpattern_fixed=True,
                 difficulty=2,
                 seed=777,
                 is_evl_mode=True):
        super(RackArrangementEnv, self).__init__()
        # init
        self.rack_size = rack_size
        self.num_classes = num_classes
        self.observation_space_dim = observation_space_dim
        self.action_space_dim = action_space_dim
        self.is_curriculum_lr = is_curriculum_lr
        self.is_goalpattern_fixed = is_goalpattern_fixed
        self.difficulty = difficulty

        # setup state, goal_pattern
        self.state = None
        self.goal_pattern = None

        # setup action space
        self.action_sapce = spaces.Discrete(self.action_space_dim)

        # set the seed for the env, ! action space and observation space should set seed as well
        self.action_sapce.seed(2 ** 31 - 3 * seed)
        self.seed(seed)

        # set the evaluation mode
        self.is_evl_mode = is_evl_mode
        # if self.is_evl_mode:
        #     print(f"It is the evaluation mode, there is {self.difficulty - 1} test tubes")

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

    def set_goal_pattern(self, goal_pattern):
        if isinstance(goal_pattern, np.ndarray):
            self.goal_pattern = RackState(goal_pattern.copy())
        elif isinstance(goal_pattern, RackState):
            self.goal_pattern = goal_pattern.copy()
        else:
            raise Exception("Undefined input type")

    def reset(self):
        if not self.is_goalpattern_fixed:
            # goal_pattern = self._np_random.randint(1, self.num_classes + 1, size=self.rack_size)
            goal_pattern = np.repeat(self._np_random.randint(1, self.num_classes + 1, size=[1, self.rack_size[1]]),
                                     self.rack_size[0], axis=0)
        else:
            if isinstance(self.goal_pattern, RackState):
                goal_pattern = self.goal_pattern._state
            elif isinstance(self.goal_pattern, np.ndarray):
                goal_pattern = self.goal_pattern
            else:
                raise Exception("No support goal state")
        if not self.is_curriculum_lr:
            initstate = get_random_states(self.rack_size, goal_pattern, np_random=self._np_random)
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
            initstate = get_random_states(self.rack_size, goal_pattern,
                                          min_obj_num=min_obj_num,
                                          obj_num=feasible_diff,
                                          np_random=self._np_random)
        return self.reset_state_goal(init_state=initstate, goal_pattern=goal_pattern)

    def _expr_action(self, action: int):
        rack_size = self.rack_size
        selected_obj = action % np.prod(rack_size)
        selected_obj_row = selected_obj // rack_size[1]
        selected_obj_column = selected_obj % rack_size[1]
        goal_pos = action // np.prod(rack_size)
        goal_pos_row = goal_pos // rack_size[1]
        goal_pos_column = goal_pos % rack_size[1]
        return (selected_obj_row, selected_obj_column), (goal_pos_row, goal_pos_column)

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
            reward += 2
        if is_move_out_pattern:
            reward -= 3
        if not is_move_to_pattern:
            reward -= 1
        return reward

    def sample(self, state: RackState = None):
        if state is None:
            state = self.state
        feasible_actions = state.feasible_action_set
        if len(feasible_actions) == 0:
            raise Exception
        else:
            return self._np_random.choice(feasible_actions)

    def step(self, action: int):
        pick_id, place_id = self._expr_action(action)

        # get current state
        curr_state = self.state
        goal_pattern = self.goal_pattern
        self.rack_state_history.append(curr_state.copy())

        # update the state
        nxt_state = curr_state.swap(ind_a=pick_id, ind_b=place_id)
        self.state = nxt_state

        # check if the new state is illegal state
        nxt_fillable_slots, nxt_movable_slots = self.state.fillable_slots, self.state.movable_slots
        if np.sum(nxt_fillable_slots) == 0 or np.sum(nxt_movable_slots) == 0:
            reward = -50
            self.reward_history.append(reward)
            return self.state, reward, True, {
                "reward_history": self.reward_history
            }

        is_finished = isdone(nxt_state.copy(), goal_pattern)
        # get reward of the action
        reward = self._get_reward(is_finished, self.rack_state_history[-1], nxt_state.copy(), goal_pattern)
        self.reward_history.append(reward)

        return nxt_state, reward, is_finished, {
            "reward_history": self.reward_history
        }

    def is_finished(self, state: RackState):
        return isdone(state, goalpattern=self.goal_pattern)

    def to_state(self, state_np: np.ndarray):
        return RackState(state_np)

    def increase_diffculty(self):
        self.difficulty = self.difficulty + 2

    def copy(self):
        return copy.deepcopy(self)


if __name__ == "__main__":
    rack_size = (5, 10)
    num_classes = 2
    observation_space_dim = (num_classes * 2, *rack_size)
    action_space_dim = np.prod(rack_size) ** 2
    env = RackArrangementEnv(rack_size=rack_size,
                             num_classes=num_classes,
                             observation_space_dim=observation_space_dim,
                             action_space_dim=action_space_dim,
                             is_goalpattern_fixed=False,
                             is_curriculum_lr=True)
    s = env.reset()
    act = env.sample()
    state, reward, is_done, _ = env.step(act)
    print(env.goal_pattern)
    print(s)
    print(state)
