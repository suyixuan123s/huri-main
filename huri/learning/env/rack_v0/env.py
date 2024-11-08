import time

import numpy as np
import copy

from shapely.geometry.polygon import Point
from torch.distributions.utils import lazy_property

from huri.components.utils.matlibplot_utils import Plot
from huri.learning.env.env_meta import Gym_Proto, spaces
from huri.definitions.rack_def import rectangle_polygon
from huri.learning.env.rack_v0.utils import (get_possible_actions,
                                             get_fillable_movable,
                                             isdone,
                                             get_random_states,
                                             get_random_goal_pattern)
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

GOAL_PATTERN_5x10 = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 2, 2],
                              [1, 1, 0, 0, 0, 0, 0, 0, 2, 2],
                              [1, 1, 0, 0, 0, 0, 0, 0, 2, 2],
                              [1, 1, 0, 0, 0, 0, 0, 0, 2, 2],
                              [1, 1, 0, 0, 0, 0, 0, 0, 2, 2]])


def to_action(rack_size, pick_id, place_id):
    place_id_int = place_id[:, 0] * rack_size[1] + place_id[:, 1]
    pick_id_int = pick_id[:, 0] * rack_size[1] + pick_id[:, 1]
    return np.sort(place_id_int * np.prod(rack_size) + pick_id_int)


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
        return self._state

    def _cal_possible_actions(self, debug=False):
        """
        An internal function, do not explicitly call it.
        Calculate movable and fillable slots; and cache the results.

        Returns
        -------

        """
        a = time.time()
        possible_actions, ma, fa = get_possible_actions(self._state)
        if debug:
            b = time.time()
            print("Time consumtion is ", b - a)
            fb, mb = get_fillable_movable(self._state)
            # print(ma[0])
            # print(fa[fa[:, 0] == 0])
            print(self.state)
            for _ in np.unique(fa[:, 0]):
                print(_)
                fillable_tst_mtx = np.zeros((5, 10))
                print(fa[fa[:, 0] == _])
                fillable_tst_mtx[(fa[fa[:, 0] == _][:, 1], tuple(fa[fa[:, 0] == _][:, 2]))] = 1
                movable_tst_mtx = np.zeros((5, 10))
                movable_tst_mtx[ma[_][0], ma[_][1]] = 1
                print(np.dstack(np.where((fillable_tst_mtx - fb) != 0)))
                print("---" * 10)
                drawer = RackStatePlot(np.zeros((5, 10)))
                tst_state = self._state.copy()
                tst_state[ma[_][0], ma[_][1]] = 0
                img = drawer.plot_states([tst_state], movable_mtx=movable_tst_mtx,
                                         fillable_mtx=fillable_tst_mtx).get_img()
                img2 = drawer.plot_states([self._state], fillable_mtx=fb).get_img()
                cv2.imshow("safd", np.concatenate((img, img2), axis=1))
                cv2.waitKey(0)

        state_str = str(self._state)
        RackState._cache[state_str] = {
            "possible_actions": possible_actions,
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
        state_str = str(state)
        if state_str not in RackState._cache:
            self._cal_possible_actions()
        possible_actions = RackState._cache[state_str]["possible_actions"]
        rack_size = state.shape
        return to_action(rack_size, possible_actions[:, 0:2], possible_actions[:, 2:4])

    # @property
    # def fillable_slots(self):
    #     """
    #     Return feasible slots indices
    #
    #     Returns np.array
    #     -------
    #
    #     """
    #     state_str = str(self._state)
    #     if state_str not in RackState._cache:
    #         self._cal_fillable_movable_slots()
    #     return RackState._cache[state_str]["fillable_slots"]
    #
    # @property
    # def movable_slots(self):
    #     """
    #     Return movable slots indices
    #
    #     Returns np.array
    #     -------
    #
    #     """
    #     state_str = str(self._state)
    #     if state_str not in RackState._cache:
    #         self._cal_fillable_movable_slots()
    #     return RackState._cache[state_str]["movable_slots"]

    @lazy_property
    def num_classes(self):
        nc = np.unique(self._state)
        return nc[nc > 0]

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
             5: "red",
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

        # plot the rack state
        self._plot_rack(self.plot.axes[-1])

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

    def _plot_cons(self, state: RackState, axe, color="red"):
        tube_indices = np.asarray(np.where(state > 0)).T
        for ind in tube_indices:
            tube_color = color
            tube_pos = self.slot_centers[ind[0], ind[1]]
            tube_polygon = Point(tube_pos[0], tube_pos[1]).buffer(
                np.min(self.slot_dim) / np.sqrt(2) / 10
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

    def plot_states(self, rack_states, row=6, fillable_mtx=None, movable_mtx=None):
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
            # axe_tmp.imshow()
            # plot tube
            self._plot_state(state, axe=axe_tmp)
            # plot arrow
            state_ind_pr = state_ind - 1
            if state_ind_pr >= 0:
                state_pr = rack_states[state_ind_pr]
                self._plot_arrow_between_2states(state_pr=state_pr, state=state, axe=axe_tmp)
            if fillable_mtx is not None or movable_mtx is not None:
                if not isinstance(state, RackState):
                    rs = RackState(state)
                else:
                    rs = state
                # fillable_slots = rs.fillable_slots
                # movable_slots = rs.movable_slots
                if movable_mtx is not None:
                    self._plot_cons(movable_mtx, axe=axe_tmp, color="red")
                if fillable_mtx is not None:
                    self._plot_cons(fillable_mtx, axe=axe_tmp, color="green")
        return Plot(fig=fig)


class RackArrangementEnv(Gym_Proto):
    @staticmethod
    def F(state):
        if not isinstance(state, RackState):
            rs = RackState(state)
        else:
            rs = state
        return len(rs.feasible_action_set)

    @staticmethod
    def E(state, goal_pattern):
        if isinstance(state, RackState):
            state = state.state
        if isinstance(goal_pattern, RackState):
            number_classes = goal_pattern.num_classes
            goal_pattern = goal_pattern.state
        else:
            number_classes = np.unique(state)
            number_classes = number_classes[number_classes > 0]
        entropy = np.zeros(len(number_classes), dtype=int)
        for i, _ in enumerate(number_classes):
            entropy[i] = len(np.where(state[goal_pattern == _] == _)[0])
        return entropy

    @staticmethod
    def action_between_states(s_current, s_next):
        if isinstance(s_current, RackState):
            s_current = s_current.state
        if isinstance(s_next, RackState):
            s_next = s_next.state
        assert s_current.shape == s_next.shape
        move = s_next - s_current
        move_to_idx = np.where(move > 0)
        move_from_idx = np.where(move < 0)
        rack_size = s_current.shape
        pick_id_int = move_from_idx[0] * rack_size[1] + move_from_idx[1]
        place_id_int = move_to_idx[0] * rack_size[1] + move_to_idx[1]
        action_ids = place_id_int * np.prod(rack_size) + pick_id_int
        return action_ids.item()

    def __init__(self,
                 rack_size=(5, 10),
                 num_classes=5,
                 observation_space_dim=(5, 10),
                 action_space_dim=10,
                 is_curriculum_lr=True,
                 is_goalpattern_fixed=True,
                 scheduler=None,
                 seed=777, ):
        super(RackArrangementEnv, self).__init__()
        # init
        self.rack_size = rack_size
        self.num_classes = num_classes
        self.observation_space_dim = observation_space_dim
        self.action_space_dim = action_space_dim
        self.is_curriculum_lr = is_curriculum_lr
        self.is_goalpattern_fixed = is_goalpattern_fixed

        # setup state, goal_pattern
        self.state = None
        self.goal_pattern = None

        # setup action space
        self.action_sapce = spaces.Discrete(self.action_space_dim)

        # set the seed for the env, ! action space and observation space should set seed as well
        self.action_sapce.seed(2 ** 31 - 3 * seed)
        self.seed(seed)

        # setup scheduler
        if scheduler is None:
            scheduler = RackStateScheduler(num_classes=num_classes, rack_size=rack_size, np_random=self._np_random)
        self.scheduler: RackStateScheduler = scheduler

    def reset_state_goal(self, init_state, goal_pattern):
        self.set_goal_pattern(goal_pattern)
        state = self.reset_state(init_state)
        return state

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
        self.scheduler.max_state_level_per_class = min(np.bincount(self.goal_pattern.state.ravel())[1:])
        return self.goal_pattern

    def reset(self):
        if not self.is_goalpattern_fixed:
            # goal_pattern = self._np_random.randint(1, self.num_classes + 1, size=self.rack_size)
            goal_pattern = self.scheduler.gen_goal()
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
            initstate = self.scheduler.gen_state(goal_pattern)
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

    def __get_reward(self, is_finished, state_current, state_next, goal_pattern):
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

    def _get_reward(self, is_finished, state_current, state_next, goal_pattern):
        if is_finished:
            # return 5 * len(state_current[state_current > 0])
            return 50
        enp = np.sum(self.E(state_next, goal_pattern) - self.E(state_current, goal_pattern))
        f = np.clip(np.sum(self.F(state_next) - self.F(state_current)), -20, 20)
        # print(self.F(state_next), self.F(state_current))
        # print("entropy: ", enp, "feasible aciton:", f)
        # return enp + .1 * f - 1
        return enp + - 1

    def sample(self, state: RackState = None):
        if state is None:
            state = self.state
        feasible_actions = state.feasible_action_set
        if len(feasible_actions) == 0:
            raise Exception
        else:
            return self._np_random.choice(feasible_actions)

    def _step(self, pick_id, place_id):
        # get current state
        curr_state = self.state
        goal_pattern = self.goal_pattern
        self.rack_state_history.append(curr_state.copy())

        # update the state
        nxt_state = curr_state.swap(ind_a=pick_id, ind_b=place_id)
        self.state = nxt_state

        # check if the new state is illegal state
        nxt_feasible_actions = self.state.feasible_action_set
        if np.sum(nxt_feasible_actions) == 0:
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
            'reward_history': self.reward_history,
            'rack_history': self.rack_state_history
        }

    def step(self, action: int):
        pick_id, place_id = self._expr_action(action)
        return self._step(pick_id, place_id)

    def is_finished(self, state: RackState, goal_state: RackState = None):
        if goal_state is None:
            goal_state = self.to_state_np(self.goal_pattern)
        else:
            goal_state = self.to_state_np(goal_state)
        return isdone(self.to_state_np(state), goalpattern=goal_state)

    def to_state(self, state_np: np.ndarray):
        return RackState(state_np)

    def to_state_np(self, state: RackState):
        if isinstance(state, RackState):
            return state.state
        elif isinstance(state, np.ndarray):
            return state
        else:
            raise Exception("No support input state")

    def to_action(self, pick_id, place_id):
        rack_size = self.rack_size
        place_id_int = place_id[0] * rack_size[1] + place_id[1]
        pick_id_int = pick_id[0] * rack_size[1] + pick_id[1]
        return place_id_int * np.prod(rack_size) + pick_id_int

    def evaluate_traj(self, paths, goal_pattern, toggle_debug=False):
        reward_traj = []
        action_traj = []
        acc_reward = 0
        for p in range(0, len(paths) - 1)[::-1]:
            s = paths[p]
            s_n = paths[p + 1]
            finished = isdone(s_n.state if isinstance(s_n, RackState) else s_n,
                              goal_pattern.state if isinstance(goal_pattern, RackState) else goal_pattern)
            acc_reward += self._get_reward(finished, s, s_n, goal_pattern)
            reward_traj.append(acc_reward)
            action_traj.append(self.action_between_states(s, s_n))
            if toggle_debug:
                print(s)
                print(s_n)
                print(self._expr_action(action_traj[-1]))
        return paths, reward_traj[::-1], action_traj[::-1]

    def copy(self):
        return copy.deepcopy(self)

    def set_seed(self, seed=None):
        self.seed(seed)
        self.scheduler._np_random = self._np_random


class RackStateScheduler(object):
    def __init__(self, num_classes, rack_size, np_random=None, state_level=1, goal_level=1, class_level=1,
                 max_state_level_per_class=None):
        assert num_classes > 0
        self.num_classes = num_classes
        self.class_ids = np.arange(1, self.num_classes + 1)
        self.rack_size = rack_size
        self.goal_level = goal_level
        self.state_level = state_level
        self.class_level = min(class_level, num_classes)

        self.max_state_level_per_class = np.prod(
            self.rack_size) if max_state_level_per_class is None else max_state_level_per_class

        self._np_random = np_random if np_random is not None else np.random

    def gen_state(self, goal_pattern):
        if isinstance(goal_pattern, RackState):
            goal_pattern_np = goal_pattern.state
        elif isinstance(goal_pattern, np.ndarray):
            goal_pattern_np = goal_pattern
        else:
            raise Exception
        goalpattern_ravel = goal_pattern_np.ravel()
        max_len = len(np.where(goalpattern_ravel > 0)[0]) + 1
        feasible_diff = min(max_len, self.state_level)
        min_obj_num = self.state_level
        state = get_random_states(self.rack_size, goal_pattern_np,
                                  min_obj_num=min_obj_num,
                                  obj_num=feasible_diff,
                                  np_random=self._np_random,
                                  max_num_class=self.class_level)

        return RackState(state)

    def gen_goal(self):
        chosed_ids = np.random.choice(self.class_ids, min(self.goal_level, self.num_classes), replace=False)
        # print("chosed ids", chosed_ids)
        # goal_pattern = self._np_random.randint(0, self.goal_level + 1, size=self.rack_size)
        p_0 = max(.3 * np.cos(np.pi / 2 * self.state_level / (np.prod(self.rack_size) - 1)), 0)
        # print("p_o is ", p_0)
        goal_pattern = self._np_random.choice(np.arange(0, self.goal_level + 1), size=self.rack_size,
                                              p=[p_0] + [(1 - p_0) / self.goal_level] * self.goal_level)

        for ind, i in enumerate(chosed_ids):
            if ind + 1 in goal_pattern:
                goal_pattern[goal_pattern == ind + 1] = -i
        return RackState(-goal_pattern)
        # return np.repeat(self._np_random.randint(1, self.num_classes + 1, size=[1, self.rack_size[1]]),
        #                  self.rack_size[0], axis=0)

    def update_goal_level(self):
        level = self.goal_level + 1
        self.goal_level = min(level, self.num_classes)
        print(f"Goal Level Update: {level - 1} -> {self.goal_level}")

    def update_state_level(self):
        level = self.state_level + 1
        self.state_level = min(level, self.max_state_level_per_class * self.class_level)
        print(f"State Level Update: {level - 1} -> {self.state_level}")

    def update_class_level(self):
        level = self.class_level + 1
        self.class_level = min(level, num_classes)
        print(f"Class Level Update: {level - 1} -> {self.class_level}")

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return f"Rack dimension is {self.rack_size}, current state level is {self.state_level}, class level is {self.class_level}"


class StripeRackStateScheduler(RackStateScheduler):
    def gen_goal(self):
        chosed_ids = np.random.choice(self.class_ids, min(self.goal_level, self.num_classes), replace=False)
        # print("chosed ids", chosed_ids)
        # goal_pattern = self._np_random.randint(0, self.goal_level + 1, size=self.rack_size)
        p_0 = max(.3 * np.cos(np.pi / 2 * self.state_level / (np.prod(self.rack_size) - 1)), 0)
        # print("p_o is ", p_0)
        goal_pattern = self._np_random.choice(np.arange(0, self.goal_level + 1), size=self.rack_size[1],
                                              p=[p_0] + [(1 - p_0) / self.goal_level] * self.goal_level)

        for ind, i in enumerate(chosed_ids):
            if ind + 1 in goal_pattern:
                goal_pattern[goal_pattern == ind + 1] = -i
        return RackState(np.repeat(-goal_pattern[None, :], self.rack_size[0], axis=0))


if __name__ == "__main__":
    import cv2

    rack_size = (5, 10)
    num_classes = 5
    observation_space_dim = (num_classes * 2, *rack_size)
    action_space_dim = np.prod(rack_size) ** 2

    env = RackArrangementEnv(rack_size=rack_size,
                             num_classes=num_classes,
                             observation_space_dim=observation_space_dim,
                             action_space_dim=action_space_dim,
                             is_goalpattern_fixed=True,
                             is_curriculum_lr=True,
                             seed=np.random.randint(999))

    env.set_goal_pattern(GOAL_PATTERN_5x10)
    scheduler = env.scheduler
    scheduler.max_state_level_per_class = min(np.bincount(GOAL_PATTERN_5x10.ravel())[1:])
    # while 1:
    #     scheduler.update_state_level()
    #     print(scheduler.state_level == (scheduler.max_state_level_per_class * scheduler.class_level))
    #     if scheduler.state_level == (scheduler.max_state_level_per_class * scheduler.class_level):
    #         scheduler.update_class_level()
    #     print(scheduler)
    #     s = env.reset()
    #     a = env.sample()
    #     nxt_state, reward, is_finished, _ = env.step(a)
    #     print("reward is ", reward)
    #     drawer = RackStatePlot(GOAL_PATTERN_5x10, )
    #     fig = drawer.plot_states([s, nxt_state, ]).get_img()
    #     cv2.imshow("test", fig)
    #     cv2.waitKey(0)
    #
    # exit(0)
    # act = env.sample()
    # state, reward, is_done, _ = env._step((4, 9 - 1), (4, 9 - 6))
    # print(env.goal_pattern)
    # print(reward)
    # rsp = RackStatePlot(env.goal_pattern, )
    # img = rsp.plot_states([s, state], plot_movable=True, plot_fillable=True).get_img()
    # cv2.imshow("tst", img)
    # cv2.waitKey(0)
    # print(s)
    # print(state)
    # failed = 0
    # for i in range(100):
    #     try:
    #         rss = StripeRackStateScheduler(num_classes=num_classes, rack_size=rack_size, state_level=30, goal_level=2)
    #         # rss.update_goal_level()
    #         rss.update_state_level()
    #         goal = rss.gen_goal()
    #         state = rss.gen_state(goal)
    #         print(goal)
    #         print(state, len(state[state.state > 0]))
    #     except:
    #         failed += 1
    # print(f"number of failed is {failed}")

    from huri.learning.A_start_teacher.A_star_teacher import TubePuzzle

    rss = StripeRackStateScheduler(num_classes=num_classes, rack_size=rack_size, state_level=12,
                                   goal_level=2)
    goal = rss.gen_goal()
    if len(np.unique(goal.state)) <= 1:
        exit(-1)
    state = rss.gen_state(goal)
    a = TubePuzzle(elearray=state.state, goalpattern=goal.state)
    is_find, path = a.atarSearch(infeasible_dict={}, max_iter_cnt=300)

    from numpy import array

    path = [array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                   [1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]), array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                                            [1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                                                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]),
            array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]), array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                            [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                                                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]),
            array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]), array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0]]),
            array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])]

    if is_find:
        dataset = []
        action_vec = np.zeros(2500)
        action_vec[:] = -np.inf
        path_traj, reward_traj, action_traj = env.evaluate_traj(path, goal, toggle_debug=True)
        for i in range(len(action_traj)):
            label = action_vec.copy()
            label[action_traj[i]] = reward_traj[i]
            dataset.append([path_traj[i], goal, label])
        print("-k" * 30)
        for idx in range(len(dataset)):
            state, goal, label = dataset[idx]
            state = state
            goal = goal
            label = label
            print(state)
            print(env._expr_action(label.argmax()))
            print(label.argmax())
            print(label.max())

        drawer = RackStatePlot(goal)
        fig = drawer.plot_states(path).get_img()
        cv2.imshow("window", fig)
        cv2.waitKey(0)
