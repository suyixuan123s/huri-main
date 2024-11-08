import time
from typing import Union

import numba
import numpy as np
import copy

import torch
from shapely.geometry.polygon import Point, Polygon
# from torch.distributions.utils import lazy_property

from huri.components.utils.matlibplot_utils import Plot
from huri.learning.env.env_meta import Gym_Proto, spaces
from huri.learning.env.rack_v3.utils import (get_possible_actions,
                                             get_fillable_movable,
                                             isdone,
                                             get_random_states,
                                             extract_submatrix_mask,
                                             get_random_goal_pattern,
                                             check_pattern_in_matrix,
                                             get_fillable,
                                             get_possible_actions_condition_set)
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
import cv2
from shapely.geometry import Polygon, LineString

matplotlib.use("Agg")

GOAL_PATTERN_5x10 = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 2, 2],
                              [1, 1, 0, 0, 0, 0, 0, 0, 2, 2],
                              [1, 1, 0, 0, 0, 0, 0, 0, 2, 2],
                              [1, 1, 0, 0, 0, 0, 0, 0, 2, 2],
                              [1, 1, 0, 0, 0, 0, 0, 0, 2, 2]])


def rectangle_polygon(rect_center, rect_extent):
    """Generate a rectangle (shapely.geometry.Polygon)"""
    lu = rect_center - rect_extent / 2
    ru = rect_center + np.array([rect_extent[0], -rect_extent[1]]) / 2
    lt = rect_center + np.array([-rect_extent[0], +rect_extent[1]]) / 2
    rt = rect_center + rect_extent / 2
    return Polygon([lu, ru, lt, rt]).convex_hull


def to_action(rack_size, pick_id, place_id):
    bins = np.cumsum(np.concatenate(([0], np.arange(np.prod(rack_size) - 1, 0, -1))))

    if isinstance(pick_id, list) or isinstance(pick_id, tuple):
        pick_id = np.array([pick_id])
    if isinstance(place_id, list) or isinstance(place_id, tuple):
        place_id = np.array([place_id])

    # if pick_id.shape[0] != 2:
    pick_id = pick_id.T

    # if place_id.shape[0] != 2:
    place_id = place_id.T

    pick_id_r = np.ravel_multi_index(pick_id, rack_size)
    place_id_r = np.ravel_multi_index(place_id, rack_size)
    pp_id = np.stack((pick_id_r, place_id_r))
    min_pp_id = np.min(pp_id, axis=0)
    max_pp_id = np.max(pp_id, axis=0)
    return (bins[min_pp_id] + max_pp_id - min_pp_id - 1)


def from_action(rack_size, abs_act_id) -> (np.ndarray, np.ndarray):
    bins = np.cumsum(np.arange(np.prod(rack_size) - 1, 0, -1))
    if bins[-1] > abs_act_id >= 0:
        pick_id = np.digitize(abs_act_id, bins)
        if pick_id == 0:
            place_id = abs_act_id + 1
        else:
            place_id = abs_act_id + 1 - bins[pick_id - 1] + pick_id
        return np.unravel_index(pick_id, rack_size), np.unravel_index(place_id, rack_size)
    else:
        return None


def from_action_to_mat(rack_size, abs_act_id) -> np.ndarray:
    bins = np.cumsum(np.arange(np.prod(rack_size) - 1, 0, -1))
    if bins[-1] > abs_act_id >= 0:
        pick_id = np.digitize(abs_act_id, bins)
        if pick_id == 0:
            place_id = abs_act_id + 1
        else:
            place_id = abs_act_id + 1 - bins[pick_id - 1] + pick_id
        idmat = np.eye(np.prod(rack_size), dtype=int)
        idmat[:, (pick_id, place_id)] = idmat[:, (place_id, pick_id)]
        return idmat
    else:
        return None


def to_flatten_ids(rack_size, p_id):
    return p_id[:, 0] * rack_size[1] + p_id[:, 1]


class RackState:
    """
    The state of rack. It provides convenient API
    """
    _cache = {}

    @staticmethod
    # Create a one-hot array of zeros of appropriate size
    def one_hot(x, num_categories):
        one_hot = np.zeros((num_categories + 1, x.shape[0], x.shape[1]))
        # Use fancy indexing to fill the right spots with ones
        one_hot[x, np.arange(x.shape[0])[:, None], np.arange(x.shape[1])] = 1
        return one_hot[1:]

    @classmethod
    def get_feasible_action_set(cls, state):
        state_s = state.copy()
        state_s[state_s > 0] = 1
        state_str = str(state_s)
        if state_str not in cls._cache:
            possible_actions, ma, fa = get_possible_actions(state_s)
            rack_size = state.shape
            possible_abs_actions = to_action(rack_size, possible_actions[:, 0:2], possible_actions[:, 2:4])

            cls._cache[state_str] = {
                "possible_abs_actions": possible_abs_actions,
                "possible_actions": possible_actions.astype(np.uint8),
            }
        else:
            possible_abs_actions = cls._cache[state_str]["possible_abs_actions"]
        return possible_abs_actions

    @classmethod
    def get_feasible_action_set_detail(cls, state):
        state_s = state.copy()
        state_s[state_s > 0] = 1
        state_str = str(state_s)
        if state_str not in cls._cache:
            possible_actions, ma, fa = get_possible_actions(state_s)
            rack_size = state.shape
            possible_abs_actions = to_action(rack_size, possible_actions[:, 0:2], possible_actions[:, 2:4])
            cls._cache[state_str] = {
                "possible_abs_actions": possible_abs_actions,
                "possible_actions": possible_actions,
            }
        else:
            possible_actions = cls._cache[state_str]["possible_actions"]
        return possible_actions

    @classmethod
    def get_obs_act_dim_by_size(cls, size: Union[np.ndarray, list, tuple]) -> (int, int):
        if isinstance(size, list) or isinstance(size, tuple) or isinstance(size, np.ndarray):
            obs_dim = size
            fobs_dim = np.prod(size)
            act_dim = int((fobs_dim - 1) * fobs_dim / 2)
            return obs_dim, act_dim
        else:
            raise Exception("Invalid input ")

    @classmethod
    def compress_abs_state(cls, abs_state):
        compressed = abs_state[np.tril_indices(abs_state.shape[0])]
        return compressed

    @classmethod
    def decompress_abs_state(cls, compressed, n):
        # Decompress the matrix by constructing a symmetric matrix
        decompressed = np.zeros((n, n))
        decompressed[np.tril_indices(n)] = compressed
        decompressed = decompressed + decompressed.T - np.diag(np.diag(decompressed))
        return decompressed

    def get_feasible_action_condition_set(self, condition_set):
        state_s = self.state.copy()
        state_s[state_s > 0] = 1
        possible_actions, ma, fa = get_possible_actions_condition_set(state_s, condition_set)
        rack_size = state_s.shape
        possible_abs_actions = to_action(rack_size, possible_actions[:, 0:2], possible_actions[:, 2:4])
        return possible_abs_actions

    def __init__(self, state: np.ndarray):
        assert isinstance(state, (np.ndarray, RackState)), f'state must be np.ndarry instead of {type(state)}'
        if isinstance(state, RackState):
            self._state = state.state.copy().astype(int)
        else:
            self._state = np.asarray(state, dtype=int)
        self._state_v = self.state.ravel()
        self._size = self._state.shape

    @property
    # (1-5 row: current state, 6-10: fillable slots, 11-15: movable slots)
    def state(self):
        return self._state

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return self._size

    @property
    def abs_size(self):
        abs_size = np.prod(self.size)
        return (abs_size, abs_size)

    @property
    def vec(self):
        return self._state_v

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

        return possible_actions

    def _cal_fillable_movable_slots(self):
        fb, mb = get_fillable_movable(self._state)
        movable = np.zeros(self._state.shape)
        return fb, mb

    def abs_state(self, n):
        """
        Abstract state
        :return:
        """
        return self.one_hot(self._state, n)

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
        new_state[ind_a[0], ind_a[1]], new_state[ind_b[0], ind_b[1]] = self._state[ind_b[0], ind_b[1]], self._state[
            ind_a[0], ind_a[1]]
        return RackState(new_state)

    # @lazy_property
    @property
    def feasible_action_set(self):
        """
        Get feasible action set

        Returns np.array
        -------

        """
        if np.any(self._state > 0):
            return self.get_feasible_action_set(self._state)
        else:
            return np.array([])

    @property
    def feasible_action_set_detail(self):
        """
        Get feasible action set

        Returns np.array
        -------

        """
        if np.any(self._state > 0):
            return self.get_feasible_action_set_detail(self._state)
        else:
            return np.array([])

    # @lazy_property
    @property
    def to_abs_state(self):
        state = self._state.copy()
        state_s = state.copy()
        state_s[state_s > 0] = 1
        possible_actions = self._cal_possible_actions()
        rack_size = state.shape
        # get ids
        pick_ids = np.ravel_multi_index(possible_actions[:, 0:2].T, rack_size)
        place_ids = np.ravel_multi_index(possible_actions[:, 2:4].T, rack_size)
        # create adjacent matrix
        m = np.zeros((np.prod(rack_size), np.prod(rack_size)), dtype=int)
        m[(pick_ids, place_ids)] = state.flatten()[pick_ids]
        m[(place_ids, pick_ids)] = state.flatten()[pick_ids]
        return m

    @property
    def compressed_abs_state(self):
        return self.compress_abs_state(self.to_abs_state)

    # @lazy_property
    @property
    def reflections(self, ):
        return [RackState(np.flipud(self.state).copy()),
                RackState(np.fliplr(self.state).copy()),
                RackState(np.flipud(np.fliplr(self.state)).copy())]

    # @lazy_property
    @property
    def nonzero_count(self):
        return np.count_nonzero(self.state)

    @property
    def fillable_movable_region(self):
        return get_fillable_movable(self.state)

    # @lazy_property
    # def adjacency_matrix(self):
    #     state = self._state.copy()
    #     rack_size = state.shape
    #     state_str = str(state)
    #     if state_str not in RackState._cache:
    #         self._cal_possible_actions()
    #     possible_actions = RackState._cache[state_str]["possible_actions"]
    #     m = np.zeros((np.prod(rack_size), np.prod(rack_size)), dtype=bool)
    #     pick_ids_flatten = to_flatten_ids(rack_size, possible_actions[:, 0:2])
    #     place_ids_flatten = to_flatten_ids(rack_size, possible_actions[:, 2:4])
    #     m[(pick_ids_flatten, place_ids_flatten)] = 1
    #     m[(place_ids_flatten, pick_ids_flatten)] = 1
    #     return m

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

    # @lazy_property
    @property
    def num_classes(self):
        nc = np.unique(self._state)
        return nc[nc > 0]

    def __sub__(self, other) -> np.ndarray:
        assert type(self) == type(other)
        return self.copy(self._state - other._state)

    def __getitem__(self, item):
        return self._state[item]

    def __eq__(self, other):
        if isinstance(other, (np.ndarray, int, float)):
            return self._state == other
        elif isinstance(other, RackState):
            return np.array_equal(self._state, other._state)
        else:
            raise ValueError("Undefined input type")

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

    def __array__(self):
        return self.state

    def copy(self, state=None):
        if state is None:
            state = self._state.copy()
        else:
            state = copy.deepcopy(state)
        return RackState(state=state)


class RackStatePlot:
    COLOR = {
        # 1: "mediumorchid",
        #      2: "royalblue",
        #      3: "lightgreen",
        # for draw
        1: "red",
        2: "blue",
        3: "green",
        # for draw
        # 4: "mediumorchid",
        5: "black",
        4: "gold",
        # "alpha": .2,
        # for draw
        "alpha": .5,
        "bg_alpha": .2,
        "line_color": "black"
    }
    LINE_WIDTH = 8

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

        # rotate goal pattern state to make it same as the view as the print
        self._goal_pattern._state = self._aligned_state(self._goal_pattern._state)

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

    def _aligned_state(self, state_np):
        return np.rot90(np.asarray(state_np), k=-1)

    def _plot_state(self, state: RackState, axe, toggle_fill=False):
        tube_indices = np.asarray(np.where(state > 0)).T
        for ind in tube_indices:
            tube_type = state[ind[0], ind[1]]
            tube_pos = self.slot_centers[ind[0], ind[1]]
            tube_polygon = Point(tube_pos[0], tube_pos[1]).buffer(
                np.min(self.slot_dim) / np.sqrt(2) / 2
            )
            if tube_type > 999:
                axe.plot(*tube_polygon.exterior.xy,
                         color='black', linestyle='--')
            elif tube_type == 99:
                axe.plot(*tube_polygon.exterior.xy,
                         color='black', linestyle='--')
            elif tube_type == 100:
                axe.plot(*tube_polygon.exterior.xy,
                         color='black', linestyle='-')
            else:
                tube_color = self.color[tube_type]
                if toggle_fill:
                    axe.fill(*tube_polygon.exterior.xy,
                             alpha=self.color["alpha"],
                             fc=tube_color,
                             ec='none')
                else:
                    axe.plot(*tube_polygon.exterior.xy, color=tube_color)

    def _plot_state_circle(self, state: RackState, axe, toggle_fill=False):
        tube_indices = np.asarray(np.where(state > 0)).T
        circle_array = np.ndarray(shape=state.shape, dtype=object)
        for ind in tube_indices:
            tube_type = state[ind[0], ind[1]]
            tube_color = self.color[tube_type]
            tube_pos = self.slot_centers[ind[0], ind[1]]
            tube_circle = plt.Circle((tube_pos[0], tube_pos[1]),
                                     np.min(self.slot_dim) / np.sqrt(2) / 2,
                                     color=tube_color,
                                     alpha=self.color["alpha"],
                                     edgecolor='none',
                                     linewidth=0)
            axe.add_patch(tube_circle)
            circle_array[ind[0], ind[1]] = tube_circle
        return circle_array

    def _plot_cons(self, state: RackState, axe, color="red"):
        tube_indices = np.asarray(np.where(state > 0)).T
        for ind in tube_indices:
            tube_color = color
            tube_pos = self.slot_centers[ind[0], ind[1]]
            tube_polygon = Point(tube_pos[0], tube_pos[1]).buffer(
                np.min(self.slot_dim) / np.sqrt(2) / 10
            )
            axe.plot(*tube_polygon.exterior.xy, color=tube_color)

    def _plot_rack(self, axe, goal_pattern=None):
        if goal_pattern is None:
            goal_pattern_np = self._goal_pattern._state
            state_shape = self.state_shape
        else:
            goal_pattern_np = np.asarray(goal_pattern)
            state_shape = goal_pattern_np.shape
        for ind in np.array(np.meshgrid(np.arange(state_shape[0]), np.arange(state_shape[1]))).T.reshape(-1, 2):
            tube_type = goal_pattern_np[ind[0], ind[1]]
            if tube_type > 0:
                tube_color = self.color[tube_type]
                c = self.color.get('bg_alpha', self.color["alpha"])
                axe.fill(*self.slot_polygons[ind[0], ind[1]].exterior.xy, alpha=c, fc=tube_color,
                         ec='none')
            elif tube_type < 0:
                # font_properties = {'family': 'Arial', 'fontsize': 12, 'weight': 'bold'}
                # font_properties = {'family': 'Times New Roman'}
                # x_axis, y_axis = self.slot_centers[ind[0], ind[1]]
                # axe.text(x_axis, y_axis, '?', fontsize=35, ha='center', va='center', fontdict=font_properties)
                axe.fill(*self.slot_polygons[ind[0], ind[1]].exterior.xy, alpha=.1, fc='w',
                         ec='none')

                def draw_diagonal_shadows_in_cell(cell_polygon,
                                                  ax,
                                                  density=10,
                                                  linewidth=1,
                                                  color='black'):
                    """
                    Draw diagonal shadow lines within a single cell.
                    :param cell_polygon: Shapely Polygon object representing the cell
                    :param ax: Matplotlib axis object where the grid will be drawn
                    :param density: The density of the shadow lines
                    """
                    minx, miny, maxx, maxy = cell_polygon.bounds
                    range_x = maxx - minx
                    range_y = maxy - miny
                    step = min(range_x, range_y) / density  # Adjust step size based on density

                    # Generate lines and check if they intersect the cell polygon
                    for offset in np.arange(-range_x, range_x + step, step):
                        line = LineString([(minx + offset, miny), (maxx + offset, maxy)])
                        if line.intersects(cell_polygon):
                            intersected_line = line.intersection(cell_polygon)
                            x, y = intersected_line.xy
                            ax.plot(x, y, color=color, linestyle="-", linewidth=linewidth)

                draw_diagonal_shadows_in_cell(self.slot_polygons[ind[0], ind[1]],
                                              axe,
                                              density=5,
                                              linewidth=1)

            axe.plot(*self.slot_polygons[ind[0], ind[1]].exterior.xy, color=self.color["line_color"],
                     linewidth=self.LINE_WIDTH)

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
        # arrow = FancyArrowPatch(start_pos, start_pos + direction, arrowstyle='-|>', mutation_scale=20,
        #                         color='k', linestyle='dashed', linewidth=2)
        # axe.add_patch(arrow)

        # axe.arrow(*start_pos,
        #           *direction,
        #           width=0.5,
        #           head_width=2,
        #           linestyle='dashed',
        #           head_length=3,
        #           fc='k',
        #           ec='k',
        #           zorder=100,
        #           overhang=0.2)

        axe.arrow(*start_pos,
                  *direction,
                  width=1,
                  head_width=4,
                  linestyle='dashed',
                  head_length=3,
                  fc='k',
                  ec='k',
                  zorder=100,
                  overhang=0.2)

    def plot_bg(self):
        self._plot_rack(axe=self.plot.axes[-1])
        return self.plot.get_img()

    def plot_action_values(self, state: RackState,
                           goal: RackState = None,
                           action_values: np.ndarray = None,
                           columns: int = 12,
                           scale: int or float = 1) -> Plot:
        if goal is None:
            goal = self._goal_pattern
        state = RackState(state)
        goal = RackState(goal)
        rack_size = state.shape
        feasible_action_set = state.feasible_action_set
        action_values_indices = [(*RackArrangementEnv.expr_action(feasible_action_set[i],
                                                                  rack_size,
                                                                  state), action_values[feasible_action_set[i]]) for
                                 i in range(len(feasible_action_set))]
        all_values = [v for _, _, v in action_values_indices if v != 0]
        min_val = min(all_values)
        max_val = max(all_values)
        groups = defaultdict(list)
        for coord in action_values_indices:
            groups[coord[0]].append(coord)

        # Define the font properties for the text
        color = ['green', 'red', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'white', 'purple', 'pink', 'brown',
                 'orange']
        matplotlib.rcParams['font.family'] = 'Times New Roman'
        matplotlib.rcParams['font.size'] = 9  # You can adjust the size as needed
        # font_properties = {'family': 'Arial', 'fontsize': 12, 'weight': 'bold'}
        font_properties = {'family': 'Arial', 'fontsize': 12, }

        # Determine the number of subplots needed
        n_subplots = len(groups)
        n_cols = min(n_subplots, columns)
        n_rows = max(1, int(np.ceil(n_subplots / n_cols)))

        # Create a figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5 * scale, n_rows * 3 * scale))
        if n_subplots == 1:
            axes = np.array([axes])
        # Flatten the axes array for easy indexing
        axes_flat = axes.flatten()
        for idx, (pick_coord, group) in enumerate(groups.items()):
            ax = axes_flat[idx]
            # Create a grid initialized to 0
            grid_size_x, grid_size_y = rack_size
            grid = np.zeros((grid_size_x, grid_size_y)) - np.inf
            # Populate the grid with counts
            for _, (x, y), v in group:
                grid[x, y] = v  # Increment the count for the grid cell at (x, y)

            # Create the heatmap
            im = ax.imshow(grid, cmap='viridis', origin='lower', vmin=min_val, vmax=max_val)
            # ax.colorbar(label='Intensity')
            ax.invert_yaxis()
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            ax.set_xticks(range(grid_size_y))
            ax.set_yticks(range(grid_size_x))
            ax.set_xlabel('Y Coordinate')
            ax.set_ylabel('X Coordinate')
            ax.set_title(f'Pick Coord: {pick_coord}')

            # Optionally, add the counts on the squares
            for (j, i), value in np.ndenumerate(grid):
                if value > -20:  # Only add text when there's a count
                    ax.text(i, j, f'{value:.1f}', ha='center', va='center', color='white',
                            fontdict=font_properties)

            # draw pick
            circle_radius = 0.3
            circle = patches.Circle((pick_coord[1], pick_coord[0]), circle_radius,
                                    edgecolor='none',
                                    facecolor=color[state[pick_coord[0], pick_coord[1]]])  # Choose your color

            ax.add_patch(circle)
            # draw the goal
            for i in goal.num_classes:
                axis = np.where(goal.state == i)
                for _ in range(len(axis[0])):
                    rect_x, rect_y = 0.9, 0.9
                    rect = patches.Rectangle((axis[1][_] - rect_x / 2, axis[0][_] - rect_y / 2), rect_x, rect_y,
                                             linewidth=4,
                                             edgecolor=color[i],
                                             facecolor='none', transform=ax.transData)
                    ax.add_patch(rect)

        # Hide any unused subplots
        for ax in axes_flat[n_subplots:]:
            ax.axis('off')
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.)
        return Plot(fig=fig, w=5, h=4, dpi=200)

    def plot_state(self, state, img_scale=1, toggle_bg=False, toggle_fill=False):
        w, h = state.shape
        h = min(h, img_scale)
        w = h * w / state.shape[1]
        fig, ax = plt.subplots(1, 1, figsize=(h, w))
        # do not show the axis in the figure
        ax.axis('off')
        # reshape the axs into min(state_num or row)
        if toggle_bg:
            self._plot_rack(axe=ax)
        else:
            self._plot_rack(axe=ax, goal_pattern=np.zeros_like(np.array(state.T)))
        state = self._aligned_state(state)
        self._plot_state(state, axe=ax, toggle_fill=toggle_fill)
        return Plot(fig=fig, w=5, h=4, dpi=300)

    def plot_rack(self, state, img_scale=1):
        w, h = state.shape
        h = min(h, img_scale)
        w = h * w / state.shape[1]
        fig, ax = plt.subplots(1, 1, figsize=(h, w))
        # do not show the axis in the figure
        ax.axis('off')
        # reshape the axs into min(state_num or row)
        state = self._aligned_state(state)
        self._plot_rack(axe=ax, goal_pattern=state)
        return Plot(fig=fig, w=5, h=4, dpi=300)

    def plot_states(self, rack_states,
                    row=6,
                    toggle_arrows=True,
                    toggle_fill=False,
                    reward_list=None,
                    fillable_mtx=None,
                    movable_mtx=None,
                    img_scale=1.5) -> Plot:
        # length of the eack
        if isinstance(rack_states, dict):
            rack_state_titles, rack_states = list(rack_states.keys()), list(rack_states.values())
        else:
            rack_state_titles = None
        state_num = len(rack_states)
        # set layout
        w, h = self.state_shape
        h = min(h, img_scale)
        w = h * w / self.state_shape[1]
        fig, axes = plt.subplots(int(np.ceil(state_num / (row))), min(state_num, row),
                                 figsize=(min(state_num, row) * w, h * int(np.ceil(state_num / (row)))))
        # print((min(state_num, row) * w, h * int(np.ceil(state_num / row))))
        if state_num == 1:
            axes = np.array([axes])
        # do not show the axis in the figure
        [i.axis('off') for i in axes.ravel()]
        # reshape the axs into min(state_num or row)
        axes = axes.reshape(-1, min(state_num, row))
        # plot
        for state_ind, state in enumerate(rack_states):
            state = self._aligned_state(state)
            axe_tmp = axes[state_ind // row, state_ind % row]
            # plot each rack state
            self._plot_rack(axe=axe_tmp, )
            # if not toggle_fill:
            #     self._plot_rack(axe=axe_tmp, )
            # else:
            #     self._plot_rack(axe=axe_tmp, goal_pattern=np.zeros_like(np.array(state)))

            # axe_tmp.imshow()
            # plot tube
            self._plot_state(state, axe=axe_tmp, toggle_fill=toggle_fill)
            # plot arrow
            if toggle_arrows:
                state_ind_pr = state_ind - 1
                if state_ind_pr >= 0:
                    state_pr = self._aligned_state(rack_states[state_ind_pr])
                    self._plot_arrow_between_2states(state_pr=state_pr, state=state, axe=axe_tmp)
            if fillable_mtx is not None or movable_mtx is not None:
                if not isinstance(state, RackState):
                    rs = RackState(state)
                else:
                    rs = state
                # fillable_slots = rs.fillable_slots
                # movable_slots = rs.movable_slots
                if movable_mtx is not None:
                    self._plot_cons(self._aligned_state(movable_mtx[state_ind]), axe=axe_tmp, color="red")
                if fillable_mtx is not None:
                    self._plot_cons(self._aligned_state(fillable_mtx[state_ind]), axe=axe_tmp, color="green")
            if rack_state_titles is not None:
                axe_tmp.set_title(f"{rack_state_titles[state_ind]}")
            if reward_list is not None:
                if state_ind > 0:
                    axe_tmp.set_title(f"reward is [{reward_list[state_ind - 1]}]")

        return Plot(fig=fig, w=5, h=4, dpi=200)

    def plot_states_animation(self,
                              rack_states,
                              toggle_fill=False,
                              speed=1,
                              img_scale=1.5) -> animation:
        assert len(rack_states) > 1, 'animation can only generated when rack states larger than 1'
        # length of the eack
        if isinstance(rack_states, dict):
            rack_state_titles, rack_states = list(rack_states.keys()), list(rack_states.values())
        else:
            rack_state_titles = None
        state_num = len(rack_states)
        # set layout
        w, h = self.state_shape
        h = min(h, img_scale)
        w = h * w / self.state_shape[1]
        fig, ax = plt.subplots(figsize=(w, h))
        # do not show the axis in the figure
        ax.axis('off')
        # plot
        self._plot_rack(axe=ax, )
        rack_states_aligned = [self._aligned_state(s) for s in rack_states]
        state = rack_states_aligned[0]
        circle_array = self._plot_state_circle(state, ax)
        start_positions = []
        goal_positions = []
        circles = []
        for state_ind, state_next in enumerate(rack_states_aligned[1:]):
            movemap = state_next - state
            if isinstance(movemap, RackState):
                movemap = movemap._state
            move_to_ind = np.concatenate(np.where(movemap > 0))
            move_from_ind = np.concatenate(np.where(movemap < 0))
            if len(move_to_ind) < 1 or len(move_from_ind) < 1:
                raise Exception("Infeasible actions")
            goal_pos = self.slot_centers[move_to_ind[0], move_to_ind[1]]
            start_pos = self.slot_centers[move_from_ind[0], move_from_ind[1]]

            start_positions.append(start_pos)
            goal_positions.append(goal_pos)
            circles.append(circle_array[move_from_ind[0], move_from_ind[1]])
            circle_array[move_to_ind[0], move_to_ind[1]] = circle_array[move_from_ind[0], move_from_ind[1]]
            circle_array[move_from_ind[0], move_from_ind[1]] = None
            state = state_next

        # Function to create a custom animation for multiple circles
        # def create_animation(start_positions, goal_positions, circles):
        #     # Ensure lists of start_positions, goal_positions, and circles are the same length
        #     if not (len(start_positions) == len(goal_positions) == len(circles)):
        #         raise ValueError("start_positions, goal_positions, and circles must have the same length")
        #
        #     # Update function for animation
        #     def animate(i):
        #         patches = []
        #         for idx, circle in enumerate(circles):
        #             new_x = start_positions[idx][0] + (goal_positions[idx][0] - start_positions[idx][0]) * i / 100
        #             new_y = start_positions[idx][1] + (goal_positions[idx][1] - start_positions[idx][1]) * i / 100
        #             circle.set_center((new_x, new_y))
        #             patches.append(circle)
        #         return patches
        #
        #     return animation.FuncAnimation(fig, animate, frames=100, interval=50, blit=True)

        # Function to create a custom animation for multiple circles
        def distance(point1, point2):
            return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

        # Function to create a custom animation for multiple circles
        def create_animation(start_positions, goal_positions, circles, speed=0.05):
            if not (len(start_positions) == len(goal_positions) == len(circles)):
                raise ValueError("start_positions, goal_positions, and circles must have the same length")

            distances = [distance(start, goal) for start, goal in zip(start_positions, goal_positions)]
            frames_per_circle = [int(d / speed) for d in distances]
            total_frames = sum(frames_per_circle)
            arrows = []
            for i in range(len(start_positions)):
                start_pos = start_positions[i]
                goal_pos = goal_positions[i]
                direction = np.array(goal_pos) - np.array(start_pos)
                direction = direction - 3 * direction / np.linalg.norm(direction)
                tmp = ax.arrow(*start_pos,
                               *direction,
                               width=1,
                               head_width=4,
                               linestyle='dashed',
                               head_length=3,
                               fc='k',
                               ec='k',
                               zorder=100,
                               overhang=0.2)
                tmp.set_visible(False)
                arrows.append(tmp)

            def animate(frame):
                patches = []
                accumulated_frames = 0
                for idx, circle in enumerate(circles):
                    start_frame = accumulated_frames
                    end_frame = accumulated_frames + frames_per_circle[idx]
                    accumulated_frames = end_frame
                    arrow = arrows[idx]
                    if start_frame <= frame < end_frame:
                        progress = (frame - start_frame) / frames_per_circle[idx]
                        new_x = start_positions[idx][0] + (goal_positions[idx][0] - start_positions[idx][0]) * progress
                        new_y = start_positions[idx][1] + (goal_positions[idx][1] - start_positions[idx][1]) * progress
                        circle.set_center((new_x, new_y))
                        arrow.set_visible(True)
                    elif frame >= end_frame:
                        circle.set_center(goal_positions[idx])
                        arrow.set_visible(False)
                    patches.append(circle)
                    patches.append(arrow)
                return patches

            return animation.FuncAnimation(fig, animate, frames=total_frames, interval=50, blit=True)

        return create_animation(start_positions, goal_positions, circles, speed=speed)

    def plot_states_nobg(self, rack_states, row=6, toggle_arrows=True, fillable_mtx=None, movable_mtx=None,
                         img_scale=1) -> Plot:
        # length of the eack
        if isinstance(rack_states, dict):
            rack_state_titles, rack_states = list(rack_states.keys()), list(rack_states.values())
        else:
            rack_state_titles = None
        state_num = len(rack_states)
        # set layout
        w, h = self.state_shape
        h = min(h, img_scale)
        w = h * w / self.state_shape[1]
        fig, axes = plt.subplots(int(np.ceil(state_num / (row))), min(state_num, row),
                                 figsize=(min(state_num, row) * w, h * int(np.ceil(state_num / (row)))))
        # print((min(state_num, row) * w, h * int(np.ceil(state_num / row))))
        if state_num == 1:
            axes = np.array([axes])
        # do not show the axis in the figure
        [i.axis('off') for i in axes.ravel()]
        # reshape the axs into min(state_num or row)
        axes = axes.reshape(-1, min(state_num, row))
        # plot
        for state_ind, state in enumerate(rack_states):
            state = self._aligned_state(state)
            axe_tmp = axes[state_ind // row, state_ind % row]
            # plot each rack state
            self._plot_rack(axe=axe_tmp, goal_pattern=np.zeros_like(state), )
            # axe_tmp.imshow()
            # plot tube
            self._plot_state(state, axe=axe_tmp, toggle_fill=True)
            # plot arrow
            if toggle_arrows:
                state_ind_pr = state_ind - 1
                if state_ind_pr >= 0:
                    state_pr = self._aligned_state(rack_states[state_ind_pr])
                    self._plot_arrow_between_2states(state_pr=state_pr, state=state, axe=axe_tmp)
            if fillable_mtx is not None or movable_mtx is not None:
                if not isinstance(state, RackState):
                    rs = RackState(state)
                else:
                    rs = state
                # fillable_slots = rs.fillable_slots
                # movable_slots = rs.movable_slots
                if movable_mtx is not None:
                    self._plot_cons(self._aligned_state(movable_mtx[state_ind]), axe=axe_tmp, color="red")
                if fillable_mtx is not None:
                    self._plot_cons(self._aligned_state(fillable_mtx[state_ind]), axe=axe_tmp, color="green")
            if rack_state_titles is not None:
                axe_tmp.set_title(f"{rack_state_titles[state_ind]}")
        return Plot(fig=fig, w=5, h=4, dpi=200)


class RackArrangementEnv(Gym_Proto):
    DEAD_PATTERN = np.array([[0, 1, 1, 0],
                             [1, 1, 1, 1],
                             [1, 1, 1, 1],
                             [0, 1, 1, 0]])

    @classmethod
    def to_action(cls, *args, **kwargs):
        return to_action(*args, **kwargs)

    @classmethod
    def from_action(cls, *args, **kwargs):
        return from_action(*args, **kwargs)

    @classmethod
    def expr_action(cls, action: int, rack_size: tuple, state: np.ndarray or RackState) -> (tuple, tuple):
        p1, p2 = from_action(rack_size, action)
        if state[p1] == 0 and state[p2] > 0:
            pick_id = p2
            place_id = p1
        elif state[p2] == 0 and state[p1] > 0:
            pick_id = p1
            place_id = p2
        else:
            raise Exception(f"Unexpected action {action, p1, p2}")
        return pick_id, place_id

    @staticmethod
    def F(state):
        if not isinstance(state, RackState):
            rs = RackState(state)
        else:
            rs = state
        return len(rs.feasible_action_set)

    @staticmethod
    def E(state, goal_pattern):
        """
        Number of the state inside the pattern
        :param state:
        :param goal_pattern:
        :return:
        """
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
    def oE(state, goal_pattern):
        """
        Number of the state outsider the pattern
        :param state:
        :param goal_pattern:
        :return:
        """
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
            entropy[i] = len(state[state == _]) - len(np.where(state[goal_pattern == _] == _)[0])
        return entropy

    @staticmethod
    def action_between_states(s_current, s_next, toggle_debug=False, toggle_strict_mode=True):
        # TODO TEST
        if isinstance(s_current, RackState):
            s_current = s_current.state
        if isinstance(s_next, RackState):
            s_next = s_next.state
        assert s_current.shape == s_next.shape, "ERROR"
        move = s_next - s_current
        move_to_idx = np.hstack(np.where(move > 0))
        move_from_idx = np.hstack(np.where(move < 0))
        if toggle_debug:
            print(s_current)
            print(s_next)
        if len(move_to_idx) > 2 or len(move_from_idx) > 2:
            raise Exception("moving error")
        rack_size = s_current.shape
        action_ids = to_action(rack_size, move_to_idx, move_from_idx)

        if action_ids not in RackState(s_current).feasible_action_set:
            if toggle_strict_mode:
                raise Exception("ERROR ACTIONS")
            else:
                return None
        return action_ids.item()

    @staticmethod
    def action_between_states_condition_set(s_current,
                                            s_next,
                                            condition_set=None,
                                            toggle_debug=False,
                                            toggle_strict_mode=True):
        # TODO TEST
        if isinstance(s_current, RackState):
            s_current = s_current.state
        if isinstance(s_next, RackState):
            s_next = s_next.state
        assert s_current.shape == s_next.shape, "ERROR"
        move = s_next - s_current
        move_to_idx = np.hstack(np.where(move > 0))
        move_from_idx = np.hstack(np.where(move < 0))
        if toggle_debug:
            print(s_current)
            print(s_next)
        if len(move_to_idx) > 2 or len(move_from_idx) > 2:
            raise Exception("moving error")
        rack_size = s_current.shape
        action_ids = to_action(rack_size, move_to_idx, move_from_idx)
        if condition_set is not None:
            feasible_actions = RackState(s_current).get_feasible_action_condition_set(condition_set)
        else:
            feasible_actions = RackState(s_current).feasible_action_set
        if action_ids not in feasible_actions:
            if toggle_strict_mode:
                raise Exception("ERROR ACTIONS")
            else:
                return None
        return action_ids.item()

    @staticmethod
    def action_between_states_constraint_free(s_current, s_next, toggle_debug=False):
        # TODO TEST
        if isinstance(s_current, RackState):
            s_current = s_current.state
        if isinstance(s_next, RackState):
            s_next = s_next.state
        assert s_current.shape == s_next.shape, "ERROR"
        move = s_next - s_current
        move_to_idx = np.hstack(np.where(move > 0))
        move_from_idx = np.hstack(np.where(move < 0))
        if toggle_debug:
            print(s_current)
            print(s_next)
        if len(move_to_idx) != 2 or len(move_from_idx) != 2:
            # print("TEST")
            return None
        rack_size = s_current.shape
        action_ids = to_action(rack_size, move_to_idx, move_from_idx)
        return action_ids.item()

    @staticmethod
    def is_safe(current_state, goal_pattern, fillable_mask, movable_mask, category_num=1):
        """
        Check if all categories in the goal pattern are fillable based on the mask.

        Parameters:
        - goal_pattern: numpy array of shape (m, n) with entries in {0, 1, 2, 3}. 0 is empty, and 1, 2, 3 are categories.
        - mask: numpy array of shape (m, n) with boolean values. True indicates the position is fillable.

        Returns:
        - True if all categories are fillable, else False
        """
        # Create a mask where entries do not match the goal pattern
        mismatch_mask_goal = (goal_pattern.state != current_state.state) & (goal_pattern.state != 0)
        mismatch_mask_state = (goal_pattern.state != current_state.state) & (current_state.state != 0)
        # Use the mask to extract unsatisfied entries from goal_pattern
        unsatisfied_goal = np.where(mismatch_mask_goal, goal_pattern, 0)
        unsatisfied_state = np.where(mismatch_mask_state, current_state, 0)
        infeasible_num_goal_list = []
        infeasible_num_state_list = []
        for category in range(1, category_num + 1):
            if np.any(unsatisfied_goal == category):
                # Check if any category position is not fillable
                infeasible_num_state_list.append(np.count_nonzero((movable_mask[unsatisfied_state == category]) == 0))
                infeasible_num_goal_list.append(np.count_nonzero((fillable_mask[unsatisfied_goal == category]) == 0))
            else:
                infeasible_num_state_list.append(0)
                infeasible_num_goal_list.append(0)
        return np.array(infeasible_num_state_list, dtype=int), np.array(infeasible_num_goal_list, dtype=int)

    @staticmethod
    def is_safe2(current_state, goal_pattern, category_num=1, toggle_fillable_map=False):
        """
        Check if all categories in the goal pattern are fillable based on the mask.

        Parameters:
        - goal_pattern: numpy array of shape (m, n) with entries in {0, 1, 2, 3}. 0 is empty, and 1, 2, 3 are categories.
        - mask: numpy array of shape (m, n) with boolean values. True indicates the position is fillable.

        Returns:
        - True if all categories are fillable, else False
        """
        feasible_actions = current_state.feasible_action_set_detail
        fillable_map, movable_map = current_state.fillable_movable_region
        movable_map = np.zeros_like(current_state, dtype=bool)
        movable_map[
            feasible_actions[:, 0], feasible_actions[:, 1]] = 1  # ke neng neng na, dan shi na le zhi hou yi bu chu qu?

        # Create a mask where entries do not match the goal pattern
        avaliable_goal_pattern = np.where((current_state.state == 0) & (goal_pattern.state > 0), goal_pattern, 0)
        mismatch_mask_state = (goal_pattern.state != current_state.state) & (current_state.state != 0)
        # Use the mask to extract unsatisfied entries from goal_pattern
        unsatisfied_state = np.where(mismatch_mask_state, current_state, 0)
        infeasible_num_goal_list = []
        infeasible_num_state_list = []
        num_feasible_goal_slot = np.zeros(category_num, dtype=int)
        num_unsatisfied_tube = np.zeros(category_num, dtype=int)
        num_tube_no_belong_to_goal = np.zeros(category_num, dtype=int)
        num_tube_no_belong_to_goal_movable = np.zeros(category_num, dtype=int)
        for category in range(1, category_num + 1):
            category_id = category - 1
            tube_no_belong_to_goal_mask = (current_state.state > 0) & (goal_pattern.state != current_state.state) & (
                    goal_pattern == category)
            tube_no_belong_to_goal_mask_indices = np.argwhere(tube_no_belong_to_goal_mask)
            num_tube_no_belong_to_goal[category_id] = len(tube_no_belong_to_goal_mask_indices)
            num_tube_no_belong_to_goal_movable[category_id] = np.count_nonzero(movable_map[tube_no_belong_to_goal_mask])
            # TODO 能movable不代表能移动出去
            if np.any(avaliable_goal_pattern == category):
                # complete goal pattern
                fillable_map_tmp = fillable_map.copy()
                for _ in np.argwhere(unsatisfied_state == category):
                    ids = feasible_actions[np.all(feasible_actions[:, :2] == _, axis=1)][:, 2:4]
                    fillable_map_tmp[ids[:, 0], ids[:, 1]] = 1
                # Check if any category position is not fillable
                # infeasible_num_state_list.append(np.count_nonzero((movable_map[unsatisfied_state == category]) == 0))
                num_feasible_goal_slot[category_id] = np.count_nonzero(
                    (fillable_map_tmp[avaliable_goal_pattern == category]))
            else:
                num_feasible_goal_slot[category_id] = 0
            num_unsatisfied_tube[category_id] = np.count_nonzero(unsatisfied_state == category)
        if toggle_fillable_map:
            return num_unsatisfied_tube, num_feasible_goal_slot, fillable_map
        else:
            return num_unsatisfied_tube, num_feasible_goal_slot, num_tube_no_belong_to_goal, num_tube_no_belong_to_goal_movable, movable_map

    def __init__(self,
                 rack_size=(5, 10),
                 num_classes=5,
                 observation_space_dim=(5, 10),
                 num_history=4,
                 action_space_dim=10,
                 is_curriculum_lr=True,
                 is_goalpattern_fixed=True,
                 scheduler: 'GoalRackStateScheduler3' or 'GoalRackStateScheduler2' = None,
                 seed=777, ):
        super(RackArrangementEnv, self).__init__()
        # init
        self.rack_size = rack_size
        self.num_classes = num_classes
        self.observation_space_dim = (num_history + 1, *observation_space_dim)
        self.action_space_dim = action_space_dim
        self.is_curriculum_lr = is_curriculum_lr
        self.is_goalpattern_fixed = is_goalpattern_fixed

        self.abs_goal_slice = slice(0, rack_size[0])
        self.abs_state_slice = slice(rack_size[0], rack_size[0] * 2)

        # setup state, goal_pattern
        self.state = None
        self.goal_pattern = None

        # setup action space
        self.action_sapce = spaces.Discrete(self.action_space_dim)

        # set the seed for the env, ! action space and observation space should set seed as well
        self.action_sapce.seed(2 ** 31 - 3 * seed)
        self.seed(seed)

        self.rack_state_history = []
        self.reward_history = []

        self._is_done = False

        # setup scheduler
        if scheduler is None:
            scheduler = GoalRackStateScheduler(num_classes=num_classes, rack_size=rack_size,
                                               np_random=self._np_random)
        elif scheduler == 'RackStateScheduler':
            scheduler = RackStateScheduler(num_classes=num_classes, rack_size=rack_size,
                                           np_random=self._np_random, )
        elif scheduler == 'RackStateScheduler2':
            scheduler = RackStateScheduler2(num_classes=num_classes,
                                            rack_size=rack_size,
                                            np_random=self._np_random, is_goalpattern_fixed=is_goalpattern_fixed)
        elif scheduler == 'RackStateScheduler2_5':
            scheduler = RackStateScheduler2_5(num_classes=num_classes,
                                              rack_size=rack_size,
                                              np_random=self._np_random, is_goalpattern_fixed=is_goalpattern_fixed)
        elif scheduler == 'GoalRackStateScheduler2':
            scheduler = GoalRackStateScheduler2(num_classes=num_classes, rack_size=rack_size,
                                                np_random=self._np_random)
        elif scheduler == 'GoalRackStateScheduler3':
            scheduler = GoalRackStateScheduler3(num_classes=num_classes, rack_size=rack_size,
                                                np_random=self._np_random)
        self.scheduler: RackStateScheduler = scheduler

        # if is_goalpattern_fixed:
        #     self.set_goal_pattern(self.scheduler.gen_goal(not self.is_curriculum_lr))
        if not is_goalpattern_fixed:
            self.reset()

    @property
    def is_done(self) -> bool:
        return self._is_done

    def is_state_valid(self, state=None):
        if state is None:
            state = self.state
        state_np = np.array(state)
        if state_np.shape[0] < self.DEAD_PATTERN.shape[0] or state_np.shape[1] < self.DEAD_PATTERN.shape[1]:
            return True
        return not check_pattern_in_matrix(big_matrix=state_np, pattern=self.DEAD_PATTERN)

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

    def reset_goal(self, goal_pattern: RackState or np.ndarray):
        if isinstance(goal_pattern, RackState):
            goal_pattern = goal_pattern
        elif isinstance(goal_pattern, np.ndarray):
            goal_pattern = RackState(goal_pattern)
        else:
            raise Exception("No support goal state")
        if hasattr(self.scheduler, 'set_goal_pattern'):
            self.scheduler.set_goal_pattern(goal_pattern)
        if not self.is_curriculum_lr:
            initstate = get_random_states(self.rack_size,
                                          goal_pattern.state,
                                          obj_num=goal_pattern.nonzero_count,
                                          min_obj_num=goal_pattern.nonzero_count,
                                          np_random=self._np_random)
        else:
            for i in range(1000):
                initstate = self.scheduler.gen_state(goal_pattern)
                if self.is_state_valid(initstate):
                    break
            else:
                raise Exception("Cannot generate vaild state")
        return self.reset_state_goal(init_state=initstate, goal_pattern=goal_pattern)

    def set_goal_pattern(self, goal_pattern):
        if isinstance(goal_pattern, np.ndarray):
            self.goal_pattern = RackState(goal_pattern.copy())
        elif isinstance(goal_pattern, RackState):
            self.goal_pattern = goal_pattern.copy()
        else:
            raise Exception("Undefined input type")
        # self.scheduler.max_state_level_per_class = min(np.bincount(self.goal_pattern.state.ravel())[1:])
        if hasattr(self.scheduler, 'set_goal_pattern'):
            self.scheduler.set_goal_pattern(goal_pattern)
        return self.goal_pattern

    def reset(self):
        if not self.is_goalpattern_fixed:
            # goal_pattern = self._np_random.randint(1, self.num_classes + 1, size=self.rack_size)
            goal_pattern = self.scheduler.gen_goal(not self.is_curriculum_lr)
        else:
            if isinstance(self.goal_pattern, RackState):
                goal_pattern = self.goal_pattern
            elif isinstance(self.goal_pattern, np.ndarray):
                goal_pattern = RackState(self.goal_pattern)
            else:
                raise Exception("No support goal state")
        if not self.is_curriculum_lr:
            # print(goal_pattern,goal_pattern.nonzero_count)
            initstate = get_random_states(self.rack_size,
                                          goal_pattern.state,
                                          obj_num=goal_pattern.nonzero_count,
                                          min_obj_num=goal_pattern.nonzero_count,
                                          np_random=self._np_random)
        else:
            for i in range(1000):
                initstate = self.scheduler.gen_state(goal_pattern)
                if self.is_state_valid(initstate):
                    break
            else:
                raise Exception("Cannot generate vaild state")
        return self.reset_state_goal(init_state=initstate, goal_pattern=goal_pattern)

    def _expr_action(self, action: int):
        rack_size = self.rack_size
        p1, p2 = from_action(rack_size, action)
        if self.state[p1] == 0 and self.state[p2] > 0:
            pick_id = p2
            place_id = p1
        elif self.state[p2] == 0 and self.state[p1] > 0:
            pick_id = p1
            place_id = p2
        else:
            raise Exception(f"Unexpected action {action, p1, p2}")
        return pick_id, place_id

    def _get_reward(self, is_finished, state_current, state_next, goal_pattern, toggle_debug=False,
                    toggle_reward_info=False):
        # if self.is_goalpattern_fixed:
        reward = self._get_reward_fixed(is_finished,
                                        state_current,
                                        state_next,
                                        goal_pattern,
                                        toggle_debug=toggle_debug, toggle_reward_info=toggle_reward_info)
        # else:
        # reward = self._get_reward_nofixed(is_finished, state_current, state_next, goal_pattern,
        #                                   toggle_debug=toggle_debug)
        # scale_factor = np.prod(self.rack_size)
        # reward = reward / scale_factor
        return reward

    def _get_reward_fixed(self,
                          is_finished: bool,
                          state_current: np.ndarray,
                          state_next: np.ndarray,
                          goal_pattern: np.ndarray, toggle_debug=False,
                          toggle_reward_info=False):
        state_current, state_next, goal_pattern = np.array(state_current), \
            np.array(state_next), np.array(goal_pattern)
        if is_finished:
            # NOTE: This value should be change according to the number of test tubes
            if isdone(state_next, goal_pattern):
                return 20
            else:
                return -20
        reward = 0  # every move will cause 1 penalty
        reward_info_str = ''
        move_map = state_next - state_current
        move_to_idx = np.where(move_map > 0)
        move_from_idx = np.where(move_map < 0)
        is_move_to_pattern = goal_pattern[move_to_idx] == state_next[move_to_idx]
        is_move_out_pattern = goal_pattern[move_from_idx] == state_current[move_from_idx]

        # state
        # TODO: There is a bug exist in determining the feasible slot
        # num_unsatisfied_tube_c, num_feasible_goal_slot_c = self.is_safe2(RackState(state_current),
        #                                                                  RackState(goal_pattern),
        #                                                                  category_num=self.num_classes)
        num_unsatisfied_tube_n, num_feasible_goal_slot_n, num_tube_no_belong_to_goal, num_tube_no_belong_to_goal_movable, \
            movable_map = self.is_safe2(RackState(state_next),
                                        RackState(goal_pattern),
                                        category_num=self.num_classes)

        # no avaliable goal slot for the tube
        if is_move_to_pattern and not is_move_out_pattern:
            obj_type = state_current[move_from_idx]
            obj_index = obj_type - 1
            is_goal_infeasible = num_feasible_goal_slot_n[obj_index] == 0
            is_other_tube_outside_pattern = num_unsatisfied_tube_n[obj_index] > 0
            is_different_tube_inside_goal = num_tube_no_belong_to_goal[obj_index] > 0
            is_different_tube_movable_inside_goal = num_tube_no_belong_to_goal_movable[obj_index] > 0
            category_goal_pattern_mask = goal_pattern == obj_type
            if toggle_debug:
                print(num_tube_no_belong_to_goal, num_tube_no_belong_to_goal_movable,
                      num_tube_no_belong_to_goal[obj_index], num_tube_no_belong_to_goal_movable[obj_index])
                # is other kind of tube movable inside the pattern
                # YES -> no minus
                # FALSE -> -3 fen
                # if a move surrounding other object
                rp = RackStatePlot(goal_pattern=self.goal_pattern)
            # p1
            # p1: dead goal pattern
            # p2: goal depandency on others
            # p3: successfully fill all test tubes into a pattern (GOOD)
            # p4 no affect after puting test tubes into the goal pattern (GOOD)
            # p5 after put test tubes into goal pattern, are movable test tubes changed, if no change, get reward
            # p6_-3: if after move, objects are block by the moved object: -3
            if is_goal_infeasible and not is_different_tube_inside_goal and is_other_tube_outside_pattern:
                if np.sum(get_fillable(extract_submatrix_mask(state_next, category_goal_pattern_mask))) <= 0:
                    reward = -3
                    reward_info_str += ' p1'
                    if toggle_debug:
                        img = rp.plot_states(rack_states=[state_current, state_next], row=8,
                                             img_scale=1.8, reward_list=[f'{reward}_p1']).get_img()
                        cv2.imshow(f"p", img)
                        cv2.waitKey(0)
                else:
                    reward = 1
            else:
                reward = 1
        elif is_move_out_pattern and not is_move_to_pattern:
            reward = -2
            reward_info_str += ' outpattern'
        else:
            reward = -1
            reward_info_str += ' randomwalk'

        # move to the other goal
        if goal_pattern[move_to_idx] > 0 and (goal_pattern[move_to_idx] != state_next[move_to_idx]):
            if toggle_debug:
                print(">>> move to the other goal: -2")
            reward -= 1
            reward_info_str += ' othergoal'
        # move out of other goal
        if goal_pattern[move_from_idx] > 0 and goal_pattern[move_from_idx] != state_current[move_from_idx] \
                and not (goal_pattern[move_to_idx] > 0 and (goal_pattern[move_to_idx] != state_next[move_to_idx])):
            if toggle_debug:
                print(">>> do not block the way: +1")
            reward += 1
            reward_info_str += ' out'
        if toggle_reward_info:
            return reward, reward_info_str
        else:
            return reward
        # return reward + f

    # def _get_reward_nofixed(self, is_finished, state_current, state_next, goal_pattern):
    #     if is_finished:
    #         # return 5 * len(state_current[state_current > 0])
    #         return np.prod(self.rack_size)
    #     enp = np.sum(self.E(state_next, goal_pattern) - self.E(state_current, goal_pattern))
    #     f = np.clip(np.sum(self.F(state_next) - self.F(state_current)), -20, 20)
    #     # print(self.F(state_next), self.F(state_current))
    #     # print("entropy: ", enp, "feasible aciton:", f)
    #     # return enp + .1 * f - 1
    #     return - 1 - np.sum(self.oE(state_next, goal_pattern))

    # def _get_reward_nofixed(self, is_finished, state_current, state_next, goal_pattern):
    #     if is_finished:
    #         return self.scheduler.state_level
    #         # return 0
    #     enp = np.sum(self.E(state_next, goal_pattern) - self.E(state_current, goal_pattern))
    #     return - 1 + enp

    # def _get_reward_nofixed(self, is_finished, state_current, state_next, goal_pattern, toggle_debug=False):
    #     if is_finished:
    #         return 10
    #     reward = 0  # every move will cause 1 penalty
    #     move_map = state_next - state_current
    #     move_to_idx = np.where(move_map > 0)
    #     move_from_idx = np.where(move_map < 0)
    #     is_move_to_pattern = goal_pattern[move_to_idx] == state_next[move_to_idx]
    #     is_move_out_pattern = goal_pattern[move_from_idx] == state_current[move_from_idx]
    #     if is_move_to_pattern and not is_move_out_pattern:
    #         reward = 1
    #     elif is_move_out_pattern and not is_move_to_pattern:
    #         reward = -2
    #     elif goal_pattern[move_to_idx] > 0 and (goal_pattern[move_to_idx] != state_next[move_to_idx]):
    #         reward = -5
    #     else:
    #         reward = -1
    #     # add action
    #
    #     clip_range = np.prod(self.rack_size) / 2
    #     f = np.clip(self.F(state_next) - self.F(state_current), -clip_range, clip_range) / clip_range
    #     return reward + f

    def _get_reward_nofixed(self, is_finished, state_current, state_next, goal_pattern, toggle_debug=False):
        if is_finished:
            # return len(state_current[state_current > 0])
            return 10
        reward = 0  # every move will cause 1 penalty
        move_map = state_next - state_current
        move_to_idx = np.where(move_map > 0)
        move_from_idx = np.where(move_map < 0)
        is_move_to_pattern = goal_pattern[move_to_idx] == state_next[move_to_idx]
        is_move_out_pattern = goal_pattern[move_from_idx] == state_current[move_from_idx]
        # check if the move of the pattern disturb the other pattern
        fb, mb = RackState(state_current).fillable_movable_region
        infeasible_slots_state_current, infeasible_slots_goal_current = self.is_safe(RackState(state_current),
                                                                                     RackState(goal_pattern),
                                                                                     fb,
                                                                                     mb,
                                                                                     category_num=self.num_classes)
        fb, mb = RackState(state_next).fillable_movable_region
        infeasible_slots_state_next, infeasible_slots_goal_next = self.is_safe(RackState(state_next),
                                                                               RackState(goal_pattern),
                                                                               fb,
                                                                               mb,
                                                                               category_num=self.num_classes)
        if toggle_debug:
            print("-" * 20)
            print("infeasible state current", infeasible_slots_state_current)
            print("infeasible state next", infeasible_slots_state_next)
            print('state:', (infeasible_slots_state_next - infeasible_slots_state_current))
            print("infeasible goal current", infeasible_slots_goal_current)
            print("infeasible goal next", infeasible_slots_goal_next)
            print('goal:', (infeasible_slots_goal_next - infeasible_slots_goal_current))

        infeasible_goal = infeasible_slots_goal_next - infeasible_slots_goal_current
        infeasible_goal[infeasible_goal > 0] *= 2
        if np.sum(infeasible_goal) < 0:
            if toggle_debug:
                print(">>> remove the block goal: + 1.5")
            reward += 1.5
        if np.sum(infeasible_goal) > 0:
            if toggle_debug:
                print(">>> increase the block goal: -2")
            reward -= 2
        # ---------
        infeasible_state = infeasible_slots_state_next - infeasible_slots_state_current
        if np.sum(infeasible_state) < 0:
            if toggle_debug:
                print(">>> remove the block state: +1")
            reward += 1
        if np.sum(infeasible_state) > 0:
            if toggle_debug:
                print(">>> increase the block state: -1")
            reward -= 1.5

        # check if the move is inside the pattern
        if is_move_to_pattern and not is_move_out_pattern:
            if toggle_debug:
                print(">>> move to the goal: +1")
            reward += 1
        elif is_move_out_pattern and not is_move_to_pattern:
            if toggle_debug:
                print(">>> move out of the goal: -2")
            reward -= 1.5
        # elif is_move_out_pattern:
        #     if toggle_debug:
        #         print(">>> move out of the goal -_-: -2")
        #     reward -= 1.5
        # elif goal_pattern[move_to_idx] > 0 and (goal_pattern[move_to_idx] != state_next[move_to_idx]):
        #     if toggle_debug:
        #         print(">>> move to the other goal: -2")
        #     reward -= 2
        else:
            reward -= 1
        if toggle_debug:
            print(goal_pattern[move_to_idx], goal_pattern[move_to_idx] > 0,
                  goal_pattern[move_to_idx] != state_next[move_to_idx], goal_pattern[move_to_idx],
                  state_next[move_to_idx])
        # if goal_pattern[move_to_idx] > 0 and (goal_pattern[move_to_idx] != state_next[move_to_idx]):
        #     if toggle_debug:
        #         print(">>> move to the wrong goal: -2")
        #     reward -= 2
        # add action
        # clip_range = np.prod(self.rack_size) ** 2 / 2
        # check if it is move out to not obscure other pattern
        # if goal_pattern[move_from_idx] > 0 and goal_pattern[move_from_idx] != state_current[move_from_idx] \
        #         and not (goal_pattern[move_to_idx] > 0 and (goal_pattern[move_to_idx] != state_next[move_to_idx])):
        #     if toggle_debug:
        #         print(">>> do not block the way: +1")
        #     reward += 1

        # num_a_c = self.F(state_current)
        # change_of_action = (self.F(state_next) - num_a_c) / num_a_c
        # f = np.clip(np.sign(change_of_action) * np.sqrt(abs(change_of_action)), -1, 1)
        if toggle_debug:
            # print(f"Reward is: {reward + f}")
            print(f'>>>>>>>>>>>> Reward is: {reward}')
        return reward
        # return reward + f

    # def _get_reward_nofixed(self, is_finished, state_current, state_next, goal_pattern):
    #     num_obj = len(goal_pattern[goal_pattern > 0])
    #     if is_finished:
    #         # return self.scheduler.state_level
    #         return num_obj
    #     c_acts = self.F(state_current)
    #     n_acts = self.F(state_next)
    #
    #     if num_obj < 30:
    #         move_map = np.array(state_next - state_current)
    #         move_to_idx = np.where(move_map > 0)
    #         move_from_idx = np.where(move_map < 0)
    #         if goal_pattern[move_to_idx].item() > 0 and goal_pattern[move_to_idx] != state_current[move_from_idx]:
    #             ext_reward = -1
    #         else:
    #             ext_reward = 0
    #     else:
    #         ext_reward = 0
    #     enp = np.sum(self.E(state_next, goal_pattern) - self.E(state_current, goal_pattern))
    #     f = np.clip(n_acts - c_acts, 0, 1)
    #     return - 1 + f + enp + ext_reward
    #     # return - 1 + f

    def sample(self, state: RackState = None, no_repeat=True):
        if state is None:
            state = self.state
        feasible_actions = state.feasible_action_set
        if no_repeat:
            repeat_acts = []
            for _ in self.rack_state_history:
                act = self.action_between_states_constraint_free(state, _)
                if act is not None:
                    repeat_acts.append(act)
            # print(feasible_actions, repeat_acts)
            # print(feasible_actions, repeat_acts)
            feasible_actions = np.setdiff1d(feasible_actions, repeat_acts)
            # if len(repeat_acts) > 5:
            #     print(repeat_acts)
            # print(feasible_actions)
            if len(feasible_actions) == 0:
                feasible_actions = state.feasible_action_set
        if len(feasible_actions) == 0:
            # raise Exception("Feasible action is empty")
            print("Feasible action is empty")
            return None
        else:
            return self._np_random.choice(feasible_actions)

    def _step(self, pick_id, place_id, toggle_debug=False) -> (RackState, float, bool, list):
        # get current state
        curr_state = self.state
        goal_pattern = self.goal_pattern
        self.rack_state_history.append(curr_state.copy())

        # update the state
        nxt_state = curr_state.copy().swap(ind_a=pick_id, ind_b=place_id)
        self.state = nxt_state

        # check if the new state is illegal state
        nxt_feasible_actions = self.state.feasible_action_set
        if np.sum(nxt_feasible_actions) == 0:
            reward = -10
            # print("?!")
            self.reward_history.append(reward)
            return nxt_state, reward, True, {
                "reward_history": self.reward_history
            }

        # is finished => ?
        # is_finished = isdone(nxt_state.copy(), goal_pattern)
        is_finished = self.is_finished(nxt_state.copy(), goal_pattern)
        # get reward of the action
        reward = self._get_reward(is_finished,
                                  curr_state.copy(),
                                  nxt_state.copy(),
                                  goal_pattern,
                                  toggle_debug=toggle_debug)
        self.reward_history.append(reward)

        return nxt_state, reward, is_finished, {
            'reward_history': self.reward_history,
            'rack_history': self.rack_state_history
        }

    def step(self, action: int, toggle_debug: bool = False) -> (RackState, float, bool, list):
        pick_id, place_id = self._expr_action(action)
        return self._step(pick_id, place_id, toggle_debug=toggle_debug)

    def is_finished(self, state: RackState = None, goal_state: RackState = None):
        if goal_state is None:
            goal_state = np.asarray(self.goal_pattern)
        else:
            goal_state = np.asarray(goal_state)

        if state is None:
            state = np.asarray(self.state)
        else:
            state = np.asarray(state)

        return isdone(state, goalpattern=goal_state) or not self.is_state_valid(state)

    def to_state(self, state_np: np.ndarray):
        return RackState(state_np)

    def to_state_np(self, state: RackState):
        if isinstance(state, RackState):
            return state.state
        elif isinstance(state, np.ndarray):
            return state
        else:
            raise Exception("No support input state")

    def observation(self):
        obs = np.zeros(self.observation_space_dim)
        assert self.observation_space_dim[0] > 0, 'Observation dim 0 must be greater than 2'
        obs[-1] = self.to_state_np(self.goal_pattern)
        obs[-2] = self.to_state_np(self.state)
        for i in range(1, self.observation_space_dim[0] - 1):
            # i \in [1, self.observation_space_dim[0]-2]
            if i <= len(self.rack_state_history):
                obs[-2 - i] = self.to_state_np(self.rack_state_history[-i])
            else:
                break
        return obs

    def legal_actions(self):
        array = np.zeros(self.action_space_dim)
        array[self.state.feasible_action_set] = 1
        return array

    def evaluate_traj(self, paths, goal_pattern, toggle_debug=False):
        reward_traj = []
        action_traj = []
        acc_reward = 0
        for p in range(0, len(paths) - 1)[::-1]:
            s = paths[p]
            s_n = paths[p + 1]
            finished = self.is_finished(s_n.state if isinstance(s_n, RackState) else s_n,
                                        goal_pattern.state if isinstance(goal_pattern, RackState) else goal_pattern)
            acc_reward += self._get_reward(finished, s, s_n, goal_pattern)
            reward_traj.append(acc_reward)
            action_traj.append(self.action_between_states(s, s_n))
            if toggle_debug:
                print(s)
                print(s_n)
                print(self._expr_action(action_traj[-1]))
        return paths, reward_traj[::-1], action_traj[::-1]

    def synthesize_reflection_data(self, goal, state, reward, next_state, done):
        if not isinstance(goal, RackState):
            goal = RackState(goal)
        if not isinstance(state, RackState):
            state = RackState(state)
        if not isinstance(next_state, RackState):
            next_state = RackState(next_state)
        goal_r = goal.reflections
        state_r = state.reflections
        next_state_r = next_state.reflections
        data_pack = []
        for (g, s, sn) in zip(goal_r, state_r, next_state_r):
            data_pack.append([g.state, s.state, self.action_between_states(s, sn), reward, sn.state, done])
        return data_pack

    def copy(self, toggle_reset=True):
        env = copy.deepcopy(self)
        if toggle_reset:
            env.reset()
        return env

    def set_seed(self, seed=None):
        self.seed(seed)
        self.scheduler._np_random = self._np_random


class RackStateScheduler(object):
    def __init__(self, num_classes, rack_size, np_random=None, state_level=1, goal_level=1, class_level=1,
                 max_state_level_per_class=8):
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

    def gen_goal(self, no_curriculum=False):
        if no_curriculum:
            chosed_ids = self._np_random.choice(self.class_ids, self.num_classes, replace=False)
        else:
            chosed_ids = self._np_random.choice(self.class_ids, min(self.goal_level, self.num_classes), replace=False)
        # print("chosed ids", chosed_ids)
        # goal_pattern = self._np_random.randint(0, self.goal_level + 1, size=self.rack_size)
        p_0 = max(.3 * np.cos(np.pi / 2 * self.state_level / (np.prod(self.rack_size) - 1)), 0)
        # print("p_o is ", p_0)
        goal_pattern = self._np_random.choice(np.arange(0, self.goal_level + 1), size=self.rack_size,
                                              p=[p_0] + [(1 - p_0) / self.goal_level] * self.goal_level)
        if np.all(goal_pattern == goal_pattern[0]):
            goal_pattern[self._np_random.randint(len(goal_pattern))] = 0

        for ind, i in enumerate(chosed_ids):
            if ind + 1 in goal_pattern:
                goal_pattern[goal_pattern == ind + 1] = -i
        return RackState(-goal_pattern)
        # return np.repeat(self._np_random.randint(1, self.num_classes + 1, size=[1, self.rack_size[1]]),
        #                  self.rack_size[0], axis=0)

    def update_goal_level(self):
        level = self.goal_level + 1
        self.goal_level = min(level, self.num_classes)
        # print(f"Goal Level Update: {level - 1} -> {self.goal_level}")

    def update_state_level(self):
        level = self.state_level + 1
        self.state_level = min(level, self.max_state_level_per_class * self.class_level)
        # print(f"State Level Update: {level - 1} -> {self.state_level}")
        return self.state_level

    def update_class_level(self):
        level = self.class_level + 1
        self.class_level = min(level, self.num_classes)
        # print(f"Class Level Update: {level - 1} -> {self.class_level}")
        return self.class_level

    # def update_training_level(self):
    #     st_level = self.state_level
    #     cls_level = self.class_level
    #
    #     cls_level_u = self.update_class_level()
    #     if cls_level_u == cls_level:  # reach max class
    #         # update state level
    #         st_level_u = self.update_state_level()
    #         if st_level == st_level_u:  # cannot update state level (state reach max)
    #             return st_level_u, cls_level_u
    #         else:
    #             self.class_level = int(st_level / self.max_state_level_per_class) + 1
    #             return st_level_u, self.class_level
    #
    #     return st_level, cls_level_u
    #
    # def set_training_level(self, training_level):
    #     self.state_level = 1 + training_level // self.num_classes
    #     self.class_level = 1 + training_level % self.num_classes

    def update_training_level(self):
        cls_level = self.class_level
        st_level_u = self.update_state_level()
        return st_level_u, cls_level

    def set_training_level(self, training_level):
        self.state_level = training_level
        self.class_level = self.num_classes

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return f"Rack dimension is {self.rack_size}, current state level is {self.state_level}, class level is {self.class_level}"


class RackStateScheduler2(RackStateScheduler):
    def __init__(self, num_classes, rack_size, np_random=None, state_level=1, goal_level=1, class_level=1,
                 max_state_level_per_class=100, is_goalpattern_fixed=False):
        super().__init__(num_classes, rack_size, np_random, state_level, goal_level, class_level,
                         max_state_level_per_class)
        self.goal_level = goal_level
        self.is_goalpattern_fixed = is_goalpattern_fixed
        self.goal_pattern_class_num = None
        self.evolve_level = 1
        self.goal_pattern_np = None
        self.training_plan = {}

    def gen_state(self, goal_pattern):
        if isinstance(goal_pattern, RackState):
            goal_pattern_np = goal_pattern.state
        elif isinstance(goal_pattern, np.ndarray):
            goal_pattern_np = goal_pattern
        else:
            raise Exception

        goalpattern_ravel = goal_pattern_np.ravel()
        if self.goal_pattern_class_num is None:
            goal_pattern_class_num = np.zeros(self.num_classes, dtype=int)
            for i in range(self.num_classes):
                goal_pattern_class_num[i] = len(np.where(goalpattern_ravel == (i + 1))[0])
            if self.is_goalpattern_fixed:
                self.goal_pattern_class_num = goal_pattern_class_num
        else:
            goal_pattern_class_num = self.goal_pattern_class_num
        final_state_result = np.zeros(self.rack_size, dtype=int)

        state_level = self.state_level
        class_level = self.class_level
        # selected_classes = np.random.choice(range(1, self.num_classes + 1), class_level, replace=False)
        selected_classes = class_level
        if selected_classes == (self.num_classes + 1):
            selected_classes = self._np_random.choice(range(1, self.num_classes + 1),
                                                      min(self.state_level, self.num_classes),
                                                      replace=False)
        else:
            num_in_goal = state_level - self.evolve_level
            total_avaliable_num = np.prod(self.rack_size)
            # for i in range(self.num_classes):
            for i in [selected_classes - 1]:
                if num_in_goal > goal_pattern_class_num[i]:
                    raise Exception("State level should not be larger than the goal pattern class num")
                elif num_in_goal + self.evolve_level > goal_pattern_class_num[i]:
                    raise Exception(f"Too many goal need's to filled, {num_in_goal + self.evolve_level},"
                                    f"i: {goal_pattern_class_num[i]}/ {goal_pattern_class_num}")
                else:
                    if num_in_goal < 1:
                        continue
                    inds = np.argwhere(goal_pattern_np == (i + 1))
                    p = inds[self._np_random.choice(len(inds), num_in_goal, replace=False)]
                    final_state_result[p[:, 0], p[:, 1]] = i + 1
                    total_avaliable_num -= len(p)
            # for i in range(self.num_classes):
            for i in [selected_classes - 1]:
                if total_avaliable_num <= self.evolve_level:
                    raise Exception('no place to put objects')
                inds = np.argwhere(goal_pattern_np != (i + 1))
                p = inds[self._np_random.choice(len(inds), self.evolve_level, replace=False)]
                final_state_result[p[:, 0], p[:, 1]] = i + 1
                total_avaliable_num -= len(p)
                return RackState(final_state_result)
        selected_tube = [_ for _ in selected_classes]
        possible_tube_add_num = state_level - len(selected_tube)
        selected_classes_indices = selected_classes - 1
        selected_tube_tmp = []
        if possible_tube_add_num > 0:
            for ind in selected_classes_indices:
                remain_num = goal_pattern_class_num[ind] - 1
                if remain_num == 0:
                    continue
                selected_tube_tmp += [ind + 1] * remain_num
        selected_tube = self._np_random.choice(selected_tube_tmp, possible_tube_add_num,
                                               replace=False).tolist() + selected_tube
        pattern_is_done = False
        while not pattern_is_done:
            coords = self._np_random.choice(np.prod(self.rack_size), size=len(selected_tube), replace=False)
            coords = np.array(np.unravel_index(coords, self.rack_size)).T
            final_state_result[coords[:, 0], coords[:, 1]] = np.array(selected_tube)
            if not isdone(final_state_result, goal_pattern_np):
                break
            else:
                final_state_result = np.zeros(self.rack_size, dtype=int)
        return RackState(final_state_result)
        # max_len = len(np.where(goalpattern_ravel > 0)[0]) + 1
        # feasible_diff = min(max_len, self.state_level)
        # min_obj_num = self.state_level
        # state = get_random_states(self.rack_size, goal_pattern_np,
        #                           min_obj_num=min_obj_num,
        #                           obj_num=feasible_diff,
        #                           np_random=self._np_random,
        #                           max_num_class=self.class_level)

    def set_training_level(self, training_level):
        if self.is_goalpattern_fixed:
            if self.goal_pattern_np is not None:
                state_level, class_level, evolve_level = self.training_plan[max(training_level, 1)]
                self.state_level = state_level
                self.class_level = class_level
                self.evolve_level = evolve_level
                return

        self.state_level = training_level
        self.class_level = self.num_classes

    def set_goal_pattern(self, goal_pattern: np.ndarray):
        goal_pattern_np = np.array(goal_pattern)
        goalpattern_ravel = goal_pattern_np.ravel()
        goal_pattern_class_num = np.zeros(self.num_classes, dtype=int)
        for i in range(self.num_classes):
            goal_pattern_class_num[i] = len(np.where(goalpattern_ravel == (i + 1))[0])
        if self.is_goalpattern_fixed:
            self.goal_pattern_class_num = goal_pattern_class_num
            self.goal_pattern_np = goal_pattern_np
            evolve_level = 1
            state_level = 1
            class_level = 1
            training_level = 1
            total_num_test_tube = np.prod(self.rack_size)
            total_goal_num = len(np.where(goal_pattern_np > 0)[0])
            total_class_level = self.num_classes + 1
            is_update = True
            while 1:
                # if state level is larger than the total goal num => break
                if state_level > total_goal_num:
                    break
                # state level, goal level
                if is_update:
                    self.training_plan[training_level] = [state_level, class_level, evolve_level]
                is_update = True
                training_level += 1
                class_level += 1
                # class level 超过num class =》 reset class =》 reset evolve =》 reset state
                if class_level > self.num_classes:
                    # 如果 evolve_level 等于 state_level (说明state_level要加1了), class_level += 1
                    if evolve_level >= state_level != 1 and class_level <= total_class_level and self.num_classes > 1:
                        continue
                    else:
                        class_level = 1
                        evolve_level = evolve_level + 1 if evolve_level + 2 > state_level else evolve_level + 2
                        if evolve_level > state_level:
                            evolve_level = 1
                            state_level += 1
                # 检查state level是否超过所有
                if class_level < total_class_level and state_level > goal_pattern_class_num[class_level - 1]:
                    is_update = False
                    training_level -= 1
            # print("")

    def update_state_level(self):
        level = self.state_level + 1
        self.state_level = min(level, self.max_state_level_per_class * self.class_level)
        # print(f"State Level Update: {level - 1} -> {self.state_level}")
        return self.state_level

    def update_evolve_num(self):
        need_update = False
        self.evolve_level += 1
        if self.evolve_level > self.state_level:
            self.evolve_level = 1
            need_update = True
        return need_update

    def update_training_level(self):
        cls_level = self.class_level
        if self.update_evolve_num():
            st_level_u = self.update_state_level()
        else:
            st_level_u = self.state_level
        return st_level_u, cls_level


class RackStateScheduler2_5(RackStateScheduler):
    def __init__(self, num_classes, rack_size, np_random=None, state_level=1, goal_level=1, class_level=1,
                 max_state_level_per_class=100, is_goalpattern_fixed=False):
        super().__init__(num_classes, rack_size, np_random, state_level, goal_level, class_level,
                         max_state_level_per_class)
        self.goal_level = goal_level
        self.is_goalpattern_fixed = is_goalpattern_fixed
        self.goal_pattern_class_num = None
        self.evolve_level = 1
        self.goal_pattern_np = None
        self.training_plan = {}

    def gen_state(self, goal_pattern):
        if isinstance(goal_pattern, RackState):
            goal_pattern_np = goal_pattern.state
        elif isinstance(goal_pattern, np.ndarray):
            goal_pattern_np = goal_pattern
        else:
            raise Exception

        goalpattern_ravel = goal_pattern_np.ravel()
        if self.goal_pattern_class_num is None:
            goal_pattern_class_num = np.zeros(self.num_classes, dtype=int)
            for i in range(self.num_classes):
                goal_pattern_class_num[i] = len(np.where(goalpattern_ravel == (i + 1))[0])
            if self.is_goalpattern_fixed:
                self.goal_pattern_class_num = goal_pattern_class_num
        else:
            goal_pattern_class_num = self.goal_pattern_class_num
        final_state_result = np.zeros(self.rack_size, dtype=int)

        state_level = self.state_level
        class_level = self.class_level
        # selected_classes = np.random.choice(range(1, self.num_classes + 1), class_level, replace=False)
        selected_classes = self._np_random.choice(range(1, self.num_classes + 1),
                                                  min(self.state_level, class_level),
                                                  replace=False)
        if class_level == 1:
            num_in_goal = state_level - self.evolve_level
            total_avaliable_num = np.prod(self.rack_size)
            # for i in range(self.num_classes):
            for i in [selected_classes - 1]:
                if num_in_goal > goal_pattern_class_num[i]:
                    raise Exception("State level should not be larger than the goal pattern class num")
                elif num_in_goal + self.evolve_level > goal_pattern_class_num[i]:
                    raise Exception(f"Too many goal need's to filled, {num_in_goal + self.evolve_level},"
                                    f"i: {goal_pattern_class_num[i]}/ {goal_pattern_class_num}")
                else:
                    if num_in_goal < 1:
                        continue
                    inds = np.argwhere(goal_pattern_np == (i + 1))
                    p = inds[self._np_random.choice(len(inds), num_in_goal, replace=False)]
                    final_state_result[p[:, 0], p[:, 1]] = i + 1
                    total_avaliable_num -= len(p)
            # for i in range(self.num_classes):
            for i in [selected_classes - 1]:
                if total_avaliable_num <= self.evolve_level:
                    raise Exception('no place to put objects')
                inds = np.argwhere(goal_pattern_np != (i + 1))
                p = inds[self._np_random.choice(len(inds), self.evolve_level, replace=False)]
                final_state_result[p[:, 0], p[:, 1]] = i + 1
                total_avaliable_num -= len(p)
                return RackState(final_state_result)

        selected_tube = [_ for _ in selected_classes]
        possible_tube_add_num = state_level - len(selected_tube)
        selected_classes_indices = selected_classes - 1
        selected_tube_tmp = []
        if possible_tube_add_num > 0:
            for ind in selected_classes_indices:
                remain_num = goal_pattern_class_num[ind] - 1
                if remain_num == 0:
                    continue
                selected_tube_tmp += [ind + 1] * remain_num
        selected_tube = self._np_random.choice(selected_tube_tmp, possible_tube_add_num,
                                               replace=False).tolist() + selected_tube
        pattern_is_done = False
        while not pattern_is_done:
            coords = self._np_random.choice(np.prod(self.rack_size), size=len(selected_tube), replace=False)
            coords = np.array(np.unravel_index(coords, self.rack_size)).T
            final_state_result[coords[:, 0], coords[:, 1]] = np.array(selected_tube)
            if not isdone(final_state_result, goal_pattern_np):
                break
            else:
                final_state_result = np.zeros(self.rack_size, dtype=int)
        return RackState(final_state_result)

    def set_training_level(self, training_level):
        if self.is_goalpattern_fixed:
            if self.goal_pattern_np is not None:
                state_level, class_level, evolve_level = self.training_plan[max(training_level, 1)]
                self.state_level = state_level
                self.class_level = class_level
                self.evolve_level = evolve_level
                return

        self.state_level = training_level
        self.class_level = self.num_classes

    def set_goal_pattern(self, goal_pattern: np.ndarray):
        goal_pattern_np = np.array(goal_pattern)
        goalpattern_ravel = goal_pattern_np.ravel()
        goal_pattern_class_num = np.zeros(self.num_classes, dtype=int)
        for i in range(self.num_classes):
            goal_pattern_class_num[i] = len(np.where(goalpattern_ravel == (i + 1))[0])
        if self.is_goalpattern_fixed:
            self.goal_pattern_class_num = goal_pattern_class_num
            n_smallest = np.sort(goal_pattern_class_num)
            self.goal_pattern_np = goal_pattern_np
            evolve_level = 1
            state_level = 1
            class_level = 1
            training_level = 1
            class_level_init = 1
            evolve_level_init = 1
            total_num_test_tube = np.prod(self.rack_size)
            total_goal_num = len(np.where(goal_pattern_np > 0)[0])
            is_update = True
            while 1:
                # if state level is larger than the total goal num => break
                if state_level > total_goal_num:
                    break
                # state level, goal level
                if is_update:
                    self.training_plan[training_level] = [state_level, class_level, evolve_level]
                is_update = True
                training_level += 1

                evolve_level = evolve_level + 1 if evolve_level + 2 > state_level else evolve_level + 2
                if evolve_level > state_level:
                    # check if the class level can be update
                    if class_level >= state_level or class_level >= self.num_classes:
                        # class leve is greater than state (state cannot represet more class
                        # evolve

                        state_level += 1
                        if state_level > n_smallest[0]:
                            class_level_init += 1
                            if class_level_init > self.num_classes:
                                class_level_init = self.num_classes
                            evolve_level_init = state_level
                        evolve_level = evolve_level_init
                        class_level = class_level_init
                        # to do if the number of each goal is not equal
                    else:
                        class_level += 1
                        evolve_level -= 1
                # if class_level < total_class_level and state_level > goal_pattern_class_num[class_level - 1]:
                #     is_update = False
                #     training_level -= 1
            # print("")

    def update_state_level(self):
        level = self.state_level + 1
        self.state_level = min(level, self.max_state_level_per_class * self.class_level)
        # print(f"State Level Update: {level - 1} -> {self.state_level}")
        return self.state_level

    def update_evolve_num(self):
        need_update = False
        self.evolve_level += 1
        if self.evolve_level > self.state_level:
            self.evolve_level = 1
            need_update = True
        return need_update

    def update_training_level(self):
        cls_level = self.class_level
        if self.update_evolve_num():
            st_level_u = self.update_state_level()
        else:
            st_level_u = self.state_level
        return st_level_u, cls_level


class RackStateScheduler3(RackStateScheduler):
    def __init__(self, num_classes, rack_size, np_random=None, state_level=1, goal_level=1, class_level=1,
                 max_state_level_per_class=100, is_goalpattern_fixed=False):
        super().__init__(num_classes, rack_size, np_random, state_level, goal_level, class_level,
                         max_state_level_per_class)
        self.goal_level = goal_level
        self.is_goalpattern_fixed = is_goalpattern_fixed
        self.goal_pattern_class_num = None
        self.evolve_level = 1
        self.goal_pattern_np = None
        self.training_plan = {}

    def gen_state(self, goal_pattern):
        if isinstance(goal_pattern, RackState):
            goal_pattern_np = goal_pattern.state
        elif isinstance(goal_pattern, np.ndarray):
            goal_pattern_np = goal_pattern
        else:
            raise Exception

        goalpattern_ravel = goal_pattern_np.ravel()
        if self.goal_pattern_class_num is None:
            goal_pattern_class_num = np.zeros(self.num_classes, dtype=int)
            for i in range(self.num_classes):
                goal_pattern_class_num[i] = len(np.where(goalpattern_ravel == (i + 1))[0])
            if self.is_goalpattern_fixed:
                self.goal_pattern_class_num = goal_pattern_class_num
        else:
            goal_pattern_class_num = self.goal_pattern_class_num
        final_state_result = np.zeros(self.rack_size, dtype=int)

        state_level = self.state_level
        class_level = self.class_level
        # selected_classes = np.random.choice(range(1, self.num_classes + 1), class_level, replace=False)
        selected_classes = class_level
        if selected_classes == (self.num_classes + 1):
            selected_classes = self._np_random.choice(range(1, self.num_classes + 1),
                                                      min(self.state_level, self.num_classes),
                                                      replace=False)
        else:
            num_in_goal = state_level - self.evolve_level
            total_avaliable_num = np.prod(self.rack_size)
            # for i in range(self.num_classes):
            for i in [selected_classes - 1]:
                if num_in_goal > goal_pattern_class_num[i]:
                    raise Exception("State level should not be larger than the goal pattern class num")
                elif num_in_goal + self.evolve_level > goal_pattern_class_num[i]:
                    raise Exception(f"Too many goal need's to filled, {num_in_goal + self.evolve_level},"
                                    f"i: {goal_pattern_class_num[i]}/ {goal_pattern_class_num}")
                else:
                    if num_in_goal < 1:
                        continue
                    inds = np.argwhere(goal_pattern_np == (i + 1))
                    p = inds[self._np_random.choice(len(inds), num_in_goal, replace=False)]
                    final_state_result[p[:, 0], p[:, 1]] = i + 1
                    total_avaliable_num -= len(p)
            # for i in range(self.num_classes):
            for i in [selected_classes - 1]:
                if total_avaliable_num <= self.evolve_level:
                    raise Exception('no place to put objects')
                inds = np.argwhere(goal_pattern_np != (i + 1))
                p = inds[self._np_random.choice(len(inds), self.evolve_level, replace=False)]
                final_state_result[p[:, 0], p[:, 1]] = i + 1
                total_avaliable_num -= len(p)
                return RackState(final_state_result)
        selected_tube = [_ for _ in selected_classes]
        possible_tube_add_num = state_level - len(selected_tube)
        selected_classes_indices = selected_classes - 1
        selected_tube_tmp = []
        if possible_tube_add_num > 0:
            for ind in selected_classes_indices:
                remain_num = goal_pattern_class_num[ind] - 1
                if remain_num == 0:
                    continue
                selected_tube_tmp += [ind + 1] * remain_num
        selected_tube = self._np_random.choice(selected_tube_tmp, possible_tube_add_num,
                                               replace=False).tolist() + selected_tube
        pattern_is_done = False
        while not pattern_is_done:
            coords = self._np_random.choice(np.prod(self.rack_size), size=len(selected_tube), replace=False)
            coords = np.array(np.unravel_index(coords, self.rack_size)).T
            final_state_result[coords[:, 0], coords[:, 1]] = np.array(selected_tube)
            if not isdone(final_state_result, goal_pattern_np):
                break
            else:
                final_state_result = np.zeros(self.rack_size, dtype=int)
        return RackState(final_state_result)
        # max_len = len(np.where(goalpattern_ravel > 0)[0]) + 1
        # feasible_diff = min(max_len, self.state_level)
        # min_obj_num = self.state_level
        # state = get_random_states(self.rack_size, goal_pattern_np,
        #                           min_obj_num=min_obj_num,
        #                           obj_num=feasible_diff,
        #                           np_random=self._np_random,
        #                           max_num_class=self.class_level)

    def set_training_level(self, training_level):
        if self.is_goalpattern_fixed:
            if self.goal_pattern_np is not None:
                state_level, class_level, evolve_level = self.training_plan[max(training_level, 1)]
                self.state_level = state_level
                self.class_level = class_level
                self.evolve_level = evolve_level
                return

        self.state_level = training_level
        self.class_level = self.num_classes

    def set_goal_pattern(self, goal_pattern: np.ndarray):
        goal_pattern_np = np.array(goal_pattern)
        goalpattern_ravel = goal_pattern_np.ravel()
        goal_pattern_class_num = np.zeros(self.num_classes, dtype=int)
        for i in range(self.num_classes):
            goal_pattern_class_num[i] = len(np.where(goalpattern_ravel == (i + 1))[0])
        if self.is_goalpattern_fixed:
            self.goal_pattern_class_num = goal_pattern_class_num
            self.goal_pattern_np = goal_pattern_np
            evolve_level = 1
            state_level = 1
            class_level = 1
            training_level = 1
            total_num_test_tube = np.prod(self.rack_size)
            total_goal_num = len(np.where(goal_pattern_np > 0)[0])
            total_class_level = self.num_classes + 1
            is_update = True
            while 1:
                # if state level is larger than the total goal num => break
                if state_level > total_goal_num:
                    break
                # state level, goal level
                if is_update:
                    self.training_plan[training_level] = [state_level, class_level, evolve_level]
                is_update = True
                training_level += 1
                class_level += 1
                # class level 超过num class =》 reset class =》 reset evolve =》 reset state
                if class_level > self.num_classes:
                    # 如果 evolve_level 等于 state_level (说明state_level要加1了), class_level += 1
                    if evolve_level == state_level != 1 and class_level <= total_class_level and self.num_classes > 1:
                        continue
                    else:
                        class_level = 1
                        evolve_level += 1
                        if evolve_level > state_level:
                            evolve_level = 1
                            state_level += 1
                # 检查state level是否超过所有
                if class_level < total_class_level and state_level > goal_pattern_class_num[class_level - 1]:
                    is_update = False
                    training_level -= 1
            # print("")

    def update_state_level(self):
        level = self.state_level + 1
        self.state_level = min(level, self.max_state_level_per_class * self.class_level)
        # print(f"State Level Update: {level - 1} -> {self.state_level}")
        return self.state_level

    def update_evolve_num(self):
        need_update = False
        self.evolve_level += 1
        if self.evolve_level > self.state_level:
            self.evolve_level = 1
            need_update = True
        return need_update

    def update_training_level(self):
        cls_level = self.class_level
        if self.update_evolve_num():
            st_level_u = self.update_state_level()
        else:
            st_level_u = self.state_level
        return st_level_u, cls_level


class StripeRackStateScheduler(RackStateScheduler):
    def gen_goal(self, no_curriculum=False):
        if no_curriculum:
            chosed_ids = self._np_random.choice(self.class_ids, self.num_classes, replace=False)
        else:
            chosed_ids = self._np_random.choice(self.class_ids, min(self.class_level, self.num_classes), replace=False)
        # print("chosed ids", chosed_ids)
        # goal_pattern = self._np_random.randint(0, self.goal_level + 1, size=self.rack_size)
        num_n_pattern = max(int(self.state_level / self.rack_size[0]) + 1, self.class_level)

        p_0 = self._np_random.randint(1, self.rack_size[1] - num_n_pattern + 1)
        goal_pattern = np.zeros((self.rack_size[1]), dtype=int)
        goal_pattern_necessary_idx = self._np_random.choice(np.arange(len(goal_pattern)), size=self.class_level,
                                                            replace=False)
        res_goal_pattern_idx = np.delete(np.arange(len(goal_pattern)), goal_pattern_necessary_idx)

        goal_pattern[goal_pattern_necessary_idx] = self._np_random.choice(chosed_ids,
                                                                          size=len(goal_pattern_necessary_idx),
                                                                          replace=False)
        goal_pattern[res_goal_pattern_idx] = self._np_random.choice(chosed_ids,
                                                                    size=len(res_goal_pattern_idx), )
        goal_pattern[self._np_random.choice(res_goal_pattern_idx, size=p_0, replace=False)] = 0
        # print(goal_pattern)
        if np.all(goal_pattern == goal_pattern[0]):
            goal_pattern[self._np_random.randint(len(goal_pattern))] = 0
        return RackState(np.repeat(goal_pattern[None, :], self.rack_size[0], axis=0))


class GoalRackStateScheduler(RackStateScheduler):
    """
    Goal pattern is randomly separate in the grids
    """

    def gen_goal(self, no_curriculum=False):
        if no_curriculum:
            chosed_ids = self._np_random.choice(self.class_ids, self.num_classes, replace=False)
            num_goals = self._np_random.choice(np.arange(1, np.prod(self.rack_size)))
        else:
            chosed_ids = self._np_random.choice(self.class_ids, min(self.class_level, self.num_classes), replace=False)
            num_goals = self.state_level
        # print("chosed ids", chosed_ids)
        # goal_pattern = self._np_random.randint(0, self.goal_level + 1, size=self.rack_size)

        rack = np.zeros(np.prod(self.rack_size), dtype=int)
        goal_pattern_necessary_idx = self._np_random.choice(np.arange(np.prod(self.rack_size)), size=num_goals,
                                                            replace=False)
        if len(chosed_ids) >= num_goals:
            rack[goal_pattern_necessary_idx] = self._np_random.choice(chosed_ids, size=num_goals, replace=False)
        else:
            rack[goal_pattern_necessary_idx] = np.concatenate((
                self._np_random.choice(chosed_ids, size=len(chosed_ids), replace=False),
                self._np_random.choice(chosed_ids, size=num_goals - len(chosed_ids))
            ))

        return RackState(rack.reshape(self.rack_size))


class GoalRackStateScheduler3(RackStateScheduler):
    """
    Goal pattern is connected instead randomly seperated
    """

    @staticmethod
    def generate_connected_grid(m, n, k):
        """
        :param m: rows
        :param n: columns
        :param k: number of objects
        :return:
        """
        # Ensure k is less than or equal to m*n
        if k > m * n:
            raise ValueError("k must be less than or equal to m*n")

        # Define directions for DFS (up, down, left, right)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        def dfs(x, y, k, visited, directions):
            if k == 0:
                return [(x, y)]

            visited[x, y] = True

            for dx, dy in np.random.permutation(
                    directions):  # Use permutation instead of shuffle for in-place operations
                nx, ny = x + dx, y + dy

                # Inline is_valid check here
                if 0 <= nx < m and 0 <= ny < n and not visited[nx, ny]:
                    result = dfs(nx, ny, k - 1, visited, directions)
                    if result:
                        return [(x, y)] + result

            visited[x, y] = False
            return None

        # Initialize visited grid using numpy arrays
        visited = np.zeros((m, n), dtype=np.bool_)

        # Start DFS from a random cell using numpy
        start_x, start_y = np.random.randint(0, m), np.random.randint(0, n)
        sampled_cells = dfs(start_x, start_y, k - 1, visited, directions)

        # Create the resulting grid
        result = np.zeros((m, n), dtype=int)
        for x, y in sampled_cells:
            result[x, y] = 1

        return result

    def gen_goal(self, no_curriculum=False):
        if no_curriculum:
            chosed_ids = self._np_random.choice(self.class_ids, self.num_classes, replace=False)
            num_goals = self._np_random.choice(np.arange(1, np.prod(self.rack_size)))
        else:
            chosed_ids = self._np_random.choice(self.class_ids, min(self.class_level, self.num_classes), replace=False)
            num_goals = self.state_level
        # print("chosed ids", chosed_ids)
        # goal_pattern = self._np_random.randint(0, self.goal_level + 1, size=self.rack_size)
        # if num_goals >= np.prod(self.rack_size) / 2:
        #     rack = np.zeros(np.prod(self.rack_size), dtype=int)
        #     goal_pattern_necessary_idx = self._np_random.choice(np.arange(np.prod(self.rack_size)), size=num_goals,
        #                                                         replace=False)
        # else:
        rack = np.zeros(np.prod(self.rack_size), dtype=int)
        goal_pattern_necessary_idx = np.where(self.generate_connected_grid(*self.rack_size, num_goals).flatten())[0]
        if len(chosed_ids) >= num_goals:
            rack[goal_pattern_necessary_idx] = self._np_random.choice(chosed_ids, size=num_goals, replace=False)
        else:
            rack[goal_pattern_necessary_idx] = np.concatenate((
                self._np_random.choice(chosed_ids, size=len(chosed_ids), replace=False),
                self._np_random.choice(chosed_ids, size=num_goals - len(chosed_ids))
            ))
            # rack[goal_pattern_necessary_idx] = self._np_random.choice(chosed_ids, size=num_goals)

        return RackState(rack.reshape(self.rack_size))


class GoalRackStateScheduler2(GoalRackStateScheduler):
    def gen_goal(self, no_curriculum=False):
        # chosed_ids = self.class_ids
        chosed_ids = self._np_random.choice(self.class_ids, min(self.class_level, self.num_classes), replace=False)
        num_goals = self._np_random.choice(np.arange(self.state_level, np.prod(self.rack_size) - self.state_level + 1))
        rack = np.zeros(np.prod(self.rack_size), dtype=int)
        goal_pattern_necessary_idx = self._np_random.choice(np.arange(np.prod(self.rack_size)), size=num_goals,
                                                            replace=False)
        if len(chosed_ids) >= num_goals:
            rack[goal_pattern_necessary_idx] = self._np_random.choice(chosed_ids, size=num_goals, replace=False)
        else:
            rack[goal_pattern_necessary_idx] = np.concatenate((
                self._np_random.choice(chosed_ids, size=len(chosed_ids), replace=False),
                self._np_random.choice(chosed_ids, size=num_goals - len(chosed_ids))
            ))

        return RackState(rack.reshape(self.rack_size))

    def gen_state(self, goal_pattern):
        if isinstance(goal_pattern, RackState):
            goal_pattern_np = goal_pattern.state
        elif isinstance(goal_pattern, np.ndarray):
            goal_pattern_np = goal_pattern
        else:
            raise Exception("Undefined Goal pattern")
        goalpattern_ravel = goal_pattern_np.ravel()
        state = goalpattern_ravel.copy()
        state_filled_inds = np.where(state > 0)[0]
        state_empty_inds = np.where(state == 0)[0]
        if len(state_filled_inds) < self.state_level or len(state_empty_inds) < self.state_level:
            raise Exception("Infeasible Goal Pattern")
        state_filled_inds_selected = self._np_random.choice(state_filled_inds, size=self.state_level, replace=False)
        state_empty_inds_selected = self._np_random.choice(state_empty_inds, size=self.state_level, replace=False)
        state[state_filled_inds_selected], state[state_empty_inds_selected] = \
            state[state_empty_inds_selected], state[state_filled_inds_selected]
        return RackState(state.reshape(self.rack_size))


if __name__ == "__main__":
    import cv2

    rack_size = (5, 10)

    obs_dim, act_dim = RackState.get_obs_act_dim_by_size(rack_size)

    num_classes = 1
    observation_space_dim = obs_dim
    action_space_dim = act_dim

    env = RackArrangementEnv(rack_size=rack_size,
                             num_classes=num_classes,
                             observation_space_dim=observation_space_dim,
                             action_space_dim=action_space_dim,
                             is_goalpattern_fixed=True,
                             is_curriculum_lr=True,
                             scheduler='RackStateScheduler2',
                             seed=np.random.randint(999))

    env.set_goal_pattern(GOAL_PATTERN_5x10)
    scheduler = env.scheduler
    # scheduler.max_state_level_per_class = min(np.bincount(GOAL_PATTERN_5x10.ravel())[1:])
    print(scheduler)
    # scheduler.state_level = 7
    # scheduler.class_level = 2
    # scheduler.max_state_level_per_class = 45
    while 1:
        # print(scheduler.state_level == (scheduler.max_state_level_per_class))
        # if scheduler.state_level == (scheduler.max_state_level_per_class):
        #     scheduler.update_class_level()
        #     scheduler.state_level = scheduler.class_level
        for i in range(10):
            print(scheduler)
            s = env.reset()
            print(np.array(env))
            a = env.sample()
            nxt_state, reward, is_finished, _ = env.step(a)
            a = time.time()
            nxt_state.to_abs_state
            b = time.time()
            print(f"t: {b - a}")
            a = time.time()
            nxt_state.to_abs_state
            b = time.time()
            print(f"t: {b - a}")
            nxt_state_cp = nxt_state.copy()
            a = time.time()
            nxt_state_cp.to_abs_state
            b = time.time()
            print(f"t: {b - a}")

            print("reward is ", reward)
            print("GOAL pattern is", repr(np.array(env.goal_pattern)))
            drawer = RackStatePlot(env.goal_pattern, )
            fig = drawer.plot_states([s, nxt_state, ]).get_img()
            fig = drawer.plot_states([s, ]).get_img()
            cv2.imshow("test", fig)
            cv2.waitKey(0)

        scheduler.update_training_level()
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
