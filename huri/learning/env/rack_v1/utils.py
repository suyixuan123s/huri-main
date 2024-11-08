import time
from typing import Optional

import scipy.signal as ss
import numpy as np
from numba import njit

mask_ucbc = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
mask_crcl = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])
mask_ul = np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]])
mask_ur = np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]])
mask_bl = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]])
mask_br = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]])


def mask_2D(mask_size=3):
    '''
    Return a 2D grid with specified spacing

    Arguments
    ---------
    bounds: (2,2) list of [[minx, miny], [maxx, maxy]]
    step:   float, separation between points

    Returns
    -------
    grid: (n, 2) list of 2D points
    '''
    x_mask = np.arange(-mask_size, mask_size + 1, step=1)
    y_mask = np.arange(-mask_size, mask_size + 1, step=1)
    mask = np.dstack(np.meshgrid(x_mask, y_mask)).reshape(-1, 2)
    mask[:, [0, 1]] = mask[:, [1, 0]]
    return mask


def get_possible_actions(node: np.ndarray):
    node = node.copy()
    node_pad = np.pad(node, pad_width=2)
    cg_ucbc = ss.correlate2d(node, mask_ucbc)[1:-1, 1:-1]
    cg_crcl = ss.correlate2d(node, mask_crcl)[1:-1, 1:-1]
    cg_ul = ss.correlate2d(node, mask_ul)[1:-1, 1:-1]
    cg_ur = ss.correlate2d(node, mask_ur)[1:-1, 1:-1]
    cg_bl = ss.correlate2d(node, mask_bl)[1:-1, 1:-1]
    cg_br = ss.correlate2d(node, mask_br)[1:-1, 1:-1]
    cf = ((cg_ucbc == 0) + (cg_crcl == 0) + (cg_ul == 0) + (cg_ur == 0) + (cg_bl == 0) + (cg_br == 0)) * (
            node == 0)
    cg_ucbc[node == 0] = -1
    cg_crcl[node == 0] = -1
    cg_ul[node == 0] = -1
    cg_ur[node == 0] = -1
    cg_bl[node == 0] = -1
    cg_br[node == 0] = -1
    cg = (cg_ucbc == 0) + (cg_crcl == 0) + (cg_ul == 0) + (cg_ur == 0) + (cg_bl == 0) + (cg_br == 0)
    fillable_coord = np.vstack(np.where(cf)).T
    movable_coord = np.vstack(np.where(cg)).T

    movable_id_fillable_indices = np.tile(np.arange(len(fillable_coord)), len(movable_coord))
    fillable_possible_actions = np.concatenate((
        np.repeat(np.arange(len(movable_coord)), len(fillable_coord))[..., None],
        fillable_coord[movable_id_fillable_indices]), axis=1)
    # 1. Extract a 5x5 matrices
    offsets = mask_2D(2) + 2
    fill = (offsets + movable_coord[:, None]).reshape(-1, 2)
    arr5x5 = node_pad[fill[:, 0], fill[:, 1]].reshape(-1, 5, 5)
    # 2. Choose the slot to be checked after picking the tube
    arr3x3 = arr5x5[:, 1:-1, 1:-1]
    arr3x3_0_coord = np.vstack(np.where(arr3x3 == 0)).T
    arr5x5[:, 2, 2] = 0
    offsets = mask_2D(1) + 1
    fill = (np.concatenate((np.zeros((len(offsets), 1), dtype=int), offsets), axis=1) + arr3x3_0_coord[:,
                                                                                        None]).reshape(-1, 3)
    fillable_slots_check = arr5x5[fill[:, 0], fill[:, 1], fill[:, 2]].reshape(-1, 3, 3)

    fillable_coord_bool = (np.sum((fillable_slots_check * mask_ucbc).reshape(len(fillable_slots_check), -1),
                                  axis=1) == 0) | (np.sum(
        (fillable_slots_check * mask_crcl).reshape(len(fillable_slots_check), -1), axis=1) == 0) | (np.sum(
        (fillable_slots_check * mask_ul).reshape(len(fillable_slots_check), -1), axis=1) == 0) | (np.sum(
        (fillable_slots_check * mask_ur).reshape(len(fillable_slots_check), -1), axis=1) == 0) | (np.sum(
        (fillable_slots_check * mask_bl).reshape(len(fillable_slots_check), -1), axis=1) == 0) | (np.sum(
        (fillable_slots_check * mask_br).reshape(len(fillable_slots_check), -1), axis=1) == 0)

    movable_coord_repeated = np.zeros((len(arr3x3_0_coord[fillable_coord_bool]), 3), dtype=int)
    movable_coord_repeated[:, 0] = arr3x3_0_coord[fillable_coord_bool][:, 0]
    movable_coord_repeated[:, 1:] = movable_coord[arr3x3_0_coord[fillable_coord_bool][:, 0]]
    movable_coord_repeated[:, 1:] = movable_coord_repeated[:, 1:] + arr3x3_0_coord[fillable_coord_bool][:, 1:] - 1
    movable_coord_repeated = movable_coord_repeated[
        (movable_coord_repeated[:, 1] >= 0) & (movable_coord_repeated[:, 1] < node.shape[0]) & (
                movable_coord_repeated[:, 2] >= 0) & (movable_coord_repeated[:, 2] < node.shape[1])]

    act = np.unique(np.vstack((movable_coord_repeated, fillable_possible_actions)), axis=0)
    # act = np.unique(np.vstack((movable_coord_repeated, fillable_possible_actions,
    #                            np.concatenate((np.arange(len(movable_coord))[..., None], movable_coord), axis=1))),
    #                 axis=0)

    fillable_acts = act[act[:, 0].argsort()]
    movable_acts = movable_coord[fillable_acts[:, 0]]
    return np.concatenate((movable_acts, fillable_acts[:, 1:]), axis=1), movable_coord, fillable_acts


def get_fillable_movable(node: np.ndarray):
    node = node.copy()
    cg_ucbc = ss.correlate2d(node, mask_ucbc)[1:-1, 1:-1]
    cg_crcl = ss.correlate2d(node, mask_crcl)[1:-1, 1:-1]
    cg_ul = ss.correlate2d(node, mask_ul)[1:-1, 1:-1]
    cg_ur = ss.correlate2d(node, mask_ur)[1:-1, 1:-1]
    cg_bl = ss.correlate2d(node, mask_bl)[1:-1, 1:-1]
    cg_br = ss.correlate2d(node, mask_br)[1:-1, 1:-1]
    cf = ((cg_ucbc == 0) + (cg_crcl == 0) + (cg_ul == 0) + (cg_ur == 0) + (cg_bl == 0) + (cg_br == 0)) * (
            node == 0)
    cg_ucbc[node == 0] = -1
    cg_crcl[node == 0] = -1
    cg_ul[node == 0] = -1
    cg_ur[node == 0] = -1
    cg_bl[node == 0] = -1
    cg_br[node == 0] = -1
    cg = (cg_ucbc == 0) + (cg_crcl == 0) + (cg_ul == 0) + (cg_ur == 0) + (cg_bl == 0) + (cg_br == 0)
    fillable_matrix = cf.astype(int)
    movable_matrix = cg.astype(int)
    return fillable_matrix, movable_matrix


def isdone(node, goalpattern):
    if np.any((goalpattern - node)[node > 0]):
        return False
    return True


# @njit
# def isdone(node, goalpattern):
#     is_not_done = False
#     for i in range(1, np.max(goalpattern) + 1):
#         is_not_done = is_not_done or np.any((goalpattern != i) * (node == i))
#         if is_not_done:
#             return False
#     return True

@njit
def check_is_repeat(old_states, new_state):
    for state in old_states:
        if np.all(new_state == state):
            return True
    return False


@njit
def check_repetition_reward(old_states, new_state):
    if check_is_repeat(old_states, new_state):
        return -5
    return 0


@njit
def get_tube_layer_id(state, layer, num_classes):
    state_layers = state[num_classes:num_classes * 2]
    for idx, state_layer in enumerate(state_layers):
        if np.sum(state_layer * layer) == 1:
            return idx + num_classes
    return None


def check_feasible(elearray, goal_patten):
    fillable, movable = get_fillable_movable(elearray)
    # is fillable and movable
    if np.sum(fillable) == 0 or np.sum(movable) == 0:
        return False
    for i in range(1, np.max(elearray) + 1):
        if np.count_nonzero(goal_patten == i) < np.count_nonzero(elearray == i):
            return False
    return True


def get_random_states(rack_size, goalpattern, obj_num=None, min_obj_num=None, np_random=None, max_num_class=None):
    if np_random is None:
        np_random = np.random
    rack_len = np.prod(rack_size)
    goalpattern_ravel = goalpattern.ravel()
    goalpattern_classes = np.unique(goalpattern_ravel)
    max_num_class = min(len(goalpattern_classes), max_num_class)
    if max_num_class == len(goalpattern_classes):
        max_num_class = None
    goal_slot_idx = np.where(goalpattern_ravel > 0)[0]
    goal_slot_len = len(goal_slot_idx)
    if obj_num is None:
        obj_num = len(goal_slot_idx) + 1
    if min_obj_num is None:
        min_obj_num = 1
    obj_num = min(len(goal_slot_idx) + 1, obj_num)
    min_obj_num = max(1, min_obj_num)
    while True:
        if min_obj_num == obj_num:
            num_random = min_obj_num
        else:
            num_random = np_random.randint(min_obj_num, obj_num)
        random_choiced_id = np_random.choice(range(rack_len), size=num_random, replace=False)
        elearray = np.zeros(rack_len)
        # select from goal
        if max_num_class is None:
            goal_selected = goal_slot_idx[np_random.choice(range(goal_slot_len), size=num_random, replace=False)]
            elearray[random_choiced_id] = goalpattern_ravel[goal_selected]
        else:
            classes_selected = np_random.choice(goalpattern_classes, max_num_class, replace=False)
            elearray[random_choiced_id] = np_random.choice(classes_selected, num_random, replace=True)
        elearray = elearray.reshape(rack_size).astype(int)
        if not check_feasible(elearray, goalpattern):
            continue
        # all the compo
        # is not done and not repeat
        if not isdone(elearray, goalpattern):
            # if not check_is_repeat(np.array(state_trained), elearray):
            break
    # state_trained.append(elearray)
    return elearray


def get_random_goal_pattern(num_classes, rack_size, np_random=None):
    if np_random is None:
        np_random = np.random
    while True:
        goal_pattern = np_random.randint(0, num_classes + 1, size=rack_size)
        if np.sum(goal_pattern) == 0 or np.all(goal_pattern == goal_pattern[0]):
            continue
        return goal_pattern


def how_much_rack_is_on_its_place(state_r, goal_pattern_r):
    return np.sum(((state_r == goal_pattern_r) & (state_r > 0)).astype(np.int))


def seperate_matrix_entry(elarray: Optional[np.ndarray]):
    if elarray is None:
        return None
    elarraydim = elarray.shape[0] * elarray.shape[1]
    output_array = np.zeros((elarraydim, elarray.shape[0], elarray.shape[1]))
    for i in range(elarraydim):
        mask = np.zeros(elarraydim)
        mask[i] = 1
        output_array[i] = mask.reshape(elarray.shape[0], elarray.shape[1]) * elarray
    return output_array


def seperate_matrix_layer(elarray, num_of_classes):
    output_array = np.zeros((num_of_classes, elarray.shape[0], elarray.shape[1]))
    for i in range(1, num_of_classes + 1):
        output_array[i - 1][elarray == i] = 1
    return output_array


def merge_matrix_layer(elarray, num_of_classes):
    boardcaster = np.repeat(np.arange(0, num_of_classes) + 1, np.prod(elarray.shape) / num_of_classes).reshape(
        elarray.shape)
    return np.sum(elarray * boardcaster, axis=0)
