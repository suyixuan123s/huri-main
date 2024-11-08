import numpy as np
from huri.learning.env.rack_v3.utils import get_satisfied_constraints


def get_3x3_local_state(state, center_id):
    return np.pad(state, 1)[center_id[0]:center_id[0] + 3,
           center_id[1]:center_id[1] + 3].copy()


angle_ranges_rad = {
    0: (-np.pi / 12, np.pi / 12),
    1: (np.pi / 2 - np.pi / 12, np.pi / 2 + np.pi / 12),
    2: (-np.pi / 2 + np.pi / 5, -np.pi / 5),
    3: (np.pi / 5, np.pi / 2 - np.pi / 5),
    4: (np.pi / 5, np.pi / 2 - np.pi / 5),
    5: (-np.pi / 2 + np.pi / 5, -np.pi / 5),
}

angle_ranges_rad2 = {
    0: (-np.pi / 12, np.pi / 12),
    1: (np.pi / 2 - np.pi / 12, np.pi / 2 + np.pi / 12),
    2: (-np.pi / 2 + np.pi / 5, -np.pi / 5),
    3: (np.pi / 5, np.pi / 2 - np.pi / 5),
    4: (np.pi / 5 + np.pi / 2, np.pi / 2 - np.pi / 5 + np.pi / 2),
    5: (-np.pi / 2 + np.pi / 5 + np.pi / 2, -np.pi / 5 + np.pi / 2),
}

angle_ranges_rad2_p = {
    0: [1, -1],
    1: [1, -1],
    2: [1],
    3: [1],
    4: [-1],
    5: [-1],
}


# Function to find the feasible ranges in radians
def find_feasible_ranges_rad(matrix, angle_ranges_rad):
    feasible_ranges_rad = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # Skip the central cell (1,1)
            if (i, j) == (1, 1):
                continue
            if matrix[i, j] == 1:
                feasible_ranges_rad.append(angle_ranges_rad[(i, j)])
    return feasible_ranges_rad


# Function to check if a vector is within the feasible angle ranges
def is_vector_in_range_rad(vector, angle_ranges_rad):
    angle = np.arctan2(vector[1], vector[0])
    # if angle < 0:
    #     angle += 2 * np.pi
    for range_start, range_end in angle_ranges_rad:
        if range_start <= angle <= range_end:
            # print('angle', range_start, angle, range_end)
            return True
    return False


import modeling.geometric_model as gm


def grasp_filter(state, coord, tube_rack, grasp_info_list, conditions=None):
    feasible_ranges_rad = get_feasible_ranges_rad(state, coord, conditions)
    # print(feasible_ranges_rad)
    tube_local_rot = tube_rack.tubes_pose[coord[0], coord[1]][:3, :3]
    grasp_rot = [tube_local_rot.dot(np.array(jaw_center_rotmat))[:, 0][:2] for
                 (jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat) in grasp_info_list]
    # print(grasp_rot)
    filtered_grasp_info_list = [grasp_info_list[fid] for fid, v in enumerate(grasp_rot) if
                                is_vector_in_range_rad(v, feasible_ranges_rad)]

    # for jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat in grasp_info_list:
    #     gm.gen_frame(pos=np.array(jaw_center_pos), rotmat=np.array(jaw_center_rotmat)).attach_to(base)

    return filtered_grasp_info_list


def get_feasible_ranges_rad(state, coord, conditions=None):
    if conditions is not None:
        avaliable_mask = np.zeros((6,), dtype=bool)
        avaliable_mask[conditions - 1] = 1
        satisfied_cons_mask = avaliable_mask
    else:
        pick_local_region = get_3x3_local_state(state, (coord[0], coord[1]))
        satisfied_cons_mask = get_satisfied_constraints(pick_local_region)
        # satisfied_cons_mask = np.array([False, True, False, False, False, False])
    feasible_ranges_rad = []
    for _ in range(len(satisfied_cons_mask)):
        if satisfied_cons_mask[_]:
            feasible_ranges_rad.append(angle_ranges_rad[_])
    feasible_ranges_rad.extend([(v[0] + np.pi, v[1] + np.pi) for v in feasible_ranges_rad]
                               + [(v[0] - np.pi, v[1] - np.pi) for v in feasible_ranges_rad])
    return feasible_ranges_rad


def is_grasp_feasible(grasp_rot, feasible_ranges_rad):
    grasp_rot_local = grasp_rot[:2, 0]
    return is_vector_in_range_rad(grasp_rot_local, feasible_ranges_rad)


def grasp_filter2(state, coord, tube_rack, grasp_info_list, conditions=None):
    feasible_ranges_rad = get_feasible_ranges_rad2(state, coord, conditions)
    # print(feasible_ranges_rad)
    tube_local_rot = tube_rack.tubes_pose[coord[0], coord[1]][:3, :3]

    # [:, 0][:2]
    grasp_rot = [tube_local_rot.dot(np.array(jaw_center_rotmat)) for
                 (jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat) in grasp_info_list]
    # print(grasp_rot)
    filtered_grasp_info_list = [grasp_info_list[fid] for fid, v in enumerate(grasp_rot) if
                                is_vector_in_range_rad2(v, feasible_ranges_rad)]

    # for jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat in grasp_info_list:
    #     gm.gen_frame(pos=np.array(jaw_center_pos), rotmat=np.array(jaw_center_rotmat)).attach_to(base)

    return filtered_grasp_info_list


def is_vector_in_range_rad2(rotmat, angle_range_rad):
    x_vector2d = rotmat[:2, 0]
    z_vector = rotmat[:3, 2]
    angle = np.arctan2(x_vector2d[1], x_vector2d[0])
    if angle < 0:
        angle += 2 * np.pi
    for range_start, range_end, sign in angle_range_rad:
        if range_start <= angle <= range_end:
            print('angle', range_start, angle, range_end)
            zangle = np.arctan2(x_vector2d[1], x_vector2d[0])
            zangle_sign = np.sign(zangle)
            print('z_vector', z_vector, zangle)
            if zangle_sign not in sign:
                return False
            return True
    return False


def get_feasible_ranges_rad2(state, coord, conditions=None):
    if conditions is not None:
        avaliable_mask = np.zeros((6,), dtype=bool)
        avaliable_mask[conditions - 1] = 1
        satisfied_cons_mask = avaliable_mask
    else:
        pick_local_region = get_3x3_local_state(state, (coord[0], coord[1]))
        satisfied_cons_mask = get_satisfied_constraints(pick_local_region)
        # satisfied_cons_mask = np.array([False, True, False, False, False, False])
    feasible_ranges_rad = []
    for _ in range(len(satisfied_cons_mask)):
        if satisfied_cons_mask[_]:
            feasible_ranges_rad.append((*angle_ranges_rad2[_], angle_ranges_rad2_p[_]))
    feasible_ranges_rad.extend([(v[0] + np.pi, v[1] + np.pi, v[2]) for v in feasible_ranges_rad])
    return feasible_ranges_rad
