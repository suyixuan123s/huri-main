import multiprocessing as mlp
import itertools
import numpy as np
import direct.stdpy.pickle

def _get_available_graspids_ik(graspid_and_graspinfo_list: list,
                            goal_pos: np.ndarray,
                            goal_rotmat: np.ndarray,
                            robot_s,
                            hand_name,
                            ):

    ik_list= []
    for graspid, grasp_info in graspid_and_graspinfo_list:
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        goal_jaw_center_pos = goal_pos + goal_rotmat.dot(jaw_center_pos)
        goal_jaw_center_rotmat = goal_rotmat.dot(jaw_center_rotmat)
        jnt_values = robot_s.ik(hand_name, goal_jaw_center_pos, goal_jaw_center_rotmat)
        ik_list.append(jnt_values)
    return ik_list


def get_available_graspids_ik(graspid_and_graspinfo_list,
                           goal_pos: np.ndarray,
                           goal_rotmat: np.ndarray,
                           robot_s,
                           hand_name,
                           num_of_workers=3):
    graspid_and_graspinfo_list = list(graspid_and_graspinfo_list)
    interval = int(len(graspid_and_graspinfo_list) / num_of_workers)
    with mlp.Pool(processes=num_of_workers) as pool:
        multiple_results = [pool.apply_async(_get_available_graspids_ik, args=(
            graspid_and_graspinfo_list[i:i + interval],
            goal_pos,
            goal_rotmat,
            robot_s,
            hand_name)) for i in range(0, len(graspid_and_graspinfo_list), interval)]
        previously_available_graspids = [res.get() for res in multiple_results]

    return list(itertools.chain(*previously_available_graspids))

