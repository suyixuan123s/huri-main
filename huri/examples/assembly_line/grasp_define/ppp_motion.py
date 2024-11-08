""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230906osaka

"""
from typing import Literal
import numpy as np


def gen_pick_place_motion(init_homomat: np.ndarray,
                          goal_homomat: np.ndarray,
                          tube_rack,
                          grasp_direction: np.ndarray = np.array([0, 0, -1]),
                          insert_direction: np.ndarray = np.array([0, 0, -1]),
                          component_name: Literal['lft_arm', 'rgt_arm'] = 'rgt_arm'):
    obj_init_homomat = init_homomat.copy()
    obj_goal_homomat = goal_homomat.copy()

    homomat_list = [obj_init_homomat, obj_goal_homomat]

    grasp_direction = grasp_direction
    insert_direction = insert_direction

    # up distance
    tube_rack_obs_cm = tube_rack.gen_collision_model()
    # planning the motion
    start2pickapproach_motion_seg, pickapproach_motion_seg, pickdepart_motion_seg, \
        pickdepart2placeapproach_motion_seg, placeapproach_motion_seg, placedepart_motion_seg, \
        placedepart2goal_motion_seg, common_grasp_iter_info, is_init_feasible = \
        pp_planner.gen_pick_and_place_motion(hnd_name=component_name,
                                             tube=moved_tube,
                                             tube_cm=moved_tube_cm,
                                             grasp_info_list=grasp_info_list,
                                             goal_homomat_list=homomat_list,
                                             start_conf=start_conf,
                                             end_conf=RBT_END_CONF,
                                             depart_direction_list=[rack_tf[:3, 2]] * len(
                                                 homomat_list),
                                             approach_direction_list=[grasp_direction, -rack_tf[:3, 2], ],
                                             depart_distance_list=DEPART_DISTANCE_LIST,
                                             approach_distance_list=APPROACH_DISTANCE_LIST,
                                             common_grasp_iter=None,
                                             approach_jawwidth=None,
                                             depart_jawwidth=None,
                                             # ad_granularity=.005,
                                             ad_granularity=.015,
                                             use_rrt=False,
                                             obstacle_list=obs_testubes_cmlist + [camera_obs],
                                             grasp_obstacle_list=[tube_rack_obs_cm],
                                             use_incremental=False,
                                             # logger=exe_logger
                                             toggle_time=True,
                                             )


if __name__ == '__main__':
    pass
