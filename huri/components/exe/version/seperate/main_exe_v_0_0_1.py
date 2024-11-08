__VERSION__ = "0.0.1_seperate"

import time

from huri.core.common_import import wd, ym, fs, np
from huri.core.constants import SENSOR_INFO
from huri.components.exe.utils import gen_camera_obs, capture_vision_info, init_real_rbt, is_restart_planning
from huri.definitions.rack_def import TubeRack
from huri.core.print_tool import text_pd, print_with_border
from huri.definitions.utils_structure import MotionElement, MotionBatch, MotionBatchPPP
# motion planning
from huri.components.planning.common_grasp_seq import CommonGraspSolver
import huri.components.planning.symmetric_pick_and_place_planner_c as ppp
# vision system
from huri.components.vision.tube_detector import TestTubeDetector
# task planning
from huri.learning.env.arrangement_planning_rack.utils import isdone
from huri.components.task_planning.tube_puzzle_learning_solver import DQNSolver
# robot execution
import huri.components.yumi_control.yumi_con as yc
# error feedback
from huri.components.exe.executer import (feedback_insertion_error,
                                          INSERT_FAILURE_ERR,
                                          GRASP_COLLISION_ERR,
                                          INSERT_FAILURE_TUBE_LOSING_ERR,
                                          GRASP_NO_TUBE_ERR,
                                          UNKNOWN_ERR)
from vision_sys import vision_system
from task_sys import task_planning, TubeType, S_GOAL_PATTERN
from huri.components.exe.constants import APPROACH_DISTANCE_LIST, DEPART_DISTANCE_LIST

RBT_END_CONF = np.array([-0.0137881, -0.97703532, -1.50848807, 0.87929688, -1.99840199,
                         0.13788101, 1.51669112])

DEBUG_CACHE = []

ANIMATION_SAVE_PATH = fs.Path("D:\chen\huri_shared\huri\components\exe\\version\\animation.pkl")


def execute():
    pass


def comb_rack_status(rack_1, rack_2, rack_3, rack_4):
    return np.vstack([_.rack_status for _ in [rack_1, rack_2, rack_3, rack_4]])


def main(open_jaw_width=.034,
         depth_sensor_debug_path=None,
         debug=False,
         retry_num=3,  # number of retry when grasping failed
         goal_place_offset_dis=.04,
         is_vision_feedback=True,
         toggle_save=False, ):
    # Simulation Environment
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

    # init the simulation robot
    yumi_robot = ym.Yumi(enable_cc=True)
    component_name = "rgt_hnd"
    yumi_robot_arm = yumi_robot.rgt_arm if "rgt" in component_name else yumi_robot.lft_arm

    # init the real robot
    yumi_con = init_real_rbt(gripper_len=open_jaw_width, toggle_debug=debug, use_motion_exe=True)
    if not debug:
        yc.to_homeconf(yumi_s=yumi_robot, yumi_x=yumi_con, component_name="both")

    # init the pick and place planner
    common_grasp_solver = CommonGraspSolver(num_workers=10)
    pp_planner = ppp.PickPlacePlanner(robot_s=yumi_robot, common_grasp_solver=common_grasp_solver)

    # setup the collision model for phoxi camera
    camera_obs = gen_camera_obs()
    camera_obs.attach_to(base)

    # vision system
    detector = TestTubeDetector(affine_mat_path=SENSOR_INFO.PNT_CLD_CALIBR_MAT_PATH, )
    (tube_rack_1, tube_rack_2, tube_rack_3, tube_rack_4), _ = vision_system(
        detector, debug_filepath=depth_sensor_debug_path,
        toggle_obb=False,
        toggle_save=toggle_save)

    # init task planning solver
    task_solver = DQNSolver()

    ## Motion planning init
    motion_batch = MotionBatch()
    grasp_cache_dict = {}  # restore grasps
    infeasible_info_dict = {}  # restore infeasible action during task planning
    infeasible_local_pattern = {}
    rack_state_history = []
    bad_status_weight_matrix = {}

    tube_rack_state = comb_rack_status(rack_1=tube_rack_1,
                                       rack_2=tube_rack_2,
                                       rack_3=tube_rack_3,
                                       rack_4=tube_rack_4)
    # init
    animation_cnt = 0

    while not isdone(tube_rack_state, S_GOAL_PATTERN):
        task_sols = task_planning(tube_rack_1=tube_rack_1,
                                  tube_rack_2=tube_rack_2,
                                  tube_rack_3=tube_rack_3,
                                  tube_rack_4=tube_rack_4,
                                  infeasible_dict=infeasible_info_dict, )

        # debug_data = {"name": "task_planning",
        #               "tube_rack_state": tube_rack.rack_status,
        #               "task_planning_solutions": task_sols.solutions}
        # DEBUG_CACHE.append(debug_data)

        sol_length = len(task_sols)
        for sol_id, sol in enumerate(task_sols):
            print(f"------- Remaining unsolved path length is {sol_length - sol_id - 1} -------")
            print_with_border("Rack State matrix", width=38)
            print(text_pd(tube_rack_state))
            # generate and execute the pick and place motions
            common_grasp_iter_info = None
            is_replanning_flag = False
            rack_state_history.append(tube_rack_state.copy())
            # moved tube
            moved_tube = sol.tube
            moved_tube_cm = moved_tube.gen_collision_model()
            # get grasp of moved tube
            if grasp_cache_dict.get(moved_tube.type, None) is None:
                grasp_cache_dict[moved_tube.type] = fs.load_json(path=moved_tube.grasps_path)
            grasp_info_list = grasp_cache_dict[moved_tube.type]

            # collision model of the moved tube
            init_slot_id = sol.init_slot_id
            goal_slot_id = sol.goal_slot_id

            # get the start joint configuration of the robot
            start_conf = motion_batch.get_last_motion_rbt_jnt(component_name=component_name)
            if start_conf is None:
                start_conf = yumi_robot_arm.homeconf

            print_with_border(f"Moving: the test tube is {moved_tube.type}"
                              f"            init coordinate is {init_slot_id}"
                              f"              goal coordinate is {goal_slot_id}"
                              f"              Number of grasps: {len(grasp_info_list)}", width=38)

            for _ in range(retry_num):
                animation_cnt += 1
                is_retry = False
                # test tubes' collision model
                obs_testubes_cmlist = sol.obs_tube_cm_collection.cm_list

                obj_init_homomat = sol.init_homomat.copy()
                obj_goal_homomat = sol.goal_homomat.copy()
                obj_goal_homomat[:3, 3] = obj_goal_homomat[:3, 3] + sol.rack_tf_from[:3, 2] * goal_place_offset_dis
                homomat_list = [obj_init_homomat, obj_goal_homomat]

                tube_rack_obs_cm = tube_rack_1.gen_collision_model()
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
                                                         depart_direction_list=[sol.rack_tf_from[:3, 2],
                                                                                sol.rack_tf_to[:3, 2]],
                                                         approach_direction_list=[-sol.rack_tf_from[:3, 2],
                                                                                  -sol.rack_tf_to[:3, 2]],
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
                                                         )
                debug_data = {"name": "motion planning",
                              "data": [
                                  start2pickapproach_motion_seg, pickapproach_motion_seg, pickdepart_motion_seg,
                                  pickdepart2placeapproach_motion_seg, placeapproach_motion_seg, placedepart_motion_seg,
                                  placedepart2goal_motion_seg, common_grasp_iter_info, is_init_feasible
                              ]}
                DEBUG_CACHE.append(debug_data)
                # planning success
                if start2pickapproach_motion_seg is not None:
                    print("---" * 17)
                    print("Planning Successfully!")
                    # add motions into motion batch
                    exe_motion_batch = MotionBatchPPP(*[ppp.motion_seg_to_element(motion_seg,
                                                                                  component_name,
                                                                                  moved_tube_cm.copy(),
                                                                                  obs_testubes_cmlist, ) for motion_seg
                                                        in [start2pickapproach_motion_seg,
                                                            pickapproach_motion_seg,
                                                            pickdepart_motion_seg,
                                                            pickdepart2placeapproach_motion_seg,
                                                            placeapproach_motion_seg,
                                                            placedepart_motion_seg,
                                                            placedepart2goal_motion_seg]])
                    a = time.time()
                    animation_batch = MotionBatch()
                    for motion_seg in [start2pickapproach_motion_seg, pickapproach_motion_seg,
                                       pickdepart_motion_seg,
                                       pickdepart2placeapproach_motion_seg, placeapproach_motion_seg,
                                       placedepart_motion_seg, placedepart2goal_motion_seg]:
                        # for i in range(len(motion_seg.conf_ls)):
                        animation_batch.append(MotionElement(obj_cm=moved_tube_cm.copy(),
                                                             objpose_list=motion_seg.objpose_ls,
                                                             conf_list=motion_seg.conf_ls,
                                                             jawwidth_list=motion_seg.jaww_ls,
                                                             hnd_name=component_name,
                                                             obs_list=obs_testubes_cmlist + [tube_rack_obs_cm]),
                                               ele_name=motion_seg.name)
                    fs.dump_pickle([animation_cnt, animation_batch],
                                   path=ANIMATION_SAVE_PATH,
                                   reminder=False)
                    b = time.time()
                    print("Time to dump animations batch is:", b - a)

                    if debug:
                        pass
                    # execution
                    else:
                        is_exec_succ, exe_code = yumi_con.exe_pick_and_place_motion(motion_batch=exe_motion_batch,
                                                                                    speed_n_place=100)
                        # execution failed
                        if not is_exec_succ:
                            print((f"Execution Failed! Execution error code {exe_code}"))
                            # TODO: Revised
                            if exe_code == GRASP_COLLISION_ERR or exe_code == GRASP_NO_TUBE_ERR:
                                tube_rack_t, rack_tf_t = vision_system(detector=detector,
                                                                       debug_filepath=depth_sensor_debug_path,
                                                                       toggle_save=toggle_save)
                                grasp_info_list = [grasp_info_list[v[0]] for v in common_grasp_iter_info[1:]]
                                task_sols.update_rack(*tube_rack_t)
                                tube_rack_1, tube_rack_2, tube_rack_3, tube_rack_4 = tube_rack_t
                                tube_rack_state = comb_rack_status(*tube_rack_t)
                                sol = task_sols.current

                                if isdone(node=tube_rack_state, goalpattern=S_GOAL_PATTERN):
                                    print("Execute the motion successfully!")
                                    exit(0)
                                # retry
                                is_retry = True
                        # execute successfully
                        else:
                            # insert into new slot
                            # tube rack operation
                            rack_from = [tube_rack_1, tube_rack_2, tube_rack_3, tube_rack_4][sol.rack_from_id]
                            rack_to = [tube_rack_1, tube_rack_2, tube_rack_3, tube_rack_4][sol.rack_to_id]
                            init_slot_id_new = init_slot_id.copy()
                            init_slot_id_new[0] = init_slot_id[0] % 5
                            rack_from.remove_slot(init_slot_id_new)
                            goal_slot_id_new = goal_slot_id.copy()
                            goal_slot_id_new[0] = goal_slot_id[0] % 5
                            rack_to.insert_tube(goal_slot_id_new, moved_tube)
                            tube_rack_state = comb_rack_status(tube_rack_1, tube_rack_2, tube_rack_3, tube_rack_4)

                            # for animation
                            for motion_seg in [start2pickapproach_motion_seg, pickapproach_motion_seg,
                                               pickdepart_motion_seg,
                                               pickdepart2placeapproach_motion_seg, placeapproach_motion_seg,
                                               placedepart_motion_seg, placedepart2goal_motion_seg]:
                                # for i in range(len(motion_seg.conf_ls)):
                                motion_batch.append(MotionElement(obj_cm=moved_tube_cm.copy(),
                                                                  objpose_list=motion_seg.objpose_ls,
                                                                  conf_list=motion_seg.conf_ls,
                                                                  jawwidth_list=motion_seg.jaww_ls,
                                                                  hnd_name=component_name,
                                                                  obs_list=obs_testubes_cmlist, ),
                                                    ele_name=motion_seg.name)
                            if is_vision_feedback:
                                print("Execution successfully")
                                tube_rack_t, rack_tf_t = vision_system(detector=detector,
                                                                       debug_filepath=depth_sensor_debug_path,
                                                                       toggle_save=toggle_save)
                                tube_rack_state_t = comb_rack_status(*tube_rack_t)
                                if isdone(node=tube_rack_state_t, goalpattern=S_GOAL_PATTERN):
                                    # TODO a better way to finish program
                                    print("Finished")
                                    fs.dump_pickle(DEBUG_CACHE, f"{__VERSION__}_{time.strftime('%Y%m%d-%H%M%S')}.pkl")
                                    exit(0)
                                # check if the motion exec successfully
                                if np.array_equal(tube_rack_state_t, tube_rack_state):
                                    print("Execute the motion successfully!")
                                    # if is_restart_planning(rack_tf_t, rack_tf, toggle_debug=True):
                                    #     # TODO will plans the new motion in the same time
                                    #     print("NEED REPLANING")
                                    #     task_sols.update_rack(tube_rack_t)
                                    #     tube_rack = tube_rack_t
                                    #     rack_tf = rack_tf_t
                                    # else:
                                    #     print("NO NEED REPLANNING")
                                    task_sols.update_rack(*tube_rack_t)
                                    tube_rack_1, tube_rack_2, tube_rack_3, tube_rack_4 = tube_rack_t
                                    tube_rack_state = tube_rack_state_t
                                else:
                                    # check if the vision system works or some test tube is out of the rack
                                    # raise Exception("TT")
                                    # when the new detected tube number are larger than the old tube number,
                                    #
                                    if len(np.where(tube_rack_state_t > 0)[0]) < len(
                                            np.where(tube_rack_state > 0)[0]):
                                        #     # TODO add a help function
                                        #     input("Need helps")
                                        print("REPLANNING NEEDED")
                                    # task_sols.update_rack(tube_rack_t)
                                    tube_rack_state = tube_rack_state_t
                                    tube_rack_1, tube_rack_2, tube_rack_3, tube_rack_4 = tube_rack_t
                                    # # restart
                                    is_replanning_flag = True
                                    # break
                # planning failed
                else:
                    print("---" * 17)
                    print("Planning Failed!! Restart Search a New Path")
                    # planning failed
                    current_state = tube_rack_state
                    next_state = current_state.copy()
                    next_state[tuple(init_slot_id)], next_state[tuple(goal_slot_id)] = next_state[tuple(goal_slot_id)], \
                                                                                       next_state[tuple(init_slot_id)]
                    # weight_matrix = bad_status_weight_matrix.get(str(current_state),
                    #                                              np.zeros_like(tube_rack.rack_status)).copy()
                    # weight_matrix[tuple(init_slot_id)] = 1
                    # weight_matrix[tuple(goal_slot_id)] = 1
                    # bad_status_weight_matrix[str(current_state)] = weight_matrix
                    # print failed reason
                    if not is_init_feasible:
                        print("Infeasible Reason: No grasp at init position")
                    print(common_grasp_iter_info)
                    # add infeasible state to dict
                    infeasible_actions = infeasible_info_dict.get(str(current_state), [])
                    is_remove_all_init_slot_id_actions = not is_init_feasible
                    infeasible_actions.extend(task_solver.action_between_states(current_state,
                                                                                next_state,
                                                                                remove_action_group=is_remove_all_init_slot_id_actions))
                    print(rack_state_history)
                    for prev_state in rack_state_history:
                        diff = current_state - prev_state
                        if len(diff[diff != 0]) != 2:  # only one step different
                            continue
                        infeasible_actions.extend(task_solver.action_between_states(current_state, prev_state))

                    infeasible_info_dict[str(current_state)] = list(set(infeasible_actions))

                    # infeaible local pattern
                    if not is_init_feasible:
                        # TODO bug still there
                        infeasible_local = infeasible_local_pattern.get(tuple(init_slot_id), {})
                        local_info = str(np.pad(current_state, 1)[init_slot_id[0]:init_slot_id[0] + 3,
                                         init_slot_id[1]:init_slot_id[1] + 3])
                        local_infeasible_actions = infeasible_local.get(local_info, [])
                        local_infeasible_actions.extend(task_solver.action_between_states(current_state,
                                                                                          next_state,
                                                                                          remove_action_group=is_remove_all_init_slot_id_actions))
                        infeasible_local[local_info] = local_infeasible_actions
                        infeasible_local_pattern[tuple(init_slot_id)] = infeasible_local

                    # print("Weight matrix of the state is ...\n", weight_matrix)
                    print("Infeasible action of the state is ...", infeasible_actions)
                    print(f"Remove all the action that pick {init_slot_id} ...", is_remove_all_init_slot_id_actions)
                    is_replanning_flag = True
                if not is_retry:
                    break

            # planning failed or execution failed
            if is_replanning_flag:
                break
    print_with_border("All test tubes have already been arranged", width=70)
    common_grasp_solver.stop()
    if debug:
        pass
    else:
        yumi_con.stop()


if __name__ == '__main__':
    main(open_jaw_width=.034,
         debug=False,
         depth_sensor_debug_path=None,
         retry_num=3,
         goal_place_offset_dis=.04,
         is_vision_feedback=True,
         toggle_save=False)
