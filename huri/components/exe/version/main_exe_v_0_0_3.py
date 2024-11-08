""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20231130osaka

"""

__VERSION__ = "0.0.2_debug"

import time

from huri.core.common_import import wd, ym, fs, np
from huri.core.constants import SENSOR_INFO
from huri.components.exe.utils import gen_camera_obs, capture_vision_info, init_real_rbt, change_led_power, \
    vision_read_data
from huri.definitions.rack_def import TubeRack
from huri.core.print_tool import text_pd, print_with_border
from huri.definitions.utils_structure import MotionElement, MotionBatch, MotionBatchPPP
# motion planning
from huri.components.planning.common_grasp_seq import CommonGraspSolver, CommonGraspSolverDebug
import huri.components.planning.symmetric_pick_and_place_planner_c as ppp
# vision system
from huri.definitions.rack_def import ArrangeTubeSolSeq, Rack_Hard_Proto
from huri.components.vision.tube_detector import TestTubeDetector
# task planning
from huri.learning.env.arrangement_planning_rack.utils import isdone
from huri.components.task_planning.tube_puzzle_learning_solver import D3QNSolver
# robot execution
import huri.components.yumi_control.yumi_con as yc
# error feedback
from huri.components.exe.executer import (feedback_insertion_error,
                                          INSERT_FAILURE_ERR,
                                          GRASP_COLLISION_ERR,
                                          INSERT_FAILURE_TUBE_LOSING_ERR,
                                          GRASP_NO_TUBE_ERR,
                                          UNKNOWN_ERR)
from huri.components.exe.constants import APPROACH_DISTANCE_LIST, DEPART_DISTANCE_LIST, TABLE_HEIGHT, RACK_HEIGHT
from huri.components.exe.utils import create_directory
from huri.learning.method.AlphaZero.log import CsvWriter, write_to_csv
from huri.learning.method.AlphaZero.utils import get_time_stamp, delete_all_files_in_directory, is_folder_empty
import uuid
from huri.components.exe.version.grasp_filter import grasp_filter, get_feasible_ranges_rad
from huri.components.planning.symmetric_pick_and_place_palnner_c_fund import PickPlacePlanner_CR
import modeling.collision_model as cm

# GOAL_PATTERN = np.array([[1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ],
#                          [1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ],
#                          [1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ],
#                          [1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ],
#                          [1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ]])

GOAL_PATTERN = np.array([[1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
                         [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
                         [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
                         [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
                         [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ]])
# GOAL_PATTERN = np.array([[1, 1, 1, 1, 0, 0, 2, 2, 2, 2],
#                          [1, 1, 1, 1, 0, 0, 2, 2, 2, 2],
#                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                          [3, 3, 3, 3, 0, 0, 4, 4, 4, 4],
#                          [3, 3, 3, 3, 0, 0, 4, 4, 4, 4]])

ITERATION_NUM = [0]
MOTION_PLANNING_SOL_LOCAL_COUNTER = [0]
MOTION_PLANNING_LOCAL_COUNTER = [0]
VISION_SYSTEM_LOCAL_COUNTER = [0]
INSERTION_FEEDBACK_COUNTER = [0]
WORK_DIRECTORY = create_directory('./run')
if not is_folder_empty(WORK_DIRECTORY):
    delete_all_files_in_directory(WORK_DIRECTORY)
WORK_DIRECTORY_VISION = create_directory('./run/vision')
WORK_DIRECTORY_TP = create_directory('./run/taskplanning')
WORK_DIRECTORY_MP = create_directory('./run/motionplanning')

LOGGER = CsvWriter(str(WORK_DIRECTORY.joinpath("running_log.csv")))

# write_to_csv(LOGGER, [
#     ('timestamp', get_time_stamp(), '%1s'),
#     ('step', ITERATION_NUM[0], '%1d'),
#     ('time_consumption', end_t - start_t, '%1s'),
#     ('module_name', module_name, '%1s'),
#     ('dump_file_name', dump_file_name, '%1s'),
# ])

RBT_END_CONF = np.array([-0.0137881, -0.97703532, -1.50848807, 0.87929688, -1.99840199,
                         0.13788101, 1.51669112])

ANIMATION_SAVE_PATH = fs.Path(r"animation.pkl")
DEBUG_PROGRAM = False


def task_planning(tube_rack: TubeRack, solver: D3QNSolver, condition_set, infeasible_action_dict) -> ArrangeTubeSolSeq:
    ##### FOR LOGGING
    st = time.time()
    save_path = WORK_DIRECTORY_TP.joinpath(f'TASK_PLANNING_DATA_ITER_{ITERATION_NUM[0]}.pkl')
    ###############
    raw_solution, solution = solver.solve(tube_rack.rack_status,
                                          condition_set=condition_set,
                                          infeasible_action_dict=infeasible_action_dict,
                                          max_iter=800,
                                          toggle_show=False)
    task_sols = ArrangeTubeSolSeq(rack=tube_rack,
                                  solution=solution,
                                  tf=tube_rack.get_homomat())
    ##### FOR LOGGING
    # et = time.time()
    # write_to_csv(LOGGER, [
    #     ('timestamp', get_time_stamp(), '%1s'),
    #     ('step', ITERATION_NUM[0], '%1d'),
    #     ('time_consumption', et - st, '%1s'),
    #     ('module_name', "TASK_PLANNING", '%1s'),
    #     ('dump_file_name', str(save_path), '%1s'),
    # ])
    # fs.dump_pickle({'raw_solution': raw_solution,
    #                 'solution': solution,
    #                 'task_sols': task_sols,
    #                 'tube_rack': tube_rack,
    #                 'condition_set': condition_set,
    #                 'infeasible_action_dict': infeasible_action_dict},
    #                save_path,
    #                reminder=False)
    ######################
    return task_sols


def vision_system(detector: TestTubeDetector,
                  toggle_yolo=False,
                  toggle_save=False,
                  debug_filepath=None,
                  debug_program=DEBUG_PROGRAM) -> (TubeRack, np.ndarray):
    toggle_debug = True if debug_filepath is not None else False
    ##### FOR LOGGING
    st = time.time()
    save_path = WORK_DIRECTORY_VISION.joinpath(
        f'VISION_DATA_ITER_{ITERATION_NUM[0]}_{VISION_SYSTEM_LOCAL_COUNTER[0]}.pkl')
    #####
    # capture the data through vision sensor
    if debug_program:
        filename = fs.workdir / "data" / "vision_exp" / "20231130-163116.pkl"
        pcd, img, depth_img, _, _ = vision_read_data(filename)
    else:
        pcd, img, depth_img = capture_vision_info(ip_adr=SENSOR_INFO.IP_ADR_DEPTH_SENSOR_DEBUG,
                                                  debug_filename=debug_filepath,
                                                  save_path=save_path,
                                                  toggle_save=False,
                                                  toggle_debug=toggle_debug, )  # TODO revise debug back to DEBUG
    detected_test_tubes, tube_rack, rack_tf, yolo_img = detector.analyze_scene(rack_proto=Rack_Hard_Proto,
                                                                               pcd=pcd,
                                                                               texture_img=img[:, :, 0],
                                                                               std_out=None,
                                                                               toggle_detect_tube_pos=True,
                                                                               toggle_yolo=True,
                                                                               save_detect=toggle_save)
    ##### FOR LOGGING
    # et = time.time()
    # write_to_csv(LOGGER, [
    #     ('timestamp', get_time_stamp(), '%1s'),
    #     ('step', ITERATION_NUM[0], '%1d'),
    #     ('time_consumption', et - st, '%1s'),
    #     ('module_name', f"VISION_ITER_{ITERATION_NUM[0]}_{VISION_SYSTEM_LOCAL_COUNTER}", '%1s'),
    #     ('dump_file_name', str(save_path), '%1s'),
    # ])
    # fs.dump_pickle({'raw_data': [pcd, img, depth_img],
    #                 'detected_results': [detected_test_tubes, tube_rack, rack_tf, yolo_img]},
    #                save_path,
    #                reminder=False)
    #############
    fs.dump_pickle([uuid.uuid4().hex, pcd, img], r"vision_data.pkl",
                   reminder=False)

    return tube_rack, rack_tf


def execute():
    pass


def main(open_jaw_width=.034,
         depth_sensor_debug_path=None,
         debug=False,
         retry_num=3,  # number of retry when grasping failed
         goal_place_offset_dis=.021,
         is_vision_feedback=True,
         toggle_save=False,
         exe_speed=200,
         rrt_ext_dis=.008,
         VISION_FEEDBACK_INTERVAL=3, ):
    # init the simulation robot
    yumi_robot = ym.Yumi(enable_cc=True)
    # init the pick and place planner
    num_workers = 6
    if not DEBUG_PROGRAM:
        common_grasp_solver = CommonGraspSolver(num_workers=num_workers)
    else:
        common_grasp_solver = CommonGraspSolverDebug()
    pp_planner = ppp.PickPlacePlanner(robot_s=yumi_robot, common_grasp_solver=common_grasp_solver)

    # Simulation Environment
    PickPlacePlanner_CR.RRT_EXT_DIS = rrt_ext_dis

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

    collision_box_for_rrt = cm.gen_box(extent=[5, 5, .12], homomat=np.eye(4))
    collision_box_for_rrt.set_pos(np.array([0, 0, .06]))
    collision_box_for_rrt.attach_to(base)

    change_led_power(ip_adr=SENSOR_INFO.IP_ADR_DEPTH_SENSOR_DEBUG, led_power=2000)
    component_name = "rgt_hnd"
    yumi_robot_arm = yumi_robot.rgt_arm if "rgt" in component_name else yumi_robot.lft_arm

    # init the real robot
    yumi_con = init_real_rbt(gripper_len=open_jaw_width, toggle_debug=debug, use_motion_exe=True)
    if not debug:
        yc.to_homeconf(yumi_s=yumi_robot, yumi_x=yumi_con, component_name="both")
        # yc.move_rrt(yumi_s=yumi_robot, yumi_x=yumi_con, goal_conf=RBT_END_CONF,
        #             component_name=component_name)
        yumi_robot.fk('rgt_arm', yumi_con.get_jnt_values('rgt_arm'))
        yumi_robot.fk('lft_arm', yumi_con.get_jnt_values('lft_arm'))

        # setup the collision model for phoxi camera
    camera_obs = gen_camera_obs()
    camera_obs.attach_to(base)

    # vision system
    detector = TestTubeDetector(affine_mat_path=SENSOR_INFO.PNT_CLD_CALIBR_MAT_PATH,
                                rack_height_lower=TABLE_HEIGHT + RACK_HEIGHT - .005,
                                rack_height_upper=TABLE_HEIGHT + RACK_HEIGHT + .005, )
    tube_rack, rack_tf = vision_system(detector, debug_filepath=depth_sensor_debug_path,
                                       toggle_save=toggle_save)

    # init task planning solver
    task_solver = D3QNSolver(goal_pattern=GOAL_PATTERN)

    ## Motion planning init
    motion_batch = MotionBatch()
    grasp_cache_dict = {}  # restore grasps
    infeasible_info_dict = {}  # restore infeasible action during task planning
    infeasible_local_pattern = {}
    rack_state_history = []
    bad_status_weight_matrix = {}

    condition_set = np.ones(([*GOAL_PATTERN.shape[:2], 6]), dtype=int)
    infeasible_action_dict = {}
    # init
    animation_cnt = 0
    need_vision_feedback = False
    need_grasp_vision_feedback = False
    need_vision_counter = 0

    while not isdone(node=tube_rack.rack_status, goalpattern=GOAL_PATTERN):
        print(len(np.where(tube_rack.rack_status == 1)[0]))
        print(len(np.where(tube_rack.rack_status == 2)[0]))
        print(len(np.where(tube_rack.rack_status == 3)[0]))
        ####### FOR LOGGING
        # ITERATION_NUM[0] += 1
        # VISION_SYSTEM_LOCAL_COUNTER[0] = 0
        # MOTION_PLANNING_SOL_LOCAL_COUNTER[0] = 0
        #######
        if not task_solver.is_state_valid(tube_rack.rack_status):
            raise Exception("The state contains unsolvable patterns!")

        # recalculate actions:
        # infeasible_actions = []
        # for _ in np.argwhere(np.sum(slot_maps, axis=2) != 6):
        #     if np.sum(
        #             task_solver.get_slot_satisfied_constraints(tube_rack.rack_status, _) * slot_maps[_[0], _[1]]) == 0:
        #         infeasible_actions += task_solver.get_all_infeasible_actions_coord(_)
        task_sols = task_planning(tube_rack=tube_rack,
                                  solver=task_solver,
                                  condition_set=condition_set,
                                  infeasible_action_dict=infeasible_action_dict)

        sol_length = len(task_sols)
        for sol_id, sol in enumerate(task_sols):
            need_vision_counter += 1
            if need_vision_counter % VISION_FEEDBACK_INTERVAL == 0:
                need_vision_feedback = True

            ####### FOR LOGGING
            # MOTION_PLANNING_SOL_LOCAL_COUNTER[0] += 1
            #####################
            print(f"------- Remaining unsolved path length is {sol_length - sol_id - 1} -------")
            print_with_border("Rack State matrix", width=38)
            print(text_pd(tube_rack.rack_status))
            # generate and execute the pick and place motions
            common_grasp_iter_info = None
            is_replanning_flag = False
            rack_state_history.append(tube_rack.rack_status.copy())
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
            # start_conf = motion_batch.get_last_motion_rbt_jnt(component_name=component_name)

            print_with_border(f"Moving: the test tube is {moved_tube.type}"
                              f"            init coordinate is {init_slot_id}"
                              f"              goal coordinate is {goal_slot_id}"
                              f"              Number of grasps: {len(grasp_info_list)}", width=38)
            ### FOR LOGGING
            # MOTION_PLANNING_LOCAL_COUNTER[0] = 0
            ###################
            for _ in range(retry_num):
                start_conf = yumi_con.get_jnt_values(component_name)
                if start_conf is None:
                    start_conf = yumi_robot_arm.homeconf
                ### FOR LOGGING
                # MOTION_PLANNING_LOCAL_COUNTER[0] += 1
                # save_path = WORK_DIRECTORY_MP.joinpath(f'MOTION_PLANNING_DATA_ITER_{ITERATION_NUM[0]}'
                #                                        f'_{MOTION_PLANNING_SOL_LOCAL_COUNTER[0]}_{MOTION_PLANNING_LOCAL_COUNTER[0]}.pkl')
                # st = time.time()
                ####################
                animation_cnt += 1
                is_retry = False
                # test tubes' collision model
                obs_testubes_cmlist = sol.obs_tube_cm_collection.cm_list

                obj_init_homomat = sol.init_homomat.copy()
                obj_goal_homomat = sol.goal_homomat.copy()
                obj_goal_homomat[:3, 3] = obj_goal_homomat[:3, 3] + rack_tf[:3, 2] * goal_place_offset_dis
                homomat_list = [obj_init_homomat, obj_goal_homomat]

                insert_direction = sol.insert_direction
                grasp_direction = sol.grasp_direction
                # up distance
                tube_rack_obs_cm = tube_rack.gen_collision_model()
                current_state = tube_rack.rack_status.copy()
                grasp_info_list = grasp_filter(current_state, init_slot_id, tube_rack, grasp_info_list)
                feasible_ranges_rad = get_feasible_ranges_rad(sol.goal_rack_status, goal_slot_id)
                # planning the motion
                print("Need vision feedback:", need_grasp_vision_feedback,
                      (need_vision_feedback and is_vision_feedback))
                start2pickapproach_motion_seg, pickapproach_motion_seg, pickdepart_motion_seg, \
                    pickdepart2placeapproach_motion_seg, placeapproach_motion_seg, placedepart_motion_seg, \
                    placedepart2goal_motion_seg, common_grasp_iter_info, is_pick_feasible, is_place_feasible = \
                    pp_planner.gen_pick_and_place_motion(hnd_name=component_name,
                                                         tube=moved_tube,
                                                         tube_cm=moved_tube_cm,
                                                         grasp_info_list=grasp_info_list,
                                                         goal_homomat_list=homomat_list,
                                                         start_conf=start_conf,
                                                         end_conf=RBT_END_CONF if (
                                                                 need_vision_feedback and is_vision_feedback) else None,
                                                         # end_conf=None,
                                                         depart_direction_list=[rack_tf[:3, 2]] * len(
                                                             homomat_list),
                                                         approach_direction_list=[grasp_direction, insert_direction],
                                                         depart_distance_list=DEPART_DISTANCE_LIST,
                                                         approach_distance_list=APPROACH_DISTANCE_LIST,
                                                         common_grasp_iter=None,
                                                         approach_jawwidth=None,
                                                         depart_jawwidth=None,
                                                         # ad_granularity=.005,
                                                         ad_granularity=.007,
                                                         use_rrt=False,
                                                         obstacle_list=obs_testubes_cmlist + [camera_obs],
                                                         grasp_obstacle_list=[tube_rack_obs_cm],
                                                         use_incremental=False,
                                                         feasible_ranges_rad=feasible_ranges_rad,
                                                         rack_tf=rack_tf,
                                                         rrt_obstacle_list=[collision_box_for_rrt],
                                                         # toggle_debug=True,
                                                         # logger=exe_logger
                                                         )
                need_grasp_vision_feedback = False
                et = time.time()
                motion_data = {"name": "motion planning",
                               "data": [
                                   start2pickapproach_motion_seg, pickapproach_motion_seg, pickdepart_motion_seg,
                                   pickdepart2placeapproach_motion_seg, placeapproach_motion_seg,
                                   placedepart_motion_seg,
                                   placedepart2goal_motion_seg, common_grasp_iter_info, is_pick_feasible,
                                   is_place_feasible
                               ]}
                # planning success
                exe_motion_batch = None
                animation_batch = None
                ### FOR LOGGING
                # write_to_csv(LOGGER, [
                #     ('timestamp', get_time_stamp(), '%1s'),
                #     ('step', ITERATION_NUM[0], '%1d'),
                #     ('time_consumption', et - st, '%1s'),
                #     ('module_name', f"MOTION_PLANNING"
                #                     f"_{MOTION_PLANNING_SOL_LOCAL_COUNTER[0]}"
                #                     f"_{MOTION_PLANNING_LOCAL_COUNTER[0]}", '%1s'),
                #     ('dump_file_name', str(save_path), '%1s'),
                # ])
                # fs.dump_pickle({'sol': sol,
                #                 'motion_data': motion_data,
                #                 'tube_rack': tube_rack,
                #                 'exe_motion_batch': exe_motion_batch,
                #                 'animation_batch': animation_batch, },
                #                save_path,
                #                reminder=False)
                ###################
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
                                                            placedepart2goal_motion_seg] if
                                                        motion_seg.conf_ls is not None])
                    # a = time.time()
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
                    # b = time.time()
                    # print("Time to dump animations batch is:", b - a)

                    if debug:
                        pass
                    # execution
                    else:
                        is_exec_succ, exe_code = yumi_con.exe_pick_and_place_motion(motion_batch=exe_motion_batch,
                                                                                    speed_n_place=exe_speed, )
                        # execution failed
                        if not is_exec_succ:
                            # yumi_con.open_gripper(component_name)
                            # yc.to_homeconf(yumi_s=yumi_robot, yumi_x=yumi_con, component_name=component_name)
                            print((f"Execution Failed! Execution error code {exe_code}"))
                            if exe_code == GRASP_COLLISION_ERR or exe_code == GRASP_NO_TUBE_ERR:
                                yc.move_rrt(yumi_s=yumi_robot, yumi_x=yumi_con, goal_conf=RBT_END_CONF,
                                            component_name=component_name, obstacle_list=obs_testubes_cmlist,
                                            speed_n=exe_speed)
                                VISION_SYSTEM_LOCAL_COUNTER[0] += 1
                                tube_rack_t, rack_tf_t = vision_system(detector=detector,
                                                                       debug_filepath=depth_sensor_debug_path,
                                                                       toggle_save=toggle_save)
                                need_vision_counter = 0
                                need_grasp_vision_feedback = True
                                grasp_info_list = [grasp_info_list[v[0]] for v in common_grasp_iter_info[1:]]
                                task_sols.update_rack(tube_rack_t)
                                tube_rack = tube_rack_t
                                rack_tf = rack_tf_t
                                if not np.array_equal(tube_rack_t.rack_status, tube_rack.rack_status):
                                    is_replanning_flag = True
                                    is_retry = False
                                else:
                                    sol = task_sols.current
                                    # retry
                                    is_retry = True
                                # TODO examine if the pattern is the same, otherwise go back to task planning
                                # is_replanning_flag =True
                            if not is_exec_succ and exe_code == INSERT_FAILURE_ERR:
                                is_recoverable, rrt_path_component_arm, to_recover_motion, approach_recover_motion = feedback_insertion_error(
                                    yumi_s=yumi_robot,
                                    yumi_con=yumi_con,
                                    tube=moved_tube.copy(),
                                    tube_pose=
                                    exe_motion_batch.place_depart.objpose_list[
                                        -1],
                                    pp_planner=pp_planner,
                                    obstacle_list=obs_testubes_cmlist + [
                                        camera_obs, ],
                                    toggle_return_motion=True)
                                if is_recoverable:
                                    INSERTION_FEEDBACK_COUNTER[0] += 1
                                    feedback_save_path = WORK_DIRECTORY_MP.joinpath(
                                        f'INSERTION_FEEDBACK_{ITERATION_NUM[0]}'
                                        f'_{INSERTION_FEEDBACK_COUNTER[0]}.pkl')
                                    ### FOR LOGGING
                                    # write_to_csv(LOGGER, [
                                    #     ('timestamp', get_time_stamp(), '%1s'),
                                    #     ('step', ITERATION_NUM[0], '%1d'),
                                    #     ('time_consumption', 0, '%1s'),
                                    #     ('module_name', f"INSERTION_FEEDBACK"
                                    #                     f"_{INSERTION_FEEDBACK_COUNTER[0]}", '%1s'),
                                    #     ('dump_file_name', str(feedback_save_path), '%1s'),
                                    # ])
                                    # fs.dump_pickle({'tube_pose': exe_motion_batch.place_depart.objpose_list[-1],
                                    #                 'exe_motion_batch': exe_motion_batch,
                                    #                 'recover_motions': [rrt_path_component_arm,
                                    #                                     to_recover_motion,
                                    #                                     approach_recover_motion]},
                                    #                feedback_save_path,
                                    #                reminder=False)
                                    #########################################
                                    yumi_con._place_contactL_torque = .03
                                    is_exec_succ, exe_code = yumi_con.exe_pick_and_place_motion(
                                        motion_batch=exe_motion_batch,
                                        speed_n_place=100,
                                        is_only_place=True)
                                    yumi_con._place_contactL_torque = .02
                                    if not is_exec_succ:
                                        raise Exception("ERRR ")
                                else:
                                    raise Exception("NOt recoverable")
                                is_retry = False
                                is_replanning_flag = False
                                tube_rack.remove_slot(init_slot_id)
                                tube_rack.insert_tube(goal_slot_id, moved_tube)
                        # execute successfully
                        else:
                            # tube rack operation
                            tube_rack.remove_slot(init_slot_id)
                            tube_rack.insert_tube(goal_slot_id, moved_tube)
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
                            if need_vision_feedback and is_vision_feedback:
                                print("Execution successfully")
                                ######### FOR LOGGINING
                                # VISION_SYSTEM_LOCAL_COUNTER[0] += 1
                                ########################
                                tube_rack_t, rack_tf_t = vision_system(detector=detector,
                                                                       debug_filepath=depth_sensor_debug_path,
                                                                       toggle_save=toggle_save)
                                need_vision_counter = 0
                                need_vision_feedback = False
                                if isdone(node=tube_rack_t.rack_status, goalpattern=GOAL_PATTERN):
                                    # TODO a better way to finish program
                                    print("Finished")
                                    # fs.dump_pickle(DEBUG_CACHE, f"{__VERSION__}_{time.strftime('%Y%m%d-%H%M%S')}.pkl")
                                    exit(0)
                                # check if the motion exec successfully
                                if np.array_equal(tube_rack_t.rack_status, tube_rack.rack_status):
                                    print("Execute the motion successfully!")
                                    # if is_restart_planning(rack_tf_t, rack_tf, toggle_debug=True):
                                    #     # TODO will plans the new motion in the same time
                                    #     print("NEED REPLANING")
                                    #     task_sols.update_rack(tube_rack_t)
                                    #     tube_rack = tube_rack_t
                                    #     rack_tf = rack_tf_t
                                    # else:
                                    #     print("NO NEED REPLANNING")
                                    task_solver.add_history_state(tube_rack.rack_status)
                                    task_sols.update_rack_pose(tube_rack_t)
                                    # tube_rack = tube_rack_t
                                    tube_rack.set_homomat(tube_rack_t.get_homomat())
                                    rack_tf = rack_tf_t
                                else:
                                    # check if the vision system works or some test tube is out of the rack
                                    # raise Exception("TT")
                                    # when the new detected tube number are larger than the old tube number,
                                    #
                                    if len(np.where(tube_rack_t.rack_status > 0)[0]) < len(
                                            np.where(tube_rack.rack_status > 0)[0]):
                                        #     # TODO add a help function
                                        #     input("Need helps")
                                        print("REPLANNING NEEDED")
                                    # task_sols.update_rack(tube_rack_t)
                                    tube_rack = tube_rack_t
                                    rack_tf = rack_tf_t
                                    # # restart
                                    is_replanning_flag = True
                                    # break
                # planning failed
                else:
                    print("---" * 17)
                    print("Planning Failed!! Restart Search a New Path")
                    # planning failed
                    current_state = tube_rack.rack_status.copy()
                    next_state = current_state.copy()
                    next_state[tuple(init_slot_id)], next_state[tuple(goal_slot_id)] = next_state[tuple(goal_slot_id)], \
                        next_state[tuple(init_slot_id)]
                    # weight_matrix = bad_status_weight_matrix.get(str(current_state),
                    #                                              np.zeros_like(tube_rack.rack_status)).copy()
                    # weight_matrix[tuple(init_slot_id)] = 1
                    # weight_matrix[tuple(goal_slot_id)] = 1
                    # bad_status_weight_matrix[str(current_state)] = weight_matrix
                    # print failed reason

                    # add infeasible state to dict
                    # infeasible_actions = infeasible_info_dict.get(str(current_state), [])

                    task_solver.cal_constraints(current_state,
                                                init_slot_id,
                                                goal_slot_id,
                                                is_pick_feasible,
                                                is_place_feasible,
                                                condition_set,
                                                infeasible_action_dict=infeasible_action_dict)
                    print("is pick feasible", is_pick_feasible, "init_slot_id", init_slot_id)
                    print("is place feasible", is_place_feasible, "goal_slot_id", goal_slot_id)
                    # recalculate infeasible_actions

                    # print(rack_state_history)
                    # for prev_state in rack_state_history:
                    #     diff = current_state - prev_state
                    #     if len(diff[diff != 0]) != 2:  # only one step different
                    #         continue
                    #     infeasible_actions.extend(task_solver.action_between_states(current_state, prev_state))
                    #
                    # infeasible_info_dict[str(current_state)] = list(set(infeasible_actions))
                    #
                    # # infeaible local pattern
                    # if not is_init_feasible:
                    #     # TODO bug still there
                    #     infeasible_local = infeasible_local_pattern.get(tuple(init_slot_id), {})
                    #     local_info = str(np.pad(current_state, 1)[init_slot_id[0]:init_slot_id[0] + 3,
                    #                      init_slot_id[1]:init_slot_id[1] + 3])
                    #     local_infeasible_actions = infeasible_local.get(local_info, [])
                    #     local_infeasible_actions.extend(task_solver.action_between_states(current_state,
                    #                                                                       next_state,
                    #                                                                       remove_action_group=is_remove_all_init_slot_id_actions))
                    #     infeasible_local[local_info] = local_infeasible_actions
                    #     infeasible_local_pattern[tuple(init_slot_id)] = infeasible_local

                    # print("Weight matrix of the state is ...\n", weight_matrix)
                    # print("Infeasible action of the state is ...", infeasible_actions)
                    print("Current SLot maps are", np.sum(condition_set, axis=2))
                    # print(f"Remove all the action that pick {init_slot_id} ...", is_remove_all_init_slot_id_actions)
                    is_replanning_flag = True

                if not is_retry:
                    break

            ### FOR LOGGING

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
    main(open_jaw_width=.031,
         debug=False,
         depth_sensor_debug_path=None,
         retry_num=5,
         goal_place_offset_dis=.025,
         is_vision_feedback=True,
         toggle_save=True,
         exe_speed=600,
         rrt_ext_dis=.025,
         VISION_FEEDBACK_INTERVAL=2)
