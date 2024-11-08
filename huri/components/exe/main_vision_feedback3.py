import time

from huri.components.exe.utils import capture_vision_info, init_real_rbt, is_restart_planning
from huri.core.common_import import *
from huri.definitions.utils_structure import MotionElement, MotionBatch, MotionBatchPPP
from huri.learning.env.arrangement_planning_rack.utils import isdone
from panda3d.core import loadPrcFileData
import motion.probabilistic.rrt_connect as rrtc

loadPrcFileData('', 'notify-level error')
from huri.test.app.animation import show_animation
from huri.components.vision.tube_detector import TestTubeDetector
import huri.components.planning.symmetric_pick_and_place_planner_c as ppp
from huri.components.pipeline.data_pipeline import motion_executer
from huri.definitions.rack_def import ArrangeTubeSolSeq
from huri.components.pipeline.data_pipeline import RenderController, MotionController
from huri.core.print_tool import text_pd, print_with_border
import basis.trimesh as tm
import visualization.panda.rpc.rviz_client as rv_client
from huri.components.task_planning.tube_puzzle_learning_solver import DQNSolver
from huri.test.app.remote_exe_client import Remote_Exe_Client
from huri.core.constants import SENSOR_INFO
from huri.components.planning.common_grasp_seq import CommonGraspSolver
from huri.components.exe.exe_loggining import exe_logger
from huri.components.exe.executer import (feedback_insertion_error,
                                          INSERT_FAILURE_ERR,
                                          GRASP_COLLISION_ERR,
                                          INSERT_FAILURE_TUBE_LOSING_ERR,
                                          GRASP_NO_TUBE_ERR,
                                          UNKNOWN_ERR)


def go_init_pose(yumi_s, yumi_x, component_name="rgt_arm", method="RRT", speed_n=300, ):
    if method.lower() == "rrt":
        # initialize the module for RRT
        rrtc_planner = rrtc.RRTConnect(yumi_s)
        # the left and right arm go initial pose
        if component_name in ["rgt_arm", "rgt_hnd", "both"]:
            rrt_path_rgt = rrtc_planner.plan(component_name="rgt_arm",
                                             start_conf=np.array(yumi_x.get_jnt_values("rgt_arm")),
                                             goal_conf=np.array(yumi_s.rgt_arm.homeconf),
                                             obstacle_list=[],
                                             ext_dist=.05,
                                             max_time=300)
            if len(rrt_path_rgt) > 5:
                yumi_x.move_jntspace_path(component_name="rgt_arm", path=rrt_path_rgt, speed_n=speed_n)
            else:
                yumi_x.move_jnts(component_name="rgt_arm", jnt_vals=yumi_s.rgt_arm.homeconf, speed_n=speed_n)
        if component_name in ["lft_arm", "lft_hnd", "both"]:
            rrt_path_lft = rrtc_planner.plan(component_name="lft_arm",
                                             start_conf=np.array(yumi_x.get_jnt_values("lft_arm")),
                                             goal_conf=np.array(yumi_s.lft_arm.homeconf),
                                             obstacle_list=[],
                                             ext_dist=.05,
                                             max_time=300)
            if len(rrt_path_lft) > 5:
                yumi_x.move_jntspace_path(component_name="lft_arm", path=rrt_path_lft, speed_n=speed_n)
            else:
                yumi_x.move_jnts(component_name="lft_arm", jnt_vals=yumi_s.lft_arm.homeconf, speed_n=speed_n)
    else:
        if component_name in ["rgt_arm", "rgt_hnd", "both"]:
            yumi_x.move_jnts(component_name="rgt_arm", jnt_vals=yumi_s.rgt_arm.homeconf, speed_n=speed_n)
        if component_name in ["lft_arm", "lft_hnd", "both"]:
            yumi_x.move_jnts(component_name="lft_arm", jnt_vals=yumi_s.lft_arm.homeconf, speed_n=speed_n)


def help_function():
    # input("Need human to correct the error!!!!!!!!!!!!!!!!!")
    pass


if __name__ == "__main__":
    st = time.time()

    # parameters
    DEBUG = False
    IS_VISION_FEEDBACK = True

    CALIB_MAT_PATH = fs.workdir / "data/calibration/qaqqq.json"

    # Simulation Environment
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

    # init the output pipeline for the model
    std_out = RenderController(root=base.tkRoot, base=base)

    # init the simulation robot
    yumi_robot = ym.Yumi(enable_cc=True)
    component_name = "rgt_hnd"
    yumi_robot_arm = yumi_robot.rgt_arm if "rgt" in component_name else yumi_robot.lft_arm

    # init the real robot
    open_gripper_len = .034
    yumi_con = init_real_rbt(gripper_len=open_gripper_len, toggle_debug=DEBUG, use_motion_exe=True)
    if not DEBUG:
        go_init_pose(yumi_s=yumi_robot, yumi_x=yumi_con, component_name="both")

    # init the pick and place planner
    common_grasp_solver = CommonGraspSolver(num_workers=10)
    # time.sleep(18)
    pp_planner = ppp.PickPlacePlanner(robot_s=yumi_robot, common_grasp_solver=common_grasp_solver)

    # setup the collision model for phoxi camera
    camera_obs = cm.CollisionModel(tm.primitives.Box(box_extents=[1, 1, .2]))
    camera_obs.attach_to(base)
    camera_obs.set_pos(np.array([0.1, 0, 1]))

    # capture the data through vision sensor
    debug_filename = fs.workdir / "data" / "vision_exp" / "20220215-154227.pkl"
    pcd, img, depth_img = capture_vision_info(ip_adr=SENSOR_INFO.IP_ADR_DEPTH_SENSOR_DEBUG,
                                              debug_filename=debug_filename,
                                              toggle_debug=DEBUG)

    # recognize the current states of the test tubes and test tube rack
    detector = TestTubeDetector(affine_mat_path=CALIB_MAT_PATH, )
    detected_test_tubes, tube_rack, rack_tf = detector.analyze_scene(pcd=pcd, texture_img=img[:, :, 0],
                                                                     std_out=None, toggle_yolo=False, save_detect=True)

    # Show the rack and tubes
    rack_cm_mdl = tube_rack.gen_collision_model()
    rack_cm_mdl.attach_to(base)

    tube_rack.gen_mesh_model(gen_tube=True).attach_to(base)

    # setup the goal pattern and task planning solver
    GOAL_PATTERN = np.array([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                             [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                             [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                             [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                             [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]])
    solver = DQNSolver()

    ## Motion planning init
    grasp_cache_dict = {}
    # setup the end configuration
    end_conf = np.array([-0.0137881, -0.97703532, -1.50848807, 0.87929688, -1.99840199,
                         0.13788101, 1.51669112])

    # Start planning
    motion_batch = MotionBatch()
    infeasible_info_dict = {}
    bad_status_weight_matrix = {}
    # useful constants
    retry_number = 3
    while not isdone(node=tube_rack.rack_status, goalpattern=GOAL_PATTERN):
        print_with_border("START PLANING", width=70)
        task_sols = ArrangeTubeSolSeq(rack=tube_rack,
                                      solution=solver.solve(tube_rack.rack_status,
                                                            GOAL_PATTERN,
                                                            infeasible_info_dict,
                                                            toggle_result=False),
                                      tf=rack_tf)
        sol_length = len(task_sols)
        cnt = 0
        for sol in task_sols:
            cnt += 1
            print(f"------- Remaining unsolved path length is {sol_length - cnt} -------")
            print_with_border("Rack State matrix", width=38)
            print(text_pd(tube_rack.rack_status))
            # generate and execute the pick and place motions
            common_grasp_iter_info = None
            is_re_task_planning_flag = False
            _sol = sol

            # moved tube
            moved_tube = _sol.tube
            moved_tube_cm = moved_tube.gen_collision_model()
            # load the grasp for the object
            if grasp_cache_dict.get(moved_tube.type, None) is None:
                grasp_cache_dict[moved_tube.type] = fs.load_json(path=moved_tube.grasps_path)
            grasp_info_list = grasp_cache_dict[moved_tube.type]

            for _ in range(retry_number):  # when the execution failed planning the motion again
                # obs test tubes
                obs_testubes_cmlist = _sol.obs_tube_cm_collection.cm_list

                # collision model of the moved tube
                init_slot_id = _sol.init_slot_id
                goal_slot_id = _sol.goal_slot_id

                obj_init_homomat = _sol.init_homomat
                obj_goal_homomat = _sol.goal_homomat.copy()
                goal_place_offset_dis = .04
                obj_goal_homomat[:3, 3] = obj_goal_homomat[:3, 3] + rack_tf[:3, 2] * goal_place_offset_dis
                homomat_list = [obj_init_homomat, obj_goal_homomat]
                print("??????", obj_goal_homomat)

                # get the start joint configuration of the robot
                start_conf = motion_batch.get_last_motion_rbt_jnt(component_name=component_name)
                if start_conf is None:
                    start_conf = yumi_robot_arm.homeconf

                print_with_border(f"Moving: the test tube is {moved_tube.type}"
                                  f"            init coordinate is {init_slot_id}"
                                  f"              goal coordinate is {goal_slot_id}"
                                  f"              Number of grasps: {len(grasp_info_list)}", width=38)
                exe_logger.info(f"[Task planner] Move {init_slot_id} to {goal_slot_id}")

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
                                                         end_conf=end_conf,
                                                         depart_direction_list=[rack_tf[:3, 2]] * len(
                                                             homomat_list),
                                                         approach_direction_list=[-rack_tf[:3, 2]] * len(
                                                             homomat_list),
                                                         depart_distance_list=[.16, .16],
                                                         approach_distance_list=[.15, .16],
                                                         common_grasp_iter=None,
                                                         approach_jawwidth=None,
                                                         depart_jawwidth=None,
                                                         ad_granularity=.005,
                                                         use_rrt=False,
                                                         obstacle_list=obs_testubes_cmlist + [camera_obs],
                                                         grasp_obstacle_list=[rack_cm_mdl],
                                                         use_incremental=False,
                                                         logger=exe_logger)

                if start2pickapproach_motion_seg is not None:
                    # motion planning successfully
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
                    if DEBUG:
                        exe_logger.info("Planning successfully")
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
                        # base.run()
                    # execute the motions
                    if not DEBUG:
                        # Motion Execution
                        is_exec_succ, exe_code = yumi_con.exe_pick_and_place_motion(motion_batch=exe_motion_batch,
                                                                                    speed_n_place=100)
                        # error
                        print(exe_code)
                        if not is_exec_succ and exe_code == INSERT_FAILURE_ERR:
                            is_recoverable = feedback_insertion_error(yumi_s=yumi_robot,
                                                                      yumi_con=yumi_con,
                                                                      tube=moved_tube.copy(),
                                                                      tube_pose=
                                                                      exe_motion_batch.place_depart.objpose_list[-1],
                                                                      pp_planner=pp_planner,
                                                                      obstacle_list=obs_testubes_cmlist + [camera_obs,
                                                                                                           rack_cm_mdl])
                            if is_recoverable:
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

                        # Vision Feedback
                        if IS_VISION_FEEDBACK or not is_exec_succ:
                            pcd_t, img_t, depth_img = capture_vision_info(
                                ip_adr=SENSOR_INFO.IP_ADR_DEPTH_SENSOR_DEBUG,
                                debug_filename=debug_filename,
                                toggle_debug=DEBUG)
                            detected_test_tubes_t, tube_rack_t, rack_tf_t = detector.analyze_scene(pcd=pcd_t,
                                                                                                   texture_img=img_t[:,
                                                                                                               :,
                                                                                                               0],
                                                                                                   std_out=None,
                                                                                                   toggle_yolo=False,
                                                                                                   save_detect=True)
                        # check if the execution is success
                        if is_exec_succ:
                            exe_logger.info("Execution successfully")
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
                            # check if motion is finished to prevent the redundant task planning
                            if IS_VISION_FEEDBACK:
                                if isdone(node=tube_rack_t.rack_status, goalpattern=GOAL_PATTERN):
                                    is_re_task_planning_flag = True
                                    break
                                # check if the motion exec successfully
                                if np.array_equal(tube_rack_t.rack_status, tube_rack.rack_status):
                                    print("Execute the motion successfully!")
                                    if is_restart_planning(rack_tf_t, rack_tf, toggle_debug=True):
                                        # TODO will plans the new motion in the same time
                                        print("NEED REPLANING")
                                        rack_cm_mdl = tube_rack_t.gen_collision_model()
                                        task_sols.update_rack(tube_rack_t)
                                        tube_rack = tube_rack_t
                                        rack_tf = rack_tf_t
                                else:
                                    # check if the vision system works or some test tube is out of the rack
                                    if len(np.where(tube_rack_t.rack_status > 0)[0]) < len(
                                            np.where(tube_rack.rack_status > 0)[0]):
                                        help_function()
                                    print("REPLANNING NEEDED")
                                    rack_cm_mdl = tube_rack_t.gen_collision_model()
                                    task_sols.update_rack(tube_rack_t)
                                    tube_rack = tube_rack_t
                                    rack_tf = rack_tf_t
                                    # restart
                                    is_re_task_planning_flag = True
                                    break
                        else:
                            exe_logger.info(f"Execution Failed! Execution error code {exe_code}")
                            # change a new grasp and retry again
                            if exe_code == GRASP_COLLISION_ERR or exe_code == GRASP_NO_TUBE_ERR:
                                # common_grasp_iter_info = common_grasp_iter_info[1:]
                                grasp_info_list = [grasp_info_list[v[0]] for v in common_grasp_iter_info[1:]]
                                # update data
                                rack_cm_mdl = tube_rack_t.gen_collision_model()
                                task_sols.update_rack(tube_rack_t)
                                tube_rack = tube_rack_t
                                rack_tf = rack_tf_t
                                _sol = task_sols.current
                                continue

                else:
                    print("---" * 17)
                    print("Planning Failed!! Restart Search a New Path")
                    # planning failed
                    current_state = tube_rack.rack_status.copy()
                    next_state = current_state.copy()
                    next_state[tuple(init_slot_id)], next_state[tuple(goal_slot_id)] = next_state[tuple(goal_slot_id)], \
                                                                                       next_state[tuple(init_slot_id)]
                    weight_matrix = bad_status_weight_matrix.get(str(current_state),
                                                                 np.zeros_like(tube_rack.rack_status)).copy()
                    weight_matrix[tuple(init_slot_id)] = 1
                    weight_matrix[tuple(goal_slot_id)] = 1
                    bad_status_weight_matrix[str(current_state)] = weight_matrix

                    # add infeasible state to dict
                    infeasible_actions = infeasible_info_dict.get(str(current_state), [])
                    is_remove_all_init_slot_id_actions = not is_init_feasible
                    infeasible_actions.extend(solver.action_between_states(current_state,
                                                                           next_state,
                                                                           remove_action_group=is_remove_all_init_slot_id_actions))
                    infeasible_info_dict[str(current_state)] = list(set(infeasible_actions))

                    print("Weight matrix of the state is ...\n", weight_matrix)
                    print("Infeasible action of the state is ...", infeasible_actions)
                    print(f"Remove all the action that pick {init_slot_id} ...", is_remove_all_init_slot_id_actions)
                    is_re_task_planning_flag = True
                break
            if is_re_task_planning_flag:
                break
    print_with_border("All test tubes have already been arranged", width=70)

    if DEBUG:
        show_animation(yumi_robot, motion_batch, base)
        print(f"Time of running is {time.time() - st}")
        common_grasp_solver.stop()
        base.run()
    else:
        yumi_con.stop()
    # base.run()
    exit(0)
