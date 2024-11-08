from huri.components.exe.utils import capture_vision_info, init_real_rbt, is_restart_planning
from huri.core.common_import import *
from huri.definitions.utils_structure import MotionElement, MotionBatch
from huri.learning.env.arrangement_planning_rack.utils import isdone
from panda3d.core import loadPrcFileData

loadPrcFileData('', 'notify-level error')
from huri.test.app.animation import show_animation
from huri.components.vision.tube_detector import TestTubeDetector
import huri.components.planning.symmetric_pick_and_place_planner as ppp
from huri.components.pipeline.data_pipeline import motion_executer
from huri.definitions.rack_def import ArrangeTubeSolSeq
from huri.components.pipeline.data_pipeline import RenderController, MotionController
from huri.core.print_tool import text_pd, print_with_border
import basis.trimesh as tm
import visualization.panda.rpc.rviz_client as rv_client
from huri.components.task_planning.tube_puzzle_learning_solver import DQNSolver
from huri.test.app.remote_exe_client import Remote_Exe_Client
import time
if __name__ == "__main__":
    st = time.time()
    DEBUG = True

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
    open_gripper_len = .035
    yumi_con = init_real_rbt(gripper_len=open_gripper_len, toggle_debug=DEBUG)

    # init the pick and place planner
    pp_planner = ppp.PickPlacePlanner(robot_s=yumi_robot)

    # setup the collision model for phoxi camera
    camera_obs = cm.CollisionModel(tm.primitives.Box(box_extents=[1, 1, .2]))
    camera_obs.attach_to(base)
    camera_obs.set_pos(np.array([0.1, 0, 1]))

    # capture the data through vision sensor
    debug_filename = fs.workdir / "data" / "vision_exp" / "20211228-234523.pkl"
    pcd, img = capture_vision_info(debug_filename=debug_filename,
                                   toggle_debug=DEBUG)

    # recognize the current states of the test tubes and test tube rack
    detector = TestTubeDetector(affine_mat_path=CALIB_MAT_PATH)
    detected_test_tubes, tube_rack, rack_tf = detector.analyze_scene(pcd=pcd, texture_img=img[:, :, 0],
                                                                     std_out=None, toggle_yolo=False, save_detect=True)

    # Show the rack and tubes
    rack_cm_mdl = tube_rack.gen_collision_model()
    rack_cm_mdl.attach_to(base)

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

            # obs test tubes
            obs_testubes_cmlist = sol.obs_tube_cm_collection.cm_list

            # collision model of the moved tube
            moved_tube = sol.tube
            moved_tube_cm = moved_tube.gen_collision_model()
            init_slot_id = sol.init_slot_id
            goal_slot_id = sol.goal_slot_id

            obj_init_homomat = sol.init_homomat
            obj_goal_homomat = sol.goal_homomat.copy()
            goal_place_offset_dis = .04
            obj_goal_homomat[:3, 3] = obj_goal_homomat[:3, 3] + rack_tf[:3, 2] * goal_place_offset_dis
            homomat_list = [obj_init_homomat, obj_goal_homomat]

            # get the start joint configuration of the robot
            start_conf = motion_batch.get_last_motion_rbt_jnt(component_name=component_name)
            if start_conf is None:
                start_conf = yumi_robot_arm.homeconf

            # load the grasp for the object
            if grasp_cache_dict.get(moved_tube.type, None) is None:
                grasp_cache_dict[moved_tube.type] = fs.load_json(path=moved_tube.grasps_path)
            grasp_info_list = grasp_cache_dict[moved_tube.type]

            print_with_border(f"Moving: the test tube is {moved_tube.type}"
                              f"            init coordinate is {init_slot_id}"
                              f"              goal coordinate is {goal_slot_id}"
                              f"              Number of grasps: {len(grasp_info_list)}", width=38)

            conf_list_tmp, jawwidth_list_tmp, objpose_list_tmp, init_feasible_iks = \
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
                                                     approach_jawwidth=None,
                                                     depart_jawwidth=None,
                                                     ad_granularity=.02,
                                                     use_rrt=False,
                                                     obstacle_list=obs_testubes_cmlist + [camera_obs],
                                                     grasp_obstacle_list=[rack_cm_mdl],
                                                     use_incremental=False, )
            if conf_list_tmp is not None:
                print("---" * 17)
                print("Planning Successfully!")
                # tube rack operation
                tube_rack.remove_slot(init_slot_id)
                tube_rack.insert_tube(goal_slot_id, moved_tube)
                # add motions into motion batch
                remote_motion_batch = MotionBatch()
                for i in range(len(conf_list_tmp)):
                    motion_batch.append(MotionElement(obj_cm=moved_tube_cm.copy(),
                                                      objpose_list=objpose_list_tmp[i],
                                                      conf_list=conf_list_tmp[i],
                                                      jawwidth_list=jawwidth_list_tmp[i],
                                                      hnd_name=component_name,
                                                      obs_list=obs_testubes_cmlist))
                    remote_motion_batch.append(MotionElement(obj_cm=moved_tube_cm.copy(),
                                                             objpose_list=objpose_list_tmp[i],
                                                             conf_list=conf_list_tmp[i],
                                                             jawwidth_list=jawwidth_list_tmp[i],
                                                             hnd_name=component_name,
                                                             obs_list=obs_testubes_cmlist + [rack_cm_mdl]))
                show_animation(yumi_robot, motion_batch, base)
                print(f"Time of running is {time.time() - st}")
                base.run()
                if not DEBUG:
                    # Execution
                    motion_executer(yumi_con=yumi_con, motion_batch=remote_motion_batch,
                                    open_gripper_len=open_gripper_len)
                    # Feedback
                    pcd_t, img_t = capture_vision_info(debug_filename=debug_filename, toggle_debug=DEBUG)
                    detected_test_tubes_t, tube_rack_t, rack_tf_t = detector.analyze_scene(pcd=pcd_t,
                                                                                           texture_img=img_t[:, :, 0],
                                                                                           std_out=None,
                                                                                           toggle_yolo=False,
                                                                                           save_detect=True)
                    if is_restart_planning(rack_tf_t, rack_tf, toggle_debug=True):
                        exit(0)
                    if np.array_equal(tube_rack_t.rack_status, tube_rack.rack_status):
                        print("Execute the motion successfully!")
                        rack_cm_mdl = tube_rack_t.gen_collision_model()
                        task_sols.update_rack(tube_rack_t)
                        tube_rack = tube_rack_t
                        rack_tf = rack_tf_t
                    else:
                        print("Potential error exist")
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
                is_remove_all_init_slot_id_actions = True if len(init_feasible_iks) == 0 else False
                infeasible_actions.extend(solver.action_between_states(current_state,
                                                                       next_state,
                                                                       remove_action_group=is_remove_all_init_slot_id_actions))
                infeasible_info_dict[str(current_state)] = list(set(infeasible_actions))

                print("Weight matrix of the state is ...\n", weight_matrix)
                print("Infeasible action of the state is ...", infeasible_actions)
                print(f"Remove all the action that pick {init_slot_id} ...", is_remove_all_init_slot_id_actions)
                break

    print_with_border("All test tubes have already been arranged", width=70)
    base.run()
    exit(0)
