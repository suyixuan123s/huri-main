import time

from huri.core.file_sys import workdir, load_pickle, load_json
from huri.core.common_import import *
from huri.definitions.utils_structure import MotionElement, MotionBatch


def is_done(state: np.ndarray, goal_pattern: np.ndarray):
    if np.any((goal_pattern - state)[state > 0]):
        return False
    else:
        return True


if __name__ == "__main__":
    st = time.time()
    from panda3d.core import loadPrcFileData

    loadPrcFileData('', 'notify-level error')
    # loadPrcFileData('', 'default-directnotify-level info')
    from huri.core.constants import IP_ADR, CALIB_MAT_PATH
    from huri.test.app.animation import show_animation
    from huri.components.vision.tube_detector import TestTubeDetector
    import huri.components.planning.symmetric_pick_and_place_planner as ppp
    from huri.vision.phoxi_capture import vision_pipeline, SensorMarkerHandler
    from time import strftime
    from huri.definitions.rack_def import ArrangeTubeSolSeq
    from huri.components.pipeline.data_pipeline import RenderController, MotionController
    from huri.core.print_tool import text_pd, print_with_border
    import basis.trimesh as tm
    import visualization.panda.rpc.rviz_client as rv_client
    from huri.components.task_planning.tube_puzzle_learning_solver import DQNSolver
    from huri.test.app.remote_exe_client import Remote_Exe_Client

    DEBUG = True
    IP_ADR = "192.168.125.100:18300"
    CALIB_MAT_PATH = workdir / "data/calibration/qaqqq.json"

    # create simulation environment
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    rvc = rv_client.RVizClient(host="localhost:9999")
    remote_exe_client = Remote_Exe_Client()

    # init the output pipeline for the model
    std_out = RenderController(root=base.tkRoot, base=base)

    # init the test tube detector
    detector = TestTubeDetector(affine_mat_path=CALIB_MAT_PATH)

    # init the simulation robot
    yumi_robot = ym.Yumi(enable_cc=True)
    component_name = "rgt_hnd"
    yumi_robot_arm = yumi_robot.rgt_arm if "rgt" in component_name else yumi_robot.lft_arm

    # init the pick and place planner
    pp_planner = ppp.PickPlacePlanner(robot_s=yumi_robot)

    # setup the collision model for phoxi camera
    camera_obs = cm.CollisionModel(tm.primitives.Box(box_extents=[1, 1, .2]))
    camera_obs.attach_to(base)
    camera_obs.set_pos(np.array([0.1, 0, 1]))

    # use the offline vision data for debug
    if DEBUG:
        filename = workdir / "data" / "vision_exp" / "20220201-002227.pkl"
        pcd, img = load_pickle(filename)
    else:
        pcd, img = vision_pipeline(SensorMarkerHandler(ip_adr=IP_ADR),
                                   workdir / "data" / "vision_exp" / f"{strftime('%Y%m%d-%H%M%S')}.pkl")

    # recognize the current states of the test tubes and test tube rack
    detected_test_tubes, tube_rack, rack_tf = detector.analyze_scene(pcd=pcd, texture_img=img[:, :, 0],
                                                                     std_out=None, toggle_yolo=False, save_detect=True)
    try:
        rvc.remote_load_img()
    except:
        print("Do not open the server")
    # Show the rack and tubes
    rack_cm_mdl = tube_rack.gen_collision_model()
    rack_cm_mdl.attach_to(base)
    # uncomment to check the collision model of the test tube rack
    # tube_rack.gen_mesh_model(gen_tube=True).attach_to(base)
    # base.run()

    # setup the goal pattern and task planning solver
    GOAL_PATTERN = np.array([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                             [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                             [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                             [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                             [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]])
    solver = DQNSolver()
    # print goal pattern
    print_with_border("Goal pattern", width=38)
    print(text_pd(GOAL_PATTERN))

    ## task planning init
    tuberack_bk = tube_rack.copy()
    motion_batch = MotionBatch()
    weight_matrix = np.zeros_like(tube_rack.rack_status)

    ## Motion planning init
    grasp_cache_dict = {}
    # setup the end configuration
    end_conf = np.array([-0.0137881, -0.97703532, -1.50848807, 0.87929688, -1.99840199,
                         0.13788101, 1.51669112])

    # bad status
    bad_status_weight_matrix = {}
    infeasible_info_dict = {}

    planning_finished_flag = False
    taskplanning_restart_flag = False
    move_map = np.zeros_like(tube_rack.rack_status)

    # real exe
    from huri.components.yumi_control.yumi_con import YumiController
    from huri.math.units import Mm
    from huri.components.pipeline.data_pipeline import motion_executer

    open_gripper_len = Mm(35)
    try:
        yumi_con = YumiController()
    except:
        yumi_con = YumiController()
    yumi_con.move_gripper(component_name="rgt_arm", width=open_gripper_len)
    yumi_con.set_gripper_speed("rgt_arm", 10)

    print("---" * 20)

    # check if everything test tube is arranged
    if is_done(state=tube_rack.rack_status, goal_pattern=GOAL_PATTERN):
        print_with_border("All test tubes have already been arranged", width=70)
    while True:
        # print infomation
        print_with_border("START PLANING", width=70)
        print_with_border("Weight matrix", width=38)
        print(text_pd(weight_matrix))
        # get the solution for the task planning
        task_sol = ArrangeTubeSolSeq(rack=tube_rack,
                                     solution=solver.solve(tube_rack.rack_status,
                                                           GOAL_PATTERN,
                                                           infeasible_info_dict,
                                                           toggle_result=False),
                                     tf=rack_tf)

        sol_length = len(task_sol)
        cnt = 0
        # planning motions
        for _sol in task_sol:
            cnt += 1
            print(f"------- Remaining unsolved path length is {sol_length - cnt} -------")
            print_with_border("Rack State matrix", width=38)
            print(text_pd(tube_rack.rack_status))
            # obs test tubes
            obs_testubes_list = _sol.obs_tube_cm_collection.cm_list

            # collision model of the moved tube
            moved_tube = _sol.tube
            moved_tube_cm = _sol.tube.gen_collision_model()
            _init_slot_id = _sol.init_slot_id
            _goal_slot_id = _sol.goal_slot_id
            # uncomment to show the collision model
            # _sol.obs_tube_cm_collection.show_cdprimit()
            # _sol.obs_tube_cm_collection.attach_to(base)
            # _sol.obs_rack_cm.show_cdprimit()
            # _sol.obs_rack_cm.attach_to(base)
            # base.run()

            # pick and place
            # setup the initial and goal pose for the manipulated test tube
            obj_init_homomat = _sol.init_homomat
            obj_goal_homomat = _sol.goal_homomat.copy()
            goal_place_offset_dis = .04
            obj_goal_homomat[:3, 3] = obj_goal_homomat[:3, 3] + rack_tf[:3, 2] * goal_place_offset_dis
            homomat_list = [obj_init_homomat, obj_goal_homomat]

            # get the start joint configuration of the robot
            start_conf = motion_batch.get_last_motion_rbt_jnt(component_name=component_name)
            if start_conf is None:
                start_conf = yumi_robot_arm.homeconf

            # load the grasp for the object
            if grasp_cache_dict.get(moved_tube.type, None) is None:
                grasp_cache_dict[moved_tube.type] = load_json(path=moved_tube.grasps_path)
            grasp_info_list = grasp_cache_dict[moved_tube.type]

            print_with_border(f"Moving: the test tube is {moved_tube.type}"
                              f"            init coordinate is {_init_slot_id}"
                              f"              goal coordinate is {_goal_slot_id}"
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
                                                     approach_distance_list=[.16, .16],
                                                     approach_jawwidth=None,
                                                     depart_jawwidth=None,
                                                     ad_granularity=.02,
                                                     use_rrt=False,
                                                     obstacle_list=obs_testubes_list + [camera_obs],
                                                     grasp_obstacle_list=[rack_cm_mdl],
                                                     use_incremental=False, )

            if conf_list_tmp is not None:
                print("---" * 17)
                print("Successfully!")
                # tube rack operation
                tube_rack.remove_slot(_init_slot_id)
                tube_rack.insert_tube(_goal_slot_id, moved_tube)
                # reset the pose of the obstacle
                # add motions into motion batch
                remote_motion_batch = MotionBatch()
                for i in range(len(conf_list_tmp)):
                    motion_batch.append(MotionElement(obj_cm=moved_tube_cm.copy(),
                                                      objpose_list=objpose_list_tmp[i],
                                                      conf_list=conf_list_tmp[i],
                                                      jawwidth_list=jawwidth_list_tmp[i],
                                                      hnd_name=component_name,
                                                      obs_list=obs_testubes_list))
                    remote_motion_batch.append(MotionElement(obj_cm=moved_tube_cm.copy(),
                                                             objpose_list=objpose_list_tmp[i],
                                                             conf_list=conf_list_tmp[i],
                                                             jawwidth_list=jawwidth_list_tmp[i],
                                                             hnd_name=component_name,
                                                             obs_list=obs_testubes_list + [rack_cm_mdl]))
                try:
                    remote_exe_client.send_motion_batch(remote_motion_batch)
                except:
                    print("Cannot connect to remote client")
                    # show_animation(yumi_robot=yumi_robot,
                    #                motion_batch=motion_batch,
                    #                base=base)
                    # base.run()
                motion_executer(yumi_con=yumi_con, motion_batch=remote_motion_batch, open_gripper_len=open_gripper_len)

                pcd, img = vision_pipeline(SensorMarkerHandler(ip_adr=IP_ADR),
                                           workdir / "data" / "vision_exp" / f"{strftime('%Y%m%d-%H%M%S')}.pkl")
                detected_test_tubes_new, tube_rack_new, rack_tf_new = detector.analyze_scene(pcd=pcd,
                                                                                             texture_img=img[:, :, 0],
                                                                                             std_out=None,
                                                                                             toggle_yolo=False,
                                                                                             save_detect=True)
                if np.array_equal(tube_rack_new.rack_status, tube_rack.rack_status) or True:
                    print("Execute the motion successfully!")
                    rack_cm_mdl = tube_rack_new.gen_collision_model()
                    task_sol.update_rack(tube_rack_new)
                    tube_rack = tube_rack_new
                    rack_tf = rack_tf_new
                else:
                    print("Potential error exist")
                    exit(0)
            else:
                print("---" * 17)
                print("Planning Failed!! Restart Search a New Path")
                # planning failed
                current_state = tube_rack.rack_status.copy()
                next_state = current_state.copy()
                next_state[tuple(_init_slot_id)], next_state[tuple(_goal_slot_id)] = next_state[tuple(_goal_slot_id)], \
                                                                                     next_state[tuple(_init_slot_id)]
                weight_matrix = bad_status_weight_matrix.get(str(current_state),
                                                             np.zeros_like(tube_rack.rack_status)).copy()
                weight_matrix[tuple(_init_slot_id)] = 1
                weight_matrix[tuple(_goal_slot_id)] = 1
                bad_status_weight_matrix[str(current_state)] = weight_matrix

                # add infeasible state to dict
                infeasible_actions = infeasible_info_dict.get(str(current_state), [])
                is_remove_all_init_slot_id_actions = False
                if len(init_feasible_iks) == 0:
                    is_remove_all_init_slot_id_actions = True
                infeasible_actions.extend(solver.action_between_states(current_state,
                                                                       next_state,
                                                                       remove_action_group=is_remove_all_init_slot_id_actions))
                infeasible_info_dict[str(current_state)] = list(set(infeasible_actions))

                print("Weight matrix of the state is ...\n", weight_matrix)
                print("Infeasible action of the state is ...", infeasible_actions)
                print(f"Remove all the action that pick {_init_slot_id} ...", is_remove_all_init_slot_id_actions)
                break
        else:
            planning_finished_flag = True

        if planning_finished_flag:
            print("Successfully planning all!")
            break

    ed = time.time()
    print("Total time consuming is:", ed - st)

    yumi_con = None

    show_animation(yumi_robot=yumi_robot,
                   motion_batch=motion_batch,
                   base=base, yumi_con=yumi_con)
    base.run()
    # exit(0)
