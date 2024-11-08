import numpy as np
import motion.probabilistic.rrt_connect as rrtc
from huri.core.file_sys import workdir, load_pickle, load_json
from huri.definitions.tube_def import TubeType, TestTube
from huri.core.common_import import *
from huri.definitions.utils_structure import MotionElement, MotionBatch


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


def is_done(state: np.ndarray, goal_pattern: np.ndarray):
    if np.any((goal_pattern - state)[state > 0]):
        return False
    else:
        return True


if __name__ == "__main__":

    from huri.test.app.animation import show_animation
    from huri.components.vision.tube_detector import TestTubeDetector
    import huri.components.planning.symmetric_pick_and_place_planner as ppp
    from huri.vision.phoxi_capture import vision_pipeline, SensorMarkerHandler
    from time import strftime
    from huri.definitions.rack_def import ArrangeTubeSolSeq
    from huri.components.pipeline.data_pipeline import RenderController, MotionController
    from huri.core.print_tool import text_pd, print_with_border
    from huri.components.pipeline.data_pipeline import motion_executer
    from huri.components.yumi_control.yumi_con import YumiController
    import basis.trimesh as tm
    from panda3d.core import loadPrcFileData
    from huri.math.units import Mm
    import visualization.panda.rpc.rviz_client as rv_client

    loadPrcFileData('', 'notify-level error')
    IP_ADR = "192.168.125.100:18300"
    CALIB_MAT_PATH = workdir / "data/calibration/qaqqq.json"
    DEBUG = False
    REAL_ROBOT = True

    # create simulation environment
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    rvc = rv_client.RVizClient(host="localhost:9999")

    # init the output pipeline for the model
    std_out = RenderController(root=base.tkRoot, base=base)
    # init the test tube detector
    detector = TestTubeDetector(affine_mat_path=CALIB_MAT_PATH)
    # init the simulation robot
    yumi_robot = ym.Yumi(enable_cc=True)
    yumi_robot.gen_meshmodel().attach_to(base)
    component_name = "rgt_hnd"
    yumi_robot_arm = yumi_robot.rgt_arm if "rgt" in component_name else yumi_robot.lft_arm
    # init the pick and place planner
    pp_planner = ppp.PickPlacePlanner(robot_s=yumi_robot)

    # setup the collision model for phoxi camera
    camera_obs = cm.CollisionModel(tm.primitives.Box(box_extents=[1, 1, .2]))
    camera_obs.attach_to(base)
    camera_obs.set_pos(np.array([0.1, 0, 1]))

    open_gripper_len = Mm(35)
    # connect to the real yumi robot
    if REAL_ROBOT:
        yumi_con = YumiController()
        go_init_pose(yumi_s=yumi_robot, yumi_x=yumi_con, component_name="both")
        yumi_con.move_gripper(component_name="rgt_arm", width=open_gripper_len)
    else:
        yumi_con = None

    # use the offline vision data for debug
    if DEBUG:
        filename = workdir / "data" / "vision_exp" / "20211116-212148.pkl"
        pcd, img = load_pickle(filename)
    else:
        pcd, img = vision_pipeline(SensorMarkerHandler(ip_adr=IP_ADR),
                                   workdir / "data" / "vision_exp" / f"{strftime('%Y%m%d-%H%M%S')}.pkl")

    # recognize the current states of the test tubes and test tube rack
    detected_test_tubes, tube_rack, rack_tf = detector.analyze_scene(pcd=pcd, texture_img=img[:, :, 0],
                                                                     std_out=None, toggle_yolo=False,
                                                                     save_detect=True)
    rvc.remote_load_img()
    # Show the rack and tubes
    rack_cm_mdl = tube_rack.gen_collision_model()
    # uncomment to check the collision model of the test tube rack
    # tube_rack.gen_mesh_model(gen_tube=True).attach_to(base)
    # base.run()

    # setup the goal pattern
    GOAL_PATTERN = np.array([[3, 3, 0, 0, 0, 0, 0, 0, 0, 1],
                             [3, 3, 0, 0, 0, 0, 0, 0, 0, 1],
                             [3, 3, 0, 0, 0, 0, 0, 0, 0, 1],
                             [3, 3, 0, 0, 0, 0, 0, 0, 0, 1],
                             [3, 3, 0, 0, 0, 0, 0, 0, 0, 1]])
    # print goal pattern
    print_with_border("Goal pattern", width=38)
    print(text_pd(GOAL_PATTERN))

    # task planning init
    tube_rack_bk = tube_rack.copy()
    motion_batch = MotionBatch()
    weight_matrix = np.zeros_like(tube_rack.rack_status)

    # bad status
    bad_status_list = []
    bad_start_goal_ids = []

    planning_finished_flag = False
    move_map = np.zeros_like(tube_rack.rack_status)
    # setup the end configuration
    end_conf = np.array([-0.0137881, -0.97703532, -1.50848807, 0.87929688, -1.99840199,
                         0.13788101, 1.51669112])
    print("---" * 20)

    while True:
        # check if everything test tube is arranged
        if is_done(state=tube_rack.rack_status, goal_pattern=GOAL_PATTERN):
            print_with_border("All test tubes have already been arranged", width=70)
            break

        # print infomation
        print_with_border("START PLANING", width=70)
        print_with_border("Weight matrix", width=38)
        print(text_pd(weight_matrix))
        print_with_border("Rack State matrix", width=38)
        print(text_pd(tube_rack.rack_status))

        # get the solution for the task planning
        task_sol = ArrangeTubeSolSeq(rack=tube_rack,
                                     solution=tube_rack.solve(GOAL_PATTERN, weight_matrix=weight_matrix),
                                     tf=rack_tf)

        # planning motions
        for _sol in task_sol:
            print(f"____________________________________{task_sol.step_cnt}_______________________")
            # obs test tubes
            obs_testubes_list = _sol.obs_tube_cm_collection.cm_list

            # collision model of the moved tube
            moved_tube = _sol.tube
            moved_tube_cm = _sol.tube.gen_collision_model()
            _init_slot_id = _sol.init_slot_id
            _goal_slot_id = _sol.goal_slot_id
            _goal_rack_status = _sol.goal_rack_status
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
            goal_place_offset_dis = .03
            obj_goal_homomat[:3, 3] = obj_goal_homomat[:3, 3] + np.array([0,0,1]) * goal_place_offset_dis
            homomat_list = [obj_init_homomat, obj_goal_homomat]

            # moved_tube_at_init = moved_tube_cm.copy()
            # moved_tube_at_init.set_homomat(obj_init_homomat)
            # moved_tube_at_init.set_rgba(np.array([1,0,0,1]))
            # moved_tube_at_init.attach_to(base)
            #
            # moved_tube_at_goal = moved_tube_cm.copy()
            # moved_tube_at_goal.set_homomat(obj_goal_homomat)
            # moved_tube_at_goal.set_rgba(np.array([0,1,0,1]))
            # moved_tube_at_goal.attach_to(base)
            #
            # moved_tube_at_goal2 = moved_tube_cm.copy()
            # moved_tube_at_goal2.set_homomat(_sol.goal_homomat.copy())
            # moved_tube_at_goal2.attach_to(base)
            #
            # rack_cm_mdl.attach_to(base)
            # base.run()

            # get the start joint configuration of the robot
            start_conf = motion_batch.get_last_motion_rbt_jnt(component_name=component_name)
            if start_conf is None:
                start_conf = yumi_robot_arm.homeconf
            # load the grasp for the object
            grasp_info_list = load_json(path=moved_tube.grasps_path)

            print_with_border(f"Moving: the test tube is {moved_tube.type}"
                              f"            init coordinate is {_init_slot_id}"
                              f"              goal coordinate is {_goal_slot_id}"
                              f"              Number of grasps: {len(grasp_info_list)}", width=38)
            print_with_border("GOAL Rack State", width=38)
            print(text_pd(_goal_rack_status))

            conf_list_tmp, jawwidth_list_tmp, objpose_list_tmp = \
                pp_planner.gen_pick_and_place_motion(hnd_name=component_name,
                                                     tube=moved_tube,
                                                     tube_cm=moved_tube_cm,
                                                     grasp_info_list=grasp_info_list,
                                                     goal_homomat_list=homomat_list,
                                                     start_conf=start_conf,
                                                     end_conf=end_conf,
                                                     depart_direction_list=[np.array([0, 0, 1])] * len(
                                                         homomat_list),
                                                     approach_direction_list=[-np.array([0, 0, 1])] * len(
                                                         homomat_list),
                                                     depart_distance_list=[.18 + 0.112 - moved_tube.height,
                                                                           .18 + 0.112 - moved_tube.height],
                                                     approach_distance_list=[.15 - goal_place_offset_dis,
                                                                             .15 - goal_place_offset_dis],
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
                # reset the pose of the obstacle
                # add motions into motion batch
                motion_tmp = MotionBatch()
                for i in range(len(conf_list_tmp)):
                    motion_batch.append(MotionElement(obj_cm=moved_tube_cm.copy(),
                                                      objpose_list=objpose_list_tmp[i],
                                                      conf_list=conf_list_tmp[i],
                                                      jawwidth_list=jawwidth_list_tmp[i],
                                                      hnd_name=component_name,
                                                      obs_list=obs_testubes_list + [rack_cm_mdl]))
                    motion_tmp.append(MotionElement(obj_cm=moved_tube_cm.copy(),
                                                    objpose_list=objpose_list_tmp[i],
                                                    conf_list=conf_list_tmp[i],
                                                    jawwidth_list=jawwidth_list_tmp[i],
                                                    hnd_name=component_name,
                                                    obs_list=obs_testubes_list + [rack_cm_mdl]))
                print("execute motions: ")
                motion_executer(yumi_con=yumi_con, motion_batch=motion_tmp, open_gripper_len=open_gripper_len)
                # test if the movement is fine
                # update the tube rack
                # reset_cnt = 0
                # while True:
                if DEBUG:
                    filename = workdir / "data" / "vision_exp" / "20210917-153348.pkl"
                    pcd, img = load_pickle(filename)
                else:
                    pcd, img = vision_pipeline(SensorMarkerHandler(ip_adr=IP_ADR),
                                               workdir / "data" / "vision_exp" / f"{strftime('%Y%m%d-%H%M%S')}.pkl")
                detected_test_tubes_new, tube_rack_new, rack_tf_new = detector.analyze_scene(pcd=pcd,
                                                                                             texture_img=img[:, :, 0],
                                                                                             std_out=None,
                                                                                             toggle_yolo=False,
                                                                                             save_detect=True)
                rvc.remote_load_img()
                rack_cm_mdl = tube_rack_new.gen_collision_model()
                task_sol.update_rack(tube_rack_new)
                tube_rack = tube_rack_new
                rack_tf = rack_tf_new
                if np.array_equal(tube_rack_new.rack_status, _goal_rack_status):
                    # tube_rack.remove_slot(_init_slot_id)
                    # tube_rack.insert_tube(_goal_slot_id, moved_tube)
                    print("Execute the motion successfully!")
                else:
                    print("Potential error exist")
                    break
            else:
                print("---" * 17)
                print("Planning Failed!! Restart Search a New Path")
                # planning failed
                current_state = tube_rack.rack_status.copy()
                bad_status_list.append(current_state)
                bad_start_goal_ids.append([_init_slot_id, _goal_slot_id])

                weight_matrix = np.zeros_like(tube_rack.rack_status)
                for id, one_bad_status in enumerate(bad_status_list):
                    if np.array_equal(one_bad_status, current_state):
                        weight_matrix[tuple(bad_start_goal_ids[id][0])] = 1
                        weight_matrix[tuple(bad_start_goal_ids[id][1])] = 1
                # motions = MotionBatch()
                break
        else:
            planning_finished_flag = True

        if planning_finished_flag:
            print("Successfully planning all!")
            break

    show_animation(yumi_robot=yumi_robot,
                   motion_batch=motion_batch,
                   base=base)

    base.run()
