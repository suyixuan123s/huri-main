import time

from huri.core.file_sys import workdir, load_pickle, load_json
from huri.definitions.testtube_def import TubeType, TestTube
from huri.core.common_import import *
from huri.definitions.utils_structure import MotionElement, MotionBatch


def is_done(state: np.ndarray, goal_pattern: np.ndarray):
    if np.any((goal_pattern - state)[state > 0]):
        return False
    else:
        return True


def get_objcm_grasps(tube: TestTube):
    objcm = cm.CollisionModel(initor=tube.model_path, cdprimit_type="cylinder", expand_radius=0.001)
    grasp_info_list = load_json(path=tube.grasps_path)
    return objcm, grasp_info_list


if __name__ == "__main__":

    from panda3d.core import loadPrcFileData

    loadPrcFileData('', 'notify-level error')
    # loadPrcFileData('', 'default-directnotify-level info')
    from huri.core.constants import IP_ADR, CALIB_MAT_PATH
    from huri.test.app.animation import show_animation
    from huri.components.vision.tube_detector import TestTubeDetector
    import huri.components.planning.pick_place_planner as ppp
    from huri.vision.phoxi_capture import vision_pipeline, SensorMarkerHandler
    from time import strftime
    from huri.definitions.rack_def import ArrangeTubeSolSeq
    from huri.components.pipeline.data_pipeline import RenderController, MotionController
    from huri.core.print_tool import text_pd, print_with_border

    DEBUG = True
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    std_out = RenderController(root=base.tkRoot, base=base)
    detector = TestTubeDetector(affine_mat_path=CALIB_MAT_PATH)
    yumi_robot = ym.Yumi(enable_cc=True)



    pp_planner = ppp.PickPlacePlanner(robot_s=yumi_robot)
    yumi_robot.gen_meshmodel().attach_to(base)
    if DEBUG:
        filename = workdir / "data" / "vision_exp" / "20210917-153348.pkl"
        pcd, img = load_pickle(filename)
    else:
        pcd, img = vision_pipeline(SensorMarkerHandler(ip_adr=IP_ADR),
                                   workdir / "data" / "vision_exp" / f"{strftime('%Y%m%d-%H%M%S')}.pkl")

    # Recognize the current states
    detected_test_tube, tuberack, tuberack_trans = detector.analyze_scene(pcd=pcd, texture_img=img[:, :, 0],
                                                                          std_out=None, toggle_yolo=False)



    print("Start")
    # Show the rack and tubes
    rack_cm = cm.CollisionModel(initor=tuberack.model_path, expand_radius=0.001)
    rack_cm.set_homomat(tuberack_trans)
    rack_cm.set_rgba(tuberack.color)
    # rack_cm.show_cdprimit()
    rack_cm.attach_to(base)

    # Task Planning
    GOAL_PATTERN = np.array([[2, 2, 2, 2, 2, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [2, 2, 2, 2, 2, 0, 0, 0, 0, 0],
                             [2, 2, 2, 2, 2, 0, 0, 0, 0, 0]])
    # print goal pattern
    print_with_border("Goal pattern", width=38)
    print(text_pd(GOAL_PATTERN))
    ## Task planning init
    tuberack_bk = tuberack.copy()
    motions = MotionBatch()
    weight_matrix = np.zeros_like(tuberack.rack_status)
    weight_matrix_bk = weight_matrix.copy()
    planning_finished_flag = False
    taskplanning_restart_flag = False
    movemap = np.zeros_like(tuberack.rack_status)
    print("---" * 20)
    # print_with_border("START PLANING", width=70)
    if is_done(state=tuberack.rack_status, goal_pattern=GOAL_PATTERN):
        print_with_border("All test tubes have already been arranged", width=70)
    # yumi_robot.offline_data = fs.load_pickle("ik_offline_db")
    st = time.time()
    while True:
        print_with_border("START PLANING", width=70)
        print_with_border("Weight matrix", width=38)
        print(text_pd(weight_matrix))
        print_with_border("Rack State matrix", width=38)
        print(text_pd(tuberack.rack_status))
        # check if everything test tube is arranged

        tuberacksolution = ArrangeTubeSolSeq(rack=tuberack,
                                             solution=tuberack.solve(GOAL_PATTERN, weight_matrix=weight_matrix),
                                             tf=tuberack_trans)

        # planning motions
        for move_tube_info, obs_tubes_info in tuberacksolution:
            # obs test tubes
            obs_testubes_list = []
            for obs_tube_info in obs_tubes_info:
                obs_tube = obs_tube_info["tube_type"]
                tube_mdl = cm.CollisionModel(obs_tube.model_path, cdprimit_type="cylinder", expand_radius=0.001)
                tube_mdl.set_homomat(obs_tube_info["tube_pose"])
                tube_mdl.set_rgba([0, 1, 0, 1])
                obs_testubes_list.append(tube_mdl)
            #     tube_mdl.show_cdprimit()
            #     tube_mdl.attach_to(base)
            # base.run()
            move_tube_type = move_tube_info["tube_type"]
            move_tube_init_pos = move_tube_info["init_pos"]
            move_tube_goal_pos = move_tube_info["goal_pos"]
            move_tube_init_coord = move_tube_info["init_coord"]
            move_tube_goal_coord = move_tube_info["goal_coord"]

            # pick and place
            hand_name = 'rgt_hnd'
            yumi_robot.rgt_arm.goto_homeconf()
            start_conf = yumi_robot.get_jnt_values(hand_name)
            goal_homomat_list = [move_tube_init_pos, move_tube_goal_pos]
            move_objcm_tmp, grasp_info_list = get_objcm_grasps(tube=move_tube_type)
            move_objcm_tmp.set_homomat(move_tube_init_pos)
            move_objcm_tmp.set_rgba(move_tube_type.color)
            print_with_border(f"Moving: the test tube is {move_tube_type}"
                              f"         init coordinate is {move_tube_init_coord}"
                              f"              goal coordinate is {move_tube_goal_coord}"
                              f"      Number of grasps: {len(grasp_info_list)}", width=38)
            conf_list_tmp, jawwidth_list_tmp, objpose_list_tmp = \
                pp_planner.gen_pick_and_place_motion(hnd_name=hand_name,
                                                     objcm=move_objcm_tmp,
                                                     grasp_info_list=grasp_info_list,
                                                     goal_homomat_list=goal_homomat_list,
                                                     start_conf=start_conf,
                                                     end_conf=start_conf,
                                                     depart_direction_list=[tuberack_trans[:3, 2]] * len(
                                                         goal_homomat_list),
                                                     approach_direction_list=[-tuberack_trans[:3, 2]] * len(
                                                         goal_homomat_list),
                                                     depart_distance_list=[.1] * len(goal_homomat_list),
                                                     approach_distance_list=[.1] * len(goal_homomat_list),
                                                     approach_jawwidth=None,
                                                     depart_jawwidth=None,
                                                     ad_granularity=.003,
                                                     use_rrt=False,
                                                     obstacle_list=obs_testubes_list,
                                                     grasp_obstacle_list=[rack_cm],
                                                     use_incremental=False, )
            if conf_list_tmp is not None:
                print("---" * 17)
                print("Successfully!")
                # reset the pose of the obstacle
                # add motions into motion batch
                for i in range(len(conf_list_tmp)):
                    motions.append(MotionElement(obj_cm=move_objcm_tmp.copy(),
                                                 objpose_list=objpose_list_tmp[i],
                                                 conf_list=conf_list_tmp[i],
                                                 jawwidth_list=jawwidth_list_tmp[i],
                                                 hnd_name=hand_name,
                                                 obs_list=obs_testubes_list))
                # # animation
                # show_animation(yumi_robot=yumi_robot,
                #                motion_batch=motions,
                #                base=base)
                # base.run()
            else:
                print("---" * 17)
                print("Planning Failed!! Restart Search a New Path")
                # planning failed
                weight_matrix[tuple(move_tube_init_coord)] = 1
                weight_matrix[tuple(move_tube_goal_coord)] = 1
                motions = MotionBatch()
                break
        else:
            planning_finished_flag = True

        if planning_finished_flag:
            print("Successfully planning all!")
            break

    ed = time.time()
    print("Total time consuming is:", ed - st)
    show_animation(yumi_robot=yumi_robot,
                   motion_batch=motions,
                   base=base)

    # base.run()
    exit(0)
