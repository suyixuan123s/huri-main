if __name__ == "__main__":
    from huri.vision.phoxi_capture import vision_pipeline, SensorMarkerHandler
    from huri.core.file_sys import workdir, load_pickle
    from time import strftime
    from huri.components.vision.tube_detector import TestTubeDetector
    from huri.core.common_import import *
    from huri.learning.env.arrangement_planning_rack.env import RackState
    from huri.definitions.tube_def import TubeType
    import huri.components.planning.symmetric_pick_and_place_planner as ppp
    import basis.trimesh as tm
    from huri.definitions.utils_structure import MotionElement, MotionBatch
    from huri.components.debug.error_feedback import RecaptureFeedbackMoveSim

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

    # setup the collision model for phoxi camera
    camera_obs = cm.CollisionModel(tm.primitives.Box(box_extents=[1, 1, .2]))
    camera_obs.attach_to(base)
    camera_obs.set_pos(np.array([0.1, 0, 1]))

    DEBUG = False
    IP_ADR = "192.168.125.100:18300"
    CALIB_MAT_PATH = workdir / "data/calibration/qaqqq.json"

    num_of_trial = 100
    detector = TestTubeDetector(affine_mat_path=CALIB_MAT_PATH)

    if not DEBUG:
        # real exe
        from huri.components.yumi_control.yumi_con import YumiController
        from huri.math.units import Mm
        from huri.components.pipeline.data_pipeline import motion_executer

        open_gripper_len = Mm(35)
        try:
            yumi_con = YumiController()
        except:
            yumi_con = YumiController()
        yumi_con.set_gripper_width(component_name="rgt_arm", width=open_gripper_len)
        yumi_con.set_gripper_speed("rgt_arm", 10)

    for __ in range(num_of_trial):
        # use the offline vision data for debug
        if DEBUG:
            filename = workdir / "data" / "vision_exp" / "20220110-131135.pkl"
            pcd, img = load_pickle(filename)
        else:
            pcd, img = vision_pipeline(SensorMarkerHandler(ip_adr=IP_ADR),
                                       workdir / "data" / "vision_exp" / f"{strftime('%Y%m%d-%H%M%S')}.pkl")

        # recognize the current states of the test tubes and test tube rack

        detected_test_tubes, tube_rack, rack_tf = detector.analyze_scene(pcd=pcd, texture_img=img[:, :, 0],
                                                                         std_out=None, toggle_yolo=False,
                                                                         save_detect=True)
        rack_state = RackState(tube_rack.rack_status)

        tube_rack_cm = tube_rack.gen_collision_model()
        tube_rack_cm.attach_to(base)

        op_tube_type = 3
        op_tube = TubeType.gen_tube_by_tubetype(op_tube_type)
        op_tube_id = np.vstack(np.where(rack_state.state == op_tube_type)).T[0]

        # init pose
        op_tube_type, op_tube_homomat_init = tube_rack.remove_slot(op_tube_id)
        op_tube_gm_init = op_tube.gen_mesh_model()
        op_tube_gm_init.set_homomat(op_tube_homomat_init)
        op_tube_gm_init.set_rgba([1, 0, 0, .7])
        obs_rack_cm, obs_tube_collection = tube_rack.gen_collision_model(gen_tube=True)

        # target pose
        fillable_slot_ids = np.vstack(np.where(rack_state.fillable_slots > 0)).T
        random_slot_id = fillable_slot_ids[[np.random.choice(np.arange(len(fillable_slot_ids)))]].flatten()
        op_tube_homomat_tgt = tube_rack.get_slot_homomat(random_slot_id)
        op_tube_cm_tgt = op_tube.gen_collision_model()
        op_tube_cm_tgt.set_homomat(op_tube_homomat_tgt.copy())
        goal_place_offset_dis = .04
        op_tube_homomat_tgt[:3, 3] = op_tube_homomat_tgt[:3, 3] + tube_rack.get_homomat()[:3, 2] * goal_place_offset_dis
        op_tube_gm_tgt = op_tube.gen_mesh_model()
        op_tube_gm_tgt.set_homomat(op_tube_homomat_tgt)
        op_tube_gm_tgt.set_rgba([0, 1, 0, .7])
        symmetry_homomat_tgt = op_tube.get_symmetry_homomat(discrete_factor=10, rot_mat=op_tube_homomat_tgt)

        grasps_list_info = fs.load_json(op_tube.grasps_path)

        # generate the yumi robot
        yumi_robot = ym.Yumi(enable_cc=True)
        component_name = "rgt_hnd"
        rfms = RecaptureFeedbackMoveSim(robot_s=yumi_robot,
                                        component_name=component_name)
        yumi_robot_arm = yumi_robot.rgt_arm if "rgt" in component_name else yumi_robot.lft_arm
        start_conf = yumi_robot_arm.homeconf
        # init the pick and place planner
        pp_planner = ppp.PickPlacePlanner(robot_s=yumi_robot)

        conf_list_tmp, jawwidth_list_tmp, objpose_list_tmp, init_feasible_iks = \
            pp_planner.gen_pick_and_place_motion(hnd_name=component_name,
                                                 tube=op_tube,
                                                 tube_cm=op_tube.gen_collision_model(),
                                                 grasp_info_list=grasps_list_info,
                                                 goal_homomat_list=[op_tube_homomat_init, op_tube_homomat_tgt],
                                                 start_conf=start_conf,
                                                 end_conf=None,
                                                 depart_direction_list=[rack_tf[:3, 2]] * 2,
                                                 approach_direction_list=[-rack_tf[:3, 2]] * 2,
                                                 depart_distance_list=[.16, .16 - goal_place_offset_dis],
                                                 approach_distance_list=[.16, .16 - goal_place_offset_dis],
                                                 approach_jawwidth=None,
                                                 depart_jawwidth=None,
                                                 ad_granularity=.02,
                                                 use_rrt=False,
                                                 obstacle_list=obs_tube_collection.cm_list + [camera_obs],
                                                 grasp_obstacle_list=[obs_rack_cm],
                                                 use_incremental=False, )
        if conf_list_tmp is not None:
            print("---" * 17)
            print("Successfully!")
            motion_batch = MotionBatch()
            for i in range(len(conf_list_tmp)):
                motion_batch.append(MotionElement(obj_cm=op_tube.gen_collision_model(),
                                                  objpose_list=objpose_list_tmp[i],
                                                  conf_list=conf_list_tmp[i],
                                                  jawwidth_list=jawwidth_list_tmp[i],
                                                  hnd_name=component_name,
                                                  obs_list=obs_tube_collection.cm_list))
            rfms.set_obs_list(obs_tube_collection.cm_list + [camera_obs, op_tube_cm_tgt])
            seed_jnt = motion_batch.get_last_motion_rbt_jnt(component_name=component_name)
            rfms.solve(seed_jnt_values=seed_jnt)
            if DEBUG:
                from huri.test.app.animation import show_animation

                show_animation(yumi_robot=yumi_robot,
                               motion_batch=motion_batch,
                               base=base, yumi_con=None)
            else:
                motion_executer(yumi_con=yumi_con, motion_batch=motion_batch, open_gripper_len=open_gripper_len)

        # base.run()
