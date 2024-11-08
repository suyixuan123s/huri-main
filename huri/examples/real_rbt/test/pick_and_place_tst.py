if __name__ == "__main__":
    from huri.vision.phoxi_capture import vision_pipeline, SensorMarkerHandler
    from huri.core.file_sys import workdir, load_pickle, load_json
    from time import strftime
    from huri.components.vision.tube_detector import TestTubeDetector
    from huri.core.common_import import *
    from huri.math.math import combination
    from huri.learning.env.arrangement_planning_rack.env import RackState
    from huri.math.fkopt_based_ik import FKOptBasedIK_C, FKOptBasedIK
    from huri.definitions.tube_def import TubeType
    import motion.optimization_based.incremental_nik as inik
    import motion.probabilistic.rrt_connect as rrtc

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

    DEBUG = True
    IP_ADR = "192.168.125.100:18300"
    CALIB_MAT_PATH = workdir / "data/calibration/qaqqq.json"

    # use the offline vision data for debug
    if DEBUG:
        filename = workdir / "data" / "vision_exp" / "20220105-200242.pkl"
        pcd, img = load_pickle(filename)
    else:
        pcd, img = vision_pipeline(SensorMarkerHandler(ip_adr=IP_ADR),
                                   workdir / "data" / "vision_exp" / f"{strftime('%Y%m%d-%H%M%S')}.pkl")

    # recognize the current states of the test tubes and test tube rack
    detector = TestTubeDetector(affine_mat_path=CALIB_MAT_PATH)
    detected_test_tubes, tube_rack, rack_tf = detector.analyze_scene(pcd=pcd, texture_img=img[:, :, 0],
                                                                     std_out=None, toggle_yolo=False, save_detect=True)
    rack_state = RackState(tube_rack.rack_status)

    tube_rack_cm = tube_rack.gen_mesh_model()
    tube_rack_cm.attach_to(base)
    fillable_slot_ids = np.vstack(np.where(rack_state.fillable_slots > 0)).T
    random_slot_id = fillable_slot_ids[[np.random.choice(np.arange(len(fillable_slot_ids)))]].flatten()

    op_tube_type = 1
    op_tube = TubeType.gen_tube_by_tubetype(op_tube_type)
    op_tube_id = np.vstack(np.where(rack_state == 1)).T[0]

    # init pose
    op_tube_homomat_init = tube_rack.get_slot_homomat(op_tube_id)
    op_tube_gm_init = op_tube.gen_mesh_model()
    op_tube_gm_init.set_homomat(op_tube_homomat_init)
    op_tube_gm_init.set_rgba([1, 0, 0, .7])

    # target pose
    op_tube_homomat_tgt = tube_rack.get_slot_homomat(random_slot_id)
    op_tube_gm_tgt = op_tube.gen_mesh_model()
    op_tube_gm_tgt.set_homomat(op_tube_homomat_tgt)
    op_tube_gm_tgt.set_rgba([0, 1, 0, .7])
    symmetry_homomat_tgt = op_tube.get_symmetry_homomat(discrete_factor=10, rot_mat=op_tube_homomat_tgt)

    grasps_list_info = fs.load_pickle(op_tube.grasps_path)

    # generate the yumi robot
    ym_rbt = ym.Yumi(enable_cc=True)
    component_name = "rgt_arm"
    # initialize the module to generate the linear motion
    inik_svlr = inik.IncrementalNIK(robot_s=ym_rbt)
    # initialize the module for RRT
    rrtc_planner = rrtc.RRTConnect(ym_rbt)

    # the list to save the pick-and-place path
    path = []

    # search the grasp that can move the object to target pose
    for ind, (jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat) in enumerate(grasps_list_info):
        print(f"--------------------- grasp pose index: {ind} ---------------------------")
        grasp_pose = rm.homomat_from_posrot(jaw_center_pos, jaw_center_rotmat)
        print(f"the homogenous matrix of the grasp pose is: {grasp_pose}")
        # find the possible ik set for the init pose
        rbt_ee_poses_init = np.dot(symmetry_homomat_init.reshape((-1, 4)), grasp_pose).reshape((-1, 4, 4))
        ik_sol_set_init = []
        for _rbt_ee_pose_init in rbt_ee_poses_init:
            ik_sol_init = ym_rbt.ik(component_name, tgt_pos=_rbt_ee_pose_init[:3, 3],
                                    tgt_rotmat=_rbt_ee_pose_init[:3, :3])
            if ik_sol_init is not None:
                ik_sol_set_init.append(ik_sol_init)
        # find the possible ik set for the goal pose
        rbt_ee_poses_tgt = np.dot(symmetry_homomat_tgt.reshape((-1, 4)), grasp_pose).reshape((-1, 4, 4))
        ik_sol_set_tgt = []
        for _rbt_ee_pose_tgt in rbt_ee_poses_tgt:
            ik_sol_tgt = ym_rbt.ik(component_name, tgt_pos=_rbt_ee_pose_tgt[:3, 3], tgt_rotmat=_rbt_ee_pose_tgt[:3, :3])
            if ik_sol_tgt is not None:
                ik_sol_set_tgt.append(ik_sol_tgt)
        if len(ik_sol_set_init) > 0 and len(ik_sol_set_tgt) > 0:
            # check collision
            for _ik_sol_init, _ik_sol_tgt in product(ik_sol_set_init, ik_sol_set_tgt):
                ym_rbt.fk(component_name, _ik_sol_init)
                is_collided_init = ym_rbt.is_collided([obs_rack_1, obs_rack_2, *obs_tube_collection.cm_list])
                ym_rbt.fk(component_name, _ik_sol_tgt)
                is_collided_tgt = ym_rbt.is_collided([obs_rack_1, obs_rack_2, *obs_tube_collection.cm_list])
                if is_collided_init or is_collided_tgt:  # check if is self-collided
                    print(">>> The robot is collided")
                    continue
                else:
                    # generate the pick motion
                    # move the arm to the init pose first
                    ym_rbt.fk(component_name, _ik_sol_init)
                    # get ee pose
                    rbt_tcp_pos, rbt_tcp_rot = ym_rbt.get_gl_tcp(component_name)
                    # grasp the object at init pose for collision detection only
                    obj_mdl_grasped = tube_cm_mdl.copy()
                    obj_mdl_grasped.set_homomat(symmetry_homomat_init[0])
                    ym_rbt.hold(hnd_name="rgt_hnd", objcm=obj_mdl_grasped, jaw_width=jaw_width)
                    path_up = inik_svlr.gen_linear_motion(component_name,
                                                          start_tcp_pos=rbt_tcp_pos,
                                                          start_tcp_rotmat=rbt_tcp_rot,
                                                          goal_tcp_pos=rbt_tcp_pos + np.array([0, 0, .15]),
                                                          goal_tcp_rotmat=rbt_tcp_rot,
                                                          obstacle_list=[*obs_tube_collection.cm_list, obs_rack_2],
                                                          granularity=0.01)
                    ym_rbt.release(hnd_name="rgt_hnd", objcm=obj_mdl_grasped)

                    # generate the place motion
                    # move the arm to the target pose first
                    ym_rbt.fk(component_name, _ik_sol_tgt)
                    # get ee pose
                    rbt_tcp_pos, rbt_tcp_rot = ym_rbt.get_gl_tcp(component_name)
                    # grasp the object at init pose for collision detection only
                    obj_mdl_grasped.set_homomat(symmetry_homomat_tgt[0])
                    ym_rbt.hold(hnd_name="rgt_hnd", objcm=obj_mdl_grasped, jaw_width=jaw_width)
                    path_down_reverse = inik_svlr.gen_linear_motion(component_name,
                                                                    start_tcp_pos=rbt_tcp_pos,
                                                                    start_tcp_rotmat=rbt_tcp_rot,
                                                                    goal_tcp_pos=rbt_tcp_pos + np.array([0, 0, .15]),
                                                                    goal_tcp_rotmat=rbt_tcp_rot,
                                                                    obstacle_list=[*obs_tube_collection.cm_list,
                                                                                   obs_rack_1],
                                                                    granularity=0.01)
                    if path_up is not None and path_down_reverse is not None:
                        path_down = path_down_reverse[:: -1]
                        rrt_path = rrtc_planner.plan(component_name=component_name,
                                                     start_conf=np.array(path_up[-1]),
                                                     goal_conf=np.array(path_down[0]),
                                                     obstacle_list=[obs_rack_1, obs_rack_2,
                                                                    *obs_tube_collection.cm_list],
                                                     ext_dist=.05,
                                                     max_time=300)
                        if rrt_path is not None:
                            # insert the tube
                            rack_2.insert_tube(slot_id=insert_ind, tube=tube_move)
                            # show the path
                            path = path_up + rrt_path + path_down
                            for jnts_s in path:
                                rack_1.gen_mesh_model(gen_tube=True).attach_to(base)
                                rack_2.gen_mesh_model(gen_tube=True).attach_to(base)
                                ym_rbt.fk(component_name, jnts_s)
                                ym_rbt.gen_meshmodel().attach_to(base)
                                obj_mdl_draw = obj_mdl_grasped.copy()
                                obj_mdl_draw.attach_to(base)
                            base.run()
                    else:
                        print(">>> Cannot generate the path to move the object to the target pose by RRT")
                        ym_rbt.release("rgt_hnd", objcm=obj_mdl_grasped)
            else:
                print(">>> Cannot generate the path to raise the tube")
                ym_rbt.release("rgt_hnd", objcm=obj_mdl_grasped)
        else:
            print("No IK solution at init or target")

    base.run()
