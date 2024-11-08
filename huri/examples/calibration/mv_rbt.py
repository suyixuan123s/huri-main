if __name__ == "__main__":
    from huri.core.common_import import *
    from huri.components.yumi_control.yumi_con import YumiController
    from huri.vision.phoxi_capture import SensorMarkerHandler, vision_pipeline
    import time
    from huri.examples.calibration.calibration_constants import CALIB_MAT_PATH

    REAL_ROBOT = True
    DEBUG = False

    # init
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    yumi_s = ym.Yumi(enable_cc=True)

    # get robot pos
    if REAL_ROBOT:
        from huri.components.yumi_control.yumi_con import YumiController

        yumi_x = YumiController()
        curr_rgt_jnt = yumi_x.get_jnt_values(component_name="rgt_arm")
        yumi_s.fk(component_name="rgt_arm", jnt_values=curr_rgt_jnt)
        rgt_pos, rgt_rot = yumi_s.get_gl_tcp("rgt_arm")
        target_rgt_jnt = yumi_s.ik(component_name="rgt_arm", tgt_pos=rgt_pos, tgt_rotmat=np.round(rgt_rot),
                                   seed_jnt_values=curr_rgt_jnt, max_niter=1000)
        yumi_s.fk(component_name="rgt_arm", jnt_values=target_rgt_jnt)

        curr_lft_jnt = yumi_x.get_jnt_values(component_name="lft_arm")
        yumi_s.fk(component_name="lft_arm", jnt_values=curr_lft_jnt)
        lft_pos, lft_rot = yumi_s.get_gl_tcp("lft_arm")
        # target_lft_jnt = yumi_s.ik(component_name="lft_arm", tgt_pos=lft_pos, tgt_rotmat=np.round(lft_rot),
        #                            seed_jnt_values=curr_lft_jnt, max_niter=1000)
        # yumi_s.fk(component_name="lft_arm", jnt_values=target_lft_jnt)
        print(curr_rgt_jnt)
        print(curr_lft_jnt)

        print(f"jnt_angle after refinement: {target_rgt_jnt}")

        yumi_s.jaw_to("rgt_hnd", 0.0)
    # Get image from sensor
    if DEBUG:
        filename = fs.workdir / "data" / "vision_exp" / "20210920-164042.pkl"
        pcd, img = fs.load_pickle(filename)
    else:
        pcd, img = vision_pipeline(SensorMarkerHandler(),
                                   dump_path=fs.workdir / "data" / "vision_exp" / f"{time.strftime('%Y%m%d-%H%M%S')}.pkl")

    affine_mat = np.asarray(fs.load_json(CALIB_MAT_PATH)['affine_mat'])
    transformed_pcd = rm.homomat_transform_points(affine_mat, points=pcd)
    gm.gen_pointcloud(transformed_pcd, ).attach_to(base)
    # generate the model of the robot
    yumi_s.gen_meshmodel().attach_to(base)

    base.run()
