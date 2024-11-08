if __name__ == "__main__":
    from huri.vision.phoxi_capture import vision_pipeline, SensorMarkerHandler
    from huri.core.file_sys import workdir, load_pickle, load_json
    from time import strftime
    from huri.components.vision.tube_detector import TestTubeDetector
    from huri.core.common_import import *
    from huri.math.math import combination
    from huri.math.fkopt_based_ik import FKOptBasedIK_C, FKOptBasedIK

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

    DEBUG = False
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
    tube_rack_cm = tube_rack.gen_mesh_model()
    tube_rack_cm.attach_to(base)
    slot_ids = combination([np.arange(1,5), np.arange(10)])
    from huri.components.yumi_control.yumi_con import YumiController

    yumi_robot = ym.Yumi(enable_cc=True)
    yumi_x = YumiController(debug=False)
    yumi_x.set_gripper_width(component_name="rgt_arm", width=.01)
    component_name = "rgt_arm"

    # FKOptBasedIK_C
    oik = FKOptBasedIK_C(yumi_robot,
                         obstacle_list=[tube_rack_cm],
                         component_name=component_name,
                         toggle_debug=False)
    seed_jnt_value = yumi_robot.get_jnt_values(component_name=component_name)

    for i in slot_ids:
        slot_homomat = tube_rack.get_slot_homomat(i)
        slot_homomat_above = slot_homomat.copy()
        slot_homomat_above[:3, 3] = slot_homomat_above[:3, 3] + slot_homomat_above[:3, 2] * .07

        robot_hnd_rotmat = slot_homomat_above[:3, :3].copy()
        robot_hnd_rotmat[:3, 2] = -robot_hnd_rotmat[:3, 2]
        gm.gen_frame(pos=slot_homomat_above[:3, 3], rotmat=robot_hnd_rotmat[:3, :3]).attach_to(base)
        # print(yumi_robot.ik(component_name, tgt_pos=slot_homomat_above[:3, 3], tgt_rotmat=robot_hnd_rotmat))
        ikv = oik.solve(tgt_pos=slot_homomat_above[:3, 3],
                        tgt_rotmat=robot_hnd_rotmat[:3, :3],
                        seed_jnt_values=seed_jnt_value)
        if ikv[0] is None:
            print("IKV is none")
            continue
        yumi_robot.fk(component_name=component_name, jnt_values=np.array(ikv[0]))
        # seed_jnt_value = ikv[0]
        yumi_robot.gen_meshmodel().attach_to(base)
        yumi_x.move_jnts(component_name=component_name, jnt_vals=ikv[0], speed_n=200)
        input("")
    base.run()
