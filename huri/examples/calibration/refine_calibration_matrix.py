import huri.core.utils as hcu
from extract_pcd_feature import extract_pcd
import huri.vision.pnt_utils as pntu

color_group = [hcu.color_hex2oct(color) for color in hcu.color_hex["Beach Towels"]]
if __name__ == "__main__":
    from huri.core.common_import import *
    from huri.components.yumi_control.yumi_con import YumiController
    from huri.vision.phoxi_capture import SensorMarkerHandler, vision_pipeline
    import huri.examples.calibration.gen_robot_model_pcd as grmp
    import time
    from huri.examples.calibration.calibration_constants import CALIB_MAT_PATH
    from huri.core.constants import SENSOR_INFO

    REAL_ROBOT = False
    DEBUG = False

    # init
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    yumi_s = ym.Yumi(enable_cc=True)

    # get robot pos
    if REAL_ROBOT:
        from huri.components.yumi_control.yumi_con import YumiController

        yumi_x = YumiController()
        yumi_x.move_jnts(component_name="rgt_arm",
                         jnt_vals=[1.44841894, -1.64704077, -1.23590585, 0.26502936, -0.7216021, 0.98803833,
                                   -3.4581817])
        yumi_x.move_jnts(component_name="lft_arm",
                         jnt_vals=[-1.09088526, -1.62867106, 1.12936082, 0.71374227, 0.60395949, 0.91902467,
                                   0.18455521])
        yumi_x.close_gripper("rgt_arm")
        yumi_x.close_gripper("lft_arm")
        yumi_s.fk(component_name="rgt_arm",
                  jnt_values=yumi_x.get_jnt_values(component_name="rgt_arm"))
        yumi_s.fk(component_name="lft_arm",
                  jnt_values=yumi_x.get_jnt_values(component_name="lft_arm"))
        yumi_s.jaw_to("rgt_hnd", 0.0)
    # Get image from sensor
    if DEBUG:
        filename = fs.workdir / "data" / "vision_exp" / "20210920-164042.pkl"
        pcd, img, depth_img, _, _ = fs.load_pickle(filename)
    else:
        pcd, img, depth_img, _, _ = vision_pipeline(SensorMarkerHandler(SENSOR_INFO.IP_ADR_DEPTH_SENSOR_DEBUG),
                                                    dump_path=fs.workdir / "data" / "vision_exp" / f"{time.strftime('%Y%m%d-%H%M%S')}.pkl",
                                                    rgb_texture=False
                                                    )

    # generate the model of the robot
    yumi_s.gen_meshmodel().attach_to(base)
    # sample the points of robot
    # rgt_arm_points = grmp.sample_arm(yumi_s, "rgt_arm", radius=0.003)
    # lft_arm_points = grmp.sample_arm(yumi_s, "lft_arm", radius=0.003)
    rgt_hnd_points = grmp.sample_hnd(yumi_s, "rgt_hnd", radius=0.002, contain_fingers=False)
    lft_hnd_points = grmp.sample_hnd(yumi_s, "lft_hnd", radius=0.002, contain_fingers=False)
    # body_points = grmp.sample_body(yumi_s,radius=.004)
    body_points = fs.load_json(fs.workdir / "data" / "calibration" / "body_points.json")
    body_points_extracted_lft_leg = extract_pcd(body_points, start_coord=np.array([0.095, 0.05, -0.03]),
                                                extend=np.array([0.055, 0.18, 0.1]))
    body_points_extracted_rgt_leg = extract_pcd(body_points, start_coord=np.array([0.095, -0.23, -0.03]),
                                                extend=np.array([0.055, 0.18, 0.1]))
    sampled_points = grmp.concatenate_list(
        (rgt_hnd_points, lft_hnd_points, body_points_extracted_lft_leg, body_points_extracted_rgt_leg))
    print("sample_generated_finished")
    print("start fitting")
    # gm.gen_pointcloud(points=sampled_points, rgbas=[color_group[0]]).attach_to(base)
    # point cloud captured from depth sensor and multiply with the affine matrix
    affine_mat = np.asarray(fs.load_json(CALIB_MAT_PATH)['affine_mat'])
    print("affine_mat is ", affine_mat)
    print(affine_mat[:3, :3])
    print(np.linalg.qr(affine_mat[:3, :3])[0].T)
    affine_mat[:3, 0] = rm.unit_vector(affine_mat[:3, 0])
    affine_mat[:3, 1] = rm.unit_vector(affine_mat[:3, 1])
    affine_mat[:3, 2] = rm.unit_vector(affine_mat[:3, 2])
    print(np.linalg.norm(affine_mat[:, 0]))
    print(np.linalg.norm(affine_mat[:, 1]))
    print(np.linalg.norm(affine_mat[:, 2]))
    transformed_pcd = rm.homomat_transform_points(affine_mat, points=pcd)
    # gm.gen_pointcloud(transformed_pcd, [color_group[1]]).attach_to(base)

    # transform_inv = pntu.icp(src=sampled_points, tgt=transformed_pcd, maximum_distance=0.005)
    # transform = np.linalg.inv(transform_inv)
    print("ICP Faied")
    transform_1 = pntu.icp(src=transformed_pcd, tgt=sampled_points, maximum_distance=0.005, relative_rmse=1e-15,
                           relative_fitness=1e-15, max_iteration=2000)
    pcd_refined = rm.homomat_transform_points(transform_1, points=transformed_pcd)
    # gm.gen_pointcloud(pcd_refined, [color_group[2]]).attach_to(base)

    # ------
    # transform_inv = pntu.icp(src=sampled_points, tgt=pcd_refined, maximum_distance=0.002, relative_rmse=1e-15,
    #                          relative_fitness=1e-15, max_iteration=2000)
    # transform = np.linalg.inv(transform_inv)

    transform_2 = pntu.icp(src=pcd_refined, tgt=sampled_points, maximum_distance=0.002, relative_rmse=1e-15,
                           relative_fitness=1e-15, max_iteration=2000)

    pcd_refined = rm.homomat_transform_points(transform_2, points=pcd_refined)
    gm.gen_pointcloud(pcd_refined, [color_group[2]]).attach_to(base)

    refined_matrix = np.dot(transform_2, np.dot(transform_1, affine_mat))
    fs.dump_json({'affine_mat': refined_matrix.tolist()},
                 CALIB_MAT_PATH.parent / f"{CALIB_MAT_PATH.name.split('.')[0]}_r.json")

    base.run()
