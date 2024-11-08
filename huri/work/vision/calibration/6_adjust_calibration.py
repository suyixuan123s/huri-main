import numpy as np

from huri.core.common_import import *
from huri.vision.phoxi_capture import SensorMarkerHandler, vision_pipeline

REAL_ROBOT = False
AFFINE_MAT_PATH = fs.workdir / "data" / "calibration" / "qaqqq.json"
pcd_trimesh_boundingbox = [None]


def test(pcd, base):
    # load affine mat
    # Transform and Plot Point Clouds
    affine_mat = w2r_mat.dot(r2cam_mat)
    pcd_r = rm.homomat_transform_points(affine_mat, points=pcd)

    affine_mat_node = [r2cam_mat]
    pcd_node = [gm.gen_pointcloud(pcd_r, pcd_color_rgba)]
    pcd_node[0].attach_to(base)
    angle_resolution = .1
    rgba = [1, 0, 0, .3]
    if mat is not None:
        marker_frame = w2r_mat.dot(r2cam_mat).dot(mat)
        gm.gen_frame(marker_frame[:3, 3], marker_frame[:3, :3]).attach_to(pcd_node[0])

    # gm.gen_sphere(w2c_mat[:3, 3],radius=.005).attach_to(pcd_node[0])
    # for _ in [results[3], results[4], results[5]]:
    #     _ = np.dot(w2r_mat.dot(r2cam_mat),(_))
    #     gm.gen_frame(_[:3, 3], _[:3, :3]).attach_to(base)

    def adjust_pcd(affine_mat_node, pcd_node_node, pcd, task):
        if base.inputmgr.keymap["a"]:
            affine_mat_node[0][:3, 3] = affine_mat_node[0][:3, 3] + np.array([0, -.0002, 0])
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.homomat_transform_points(w2r_mat.dot(r2cam_mat), points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, pcd_color_rgba)
            pcd_node_node[0].attach_to(base)
            if mat is not None:
                marker_frame = w2r_mat.dot(r2cam_mat).dot(mat)
                gm.gen_frame(marker_frame[:3, 3], marker_frame[:3, :3]).attach_to(pcd_node_node[0])
            w2c_mat = w2r_mat.dot(r2cam_mat)
            gm.gen_sphere(w2c_mat[:3, 3], radius=.005).attach_to(pcd_node[0])
            print(repr(affine_mat_node[0]))
        elif base.inputmgr.keymap["d"]:
            affine_mat_node[0][:3, 3] = affine_mat_node[0][:3, 3] + np.array([0, .0005, 0])
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.homomat_transform_points(w2r_mat.dot(r2cam_mat), points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, pcd_color_rgba)
            pcd_node_node[0].attach_to(base)
            if mat is not None:
                marker_frame = w2r_mat.dot(r2cam_mat).dot(mat)
                gm.gen_frame(marker_frame[:3, 3], marker_frame[:3, :3]).attach_to(pcd_node_node[0])
            w2c_mat = w2r_mat.dot(r2cam_mat)
            gm.gen_sphere(w2c_mat[:3, 3], radius=.005).attach_to(pcd_node[0])
            print(repr(affine_mat_node[0]))
        elif base.inputmgr.keymap["w"]:
            affine_mat_node[0][:3, 3] = affine_mat_node[0][:3, 3] + np.array([-.0002, 0, 0])
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.homomat_transform_points(w2r_mat.dot(r2cam_mat), points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, pcd_color_rgba)
            pcd_node_node[0].attach_to(base)
            if mat is not None:
                marker_frame = w2r_mat.dot(r2cam_mat).dot(mat)
                gm.gen_frame(marker_frame[:3, 3], marker_frame[:3, :3]).attach_to(pcd_node_node[0])
            w2c_mat = w2r_mat.dot(r2cam_mat)
            gm.gen_sphere(w2c_mat[:3, 3], radius=.005).attach_to(pcd_node[0])
            print(repr(affine_mat_node[0]))
        elif base.inputmgr.keymap["s"]:
            affine_mat_node[0][:3, 3] = affine_mat_node[0][:3, 3] + np.array([.0005, 0, 0])
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.homomat_transform_points(w2r_mat.dot(r2cam_mat), points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, pcd_color_rgba)
            pcd_node_node[0].attach_to(base)
            if mat is not None:
                marker_frame = w2r_mat.dot(r2cam_mat).dot(mat)
                gm.gen_frame(marker_frame[:3, 3], marker_frame[:3, :3]).attach_to(pcd_node_node[0])
            w2c_mat = w2r_mat.dot(r2cam_mat)
            gm.gen_sphere(w2c_mat[:3, 3], radius=.005).attach_to(pcd_node[0])
            print(repr(affine_mat_node[0]))
        elif base.inputmgr.keymap["q"]:
            affine_mat_node[0][:3, 3] = affine_mat_node[0][:3, 3] + np.array([0, 0, .0005])
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.homomat_transform_points(w2r_mat.dot(r2cam_mat), points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, pcd_color_rgba)
            pcd_node_node[0].attach_to(base)
            if mat is not None:
                marker_frame = w2r_mat.dot(r2cam_mat).dot(mat)
                gm.gen_frame(marker_frame[:3, 3], marker_frame[:3, :3]).attach_to(pcd_node_node[0])
            print(repr(affine_mat_node[0]))
        elif base.inputmgr.keymap["e"]:
            affine_mat_node[0][:3, 3] = affine_mat_node[0][:3, 3] + np.array([0, 0, -.0002])
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.homomat_transform_points(w2r_mat.dot(r2cam_mat), points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, pcd_color_rgba)
            pcd_node_node[0].attach_to(base)
            if mat is not None:
                marker_frame = w2r_mat.dot(r2cam_mat).dot(mat)
                gm.gen_frame(marker_frame[:3, 3], marker_frame[:3, :3]).attach_to(pcd_node_node[0])
            print(repr(affine_mat_node[0]))
        elif base.inputmgr.keymap["z"]:
            affine_mat_node[0][:3, :3] = np.dot(
                rm.rotmat_from_axangle(np.array([1, 0, 0]), -np.radians(angle_resolution)),
                affine_mat_node[0][:3, :3])
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.homomat_transform_points(w2r_mat.dot(r2cam_mat), points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, pcd_color_rgba)
            pcd_node_node[0].attach_to(base)
            if mat is not None:
                marker_frame = w2r_mat.dot(r2cam_mat).dot(mat)
                gm.gen_frame(marker_frame[:3, 3], marker_frame[:3, :3]).attach_to(pcd_node_node[0])
            print(repr(affine_mat_node[0]))
        elif base.inputmgr.keymap["x"]:
            affine_mat_node[0][:3, :3] = np.dot(
                rm.rotmat_from_axangle(np.array([1, 0, 0]), np.radians(angle_resolution)),
                affine_mat_node[0][:3, :3])
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.homomat_transform_points(w2r_mat.dot(r2cam_mat), points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, pcd_color_rgba)
            pcd_node_node[0].attach_to(base)
            if mat is not None:
                marker_frame = w2r_mat.dot(r2cam_mat).dot(mat)
                gm.gen_frame(marker_frame[:3, 3], marker_frame[:3, :3]).attach_to(pcd_node_node[0])
            print(repr(affine_mat_node[0]))
        elif base.inputmgr.keymap["c"]:
            affine_mat_node[0][:3, :3] = np.dot(
                rm.rotmat_from_axangle(np.array([0, 1, 0]), -np.radians(angle_resolution)),
                affine_mat_node[0][:3, :3])
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.homomat_transform_points(w2r_mat.dot(r2cam_mat), points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, pcd_color_rgba)
            pcd_node_node[0].attach_to(base)
            if mat is not None:
                marker_frame = w2r_mat.dot(r2cam_mat).dot(mat)
                gm.gen_frame(marker_frame[:3, 3], marker_frame[:3, :3]).attach_to(pcd_node_node[0])
            print(repr(affine_mat_node[0]))
        elif base.inputmgr.keymap["v"]:
            affine_mat_node[0][:3, :3] = np.dot(
                rm.rotmat_from_axangle(np.array([0, 1, 0]), np.radians(angle_resolution)),
                affine_mat_node[0][:3, :3])
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.homomat_transform_points(w2r_mat.dot(r2cam_mat), points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, pcd_color_rgba)
            pcd_node_node[0].attach_to(base)
            if mat is not None:
                marker_frame = w2r_mat.dot(r2cam_mat).dot(mat)
                gm.gen_frame(marker_frame[:3, 3], marker_frame[:3, :3]).attach_to(pcd_node_node[0])
            print(repr(affine_mat_node[0]))
        elif base.inputmgr.keymap["b"]:
            affine_mat_node[0][:3, :3] = np.dot(
                rm.rotmat_from_axangle(np.array([0, 0, 1]), -np.radians(angle_resolution)),
                affine_mat_node[0][:3, :3])
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.homomat_transform_points(w2r_mat.dot(r2cam_mat), points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, pcd_color_rgba)
            pcd_node_node[0].attach_to(base)
            if mat is not None:
                marker_frame = w2r_mat.dot(r2cam_mat).dot(mat)
                gm.gen_frame(marker_frame[:3, 3], marker_frame[:3, :3]).attach_to(pcd_node_node[0])
            print(repr(affine_mat_node[0]))
        elif base.inputmgr.keymap["n"]:
            affine_mat_node[0][:3, :3] = np.dot(
                rm.rotmat_from_axangle(np.array([0, 0, 1]), np.radians(angle_resolution)),
                affine_mat_node[0][:3, :3])
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.homomat_transform_points(w2r_mat.dot(r2cam_mat), points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, pcd_color_rgba)
            pcd_node_node[0].attach_to(base)
            if mat is not None:
                marker_frame = w2r_mat.dot(r2cam_mat).dot(mat)
                gm.gen_frame(marker_frame[:3, 3], marker_frame[:3, :3]).attach_to(pcd_node_node[0])
            print(repr(affine_mat_node[0]))

        return task.again

    base.boost.add_task(adjust_pcd, args=[affine_mat_node, pcd_node, pcd])

    base.run()


if __name__ == "__main__":
    import numpy as np
    import cv2
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    from drivers.devices.realsense_d405.d405_driver import RealSenseD405
    import basis.robot_math as rm
    from robot_con.xarm_lite6.xarm_lite6_x import XArmLite6X
    import huri.core.file_sys as fs
    from robot_sim.robots.xarm_lite6_wrs.xarm_lite6_wrs import XArmLite6WRSGripper
    from huri.core.base_boost import zoombase, boost_base

    r2cam_mat = np.array([[0., 1., 0., -0.05074001],
                             [-1., 0., 0., 0.01460267],
                             [0., 0., 1., -0.08705009],
                             [0., 0., 0., 1.]])


    # r2cam_mat= np.array([[-0.00621905, 0.99996356, -0.00584907, -0.04748484],
    #                          [-0.99998007, -0.00622532, -0.00105357, 0.016],
    #                          [-0.00108994, 0.0058424, 0.99998234, -0.08955787],
    #                          [0., 0., 0., 1.]])

    # r2cam_mat = np.array([[5.17837396e-05, 9.99945170e-01, -1.04716432e-02,
    #                        -0.049],
    #                       [-9.99986292e-01, -3.04614327e-06, -5.23595054e-03,
    #                        0.015],
    #                       [-5.23569535e-03, 1.04717708e-02, 9.99931462e-01,
    #                        -8.70500900e-02],
    #                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    #                        1.00000000e+00]])
    # np.array([1,0,0,-0.049])
    hand2eye_mat = np.array([[0., 1., 0., -0.0495],
                             [-1., 0., 0., 0.015],
                             [0., 0., 1., -0.087],
                             [0., 0., 0., 1.]])

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, 0])
    base = boost_base(base)
    rs_pipe = RealSenseD405(toggle_new_process=False)
    # the first frame contains no data information
    rs_pipe.get_pcd_texture_depth()
    rs_pipe.get_pcd_texture_depth()
    rbtx = XArmLite6X()
    rbt = XArmLite6WRSGripper()

    rbt.fk("arm", rbtx.get_jnt_values())
    pos, rot = rbt.get_gl_tcp("arm")
    # jj = rbtx.ik(pos-np.array([0,0,.052]),rot)
    # rbt.fk("arm", jj)
    rbt.jaw_to("hnd", .03)
    rbt.gen_meshmodel().attach_to(base)

    rbt_pose = rbtx.get_pose()
    w2r_mat = rm.homomat_from_posrot(*rbt_pose)
    w2c_mat = w2r_mat.dot(r2cam_mat)

    # get data from D405
    pcd, pcd_color, depth_img, color_img = rs_pipe.get_pcd_texture_depth()
    results = rs_pipe.recognize_ar_marker(aruco_marker_size=.025)
    print(results)
    # wmarker_homomat = rm.homomat_average(marker_results)
    # w2m_mat = w2c_mat.dot(wmarker_homomat)
    # gm.gen_frame(w2m_mat[:3, 3], w2m_mat[:3, :3]).attach_to(base)
    # gm.gen_frame(w2r_mat[:3, 3], w2r_mat[:3, :3]).attach_to(base)
    # gm.gen_frame(w2c_mat[:3, 3], w2m_mat[:3, :3]).attach_to(base)
    # print(w2m_mat)

    if 3 in results and 4 in results and 5 in results:
        # for _ in [results[3], results[4], results[5]]:
        #     # _ = np.dot(w2r_mat.dot(r2cam_mat), (_))
        #     gm.gen_frame(_[:3, 3], _[:3, :3]).attach_to(base)
        mat = rm.homomat_average([results[3], results[4], results[5]], bandwidth=None)
    # pcd_color_rgba = np.append(pcd_color, np.ones((len(pcd_color), 1)), axis=1)
    # gm.gen_pointcloud(pcd,pcd_color_rgba).attach_to(base)
    # base.run()

    elif 0 in results and 1 in results and 2 in results:
        mat = rm.homomat_average([results[0], results[1], results[2]], bandwidth=None)
    else:
        mat = None

    marker_center = np.array([0, -.13 / 2 - 0.03 / 2, 0])
    marker_rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).T
    w2m_mat = rm.homomat_from_posrot(marker_center, marker_rot)
    gm.gen_frame(marker_center, marker_rot).attach_to(base)

    pcd_align = rm.homomat_transform_points(w2c_mat, pcd)
    pcd_color_rgba = np.append(pcd_color, np.ones((len(pcd_color), 1)), axis=1)
    # gm.gen_pointcloud(pcd_align, rgbas=pcd_color_rgba).attach_to(base)
    test(pcd, base)
