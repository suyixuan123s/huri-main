"""
Example for utilize the D405 driver
Author: Chen Hao (chen960216@gmail.com), 20220916, osaka
"""
import time

import numpy as np
import cv2
import visualization.panda.world as wd
import modeling.geometric_model as gm
from drivers.devices.realsense_d405.d405_driver import RealSenseD405
import basis.robot_math as rm
from robot_con.xarm_lite6.xarm_lite6_x import XArmLite6X
import huri.core.file_sys as fs
from robot_sim.robots.xarm_lite6_wrs.xarm_lite6_wrs import XArmLite6WRSGripper

if __name__ == "__main__":
    # r2cam_mat = np.array([[-0.01187041, 0.99990454, 0.00707167, -0.04834394],
    #                       [-0.99991552, -0.01183249, -0.00538058, 0.01083987],
    #                       [-0.00529639, -0.00713494, 0.99996052, -0.08760719],
    #                       [0., 0., 0., 1.]])

    r2cam_mat = np.array([[0, 1, 0, -0.04783408],
                          [-1, 0, 0, 0.00829641],
                          [0, 0, 1, -0.08705009],
                          [0., 0., 0., 1.]])
    r2cam_mat = np.array([[-0.00507709, 0.99984001, 0.01715175, -0.04686391],
                          [-0.99988338, -0.00482877, -0.01448858, 0.0098785],
                          [-0.01440344, -0.01722331, 0.99974792, -0.08519608],
                          [0., 0., 0., 1.]])

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, 0])
    rs_pipe = RealSenseD405(toggle_new_process=False)
    # the first frame contains no data information
    rs_pipe.get_pcd_texture_depth()
    rs_pipe.get_pcd_texture_depth()
    rbtx = XArmLite6X()
    rbt = XArmLite6WRSGripper()

    rbt.fk("arm", rbtx.get_jnt_values())
    rbt.gen_meshmodel().attach_to(base)

    rbt_pose = rbtx.get_pose()
    w2r_mat = rm.homomat_from_posrot(*rbt_pose)
    w2c_mat = w2r_mat.dot(r2cam_mat)

    # get data from D405
    pcd, pcd_color, depth_img, color_img = rs_pipe.get_pcd_texture_depth()
    marker_results = list(rs_pipe.recognize_ar_marker().values())
    wmarker_homomat = rm.homomat_average(marker_results)
    w2m_mat = w2c_mat.dot(wmarker_homomat)
    gm.gen_frame(w2m_mat[:3, 3], w2m_mat[:3, :3]).attach_to(base)
    gm.gen_frame(w2r_mat[:3, 3], w2r_mat[:3, :3]).attach_to(base)
    gm.gen_frame(w2c_mat[:3, 3], w2m_mat[:3, :3]).attach_to(base)
    gm.gen_sphere(w2c_mat[:3, 3]).attach_to(base)
    print(w2m_mat)

    pcd_align = rm.homomat_transform_points(w2c_mat, pcd)
    pcd_color_rgba = np.append(pcd_color, np.ones((len(pcd_color), 1)), axis=1)
    gm.gen_pointcloud(pcd_align, rgbas=pcd_color_rgba).attach_to(base)
    base.run()
