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

if __name__ == "__main__":
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, 0])
    rs_pipe = RealSenseD405(toggle_new_process=False)
    # the first frame contains no data information
    rs_pipe.get_pcd_texture_depth()
    rs_pipe.get_pcd_texture_depth()
    rbtx = XArmLite6X()
    # rbtx.homeconf()

    toggle_debug = True
    w2tcp_homomats_012 = []
    w2tcp_homomats_345 = []
    c2m_homomats_012 = []
    c2m_homomats_345 = []
    rbt_pose = rbtx.get_pose()
    # get data from D405
    pcd, pcd_color, depth_img, color_img = rs_pipe.get_pcd_texture_depth()
    # convert RGB pcd color to RGBA
    results: dict = rs_pipe.recognize_ar_marker(aruco_marker_size=.025)
    print(results)
    filename = ""
    if 3 in results and 4 in results and 5 in results:
        mat_list = [results[3], results[4], results[5]]
        c2m_homomats = c2m_homomats_345
        w2tcp_homomats = w2tcp_homomats_345
        filename = "345"
    elif 0 in results and 1 in results and 2 in results:
        mat_list = [results[0], results[1], results[2]]
        c2m_homomats = c2m_homomats_012
        w2tcp_homomats = w2tcp_homomats_012
        filename = "012"
    else:
        exit(0)
    mat = rm.homomat_average(list(mat_list))
    c2m_homomats.append(mat)
    w2tcp_homomats.append(rm.homomat_from_posrot(*rbt_pose))

    # from robot_sim.robots.xarm_lite6_wrs.xarm_lite6_wrs import XArmLite6WRSGripper
    # rbt = XArmLite6WRSGripper()
    # rbt.fk("arm", jnt)
    # rbt.gen_meshmodel().attach_to(base)
    # rbt_pose = w2tcp_homomats[-1]
    # gm.gen_frame(rbt_pose[:3, 3], rbt_pose[:3, :3]).attach_to(base)
    # base.run()
    print(mat)
    print("-" * 10)

    if toggle_debug:
        gm.gen_frame().attach_to(base)
        pcd_color_rgba = np.append(pcd_color, np.ones((len(pcd_color), 1)), axis=1)
        gm.gen_pointcloud(pcd[:, :3], rgbas=pcd_color_rgba).attach_to(base)
        gm.gen_frame(mat[:3, 3], mat[:3, :3], thickness=.002).attach_to(base)
    cv2.imshow("r", color_img)
    cv2.waitKey(100)

    fs.dump_pickle([c2m_homomats, w2tcp_homomats], f'obs_file_{filename}.pkl')

    base.run()

    exit()
