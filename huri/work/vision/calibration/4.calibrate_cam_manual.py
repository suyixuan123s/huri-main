"""
Example for utilize the D405 driver
Author: Chen Hao (chen960216@gmail.com), 20220916, osaka
"""
import time

import numpy as np
import cv2
import visualization.panda.world as wd
import huri.core.file_sys as fs
from robot_sim.robots.xarm_lite6_wrs.xarm_lite6_wrs import XArmLite6WRSGripper
import modeling.geometric_model as gm
import basis.robot_math as rm

if __name__ == "__main__":
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, 0])
    rbt = XArmLite6WRSGripper()
    rbt.gen_meshmodel().attach_to(base)

    # c2m_homomats_012
    marker_center = np.array([0, .13 / 2 + 0.03 / 2, 0])
    marker_rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).T
    w2m_mat = rm.homomat_from_posrot(marker_center, marker_rot)
    gm.gen_frame(marker_center, marker_rot).attach_to(base)

    r2c_homomat_012 = None
    r2c_homomat_345 = None

    if fs.Path('obs_file_012.pkl').exists():
        print("!")
        c2m_homomats_012, w2tcp_homomats_012 = fs.load_pickle('obs_file_012.pkl')
        r2c_homomats = []
        w2tcp_homomats = w2tcp_homomats_012
        c2m_homomats = c2m_homomats_012
        for i in range(len(w2tcp_homomats)):
            w2tcp_mat = w2tcp_homomats[i]
            c2m_mat = c2m_homomats[i]
            r2c_mat = np.linalg.inv(w2tcp_mat).dot(w2m_mat).dot(np.linalg.inv(c2m_mat))
            r2c_homomats.append(r2c_mat)
            # gm.gen_frame(r2c_mat[:3, 3], r2c_mat[:3, :3]).attach_to(base)
            gm.gen_frame(r2c_mat[:3, 3], r2c_mat[:3, :3]).attach_to(base)
        r2c_homomat_012 = r2c_homomats

    # ------------------
    if fs.Path('obs_file_345.pkl').exists():
        # c2m_homomats_345
        c2m_homomats_345, w2tcp_homomats_345 = fs.load_pickle('obs_file_345.pkl')
        marker_center = np.array([0, -.13 / 2 - 0.03 / 2, 0])
        marker_rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).T
        w2m_mat = rm.homomat_from_posrot(marker_center, marker_rot)
        gm.gen_frame(marker_center, marker_rot).attach_to(base)

        r2c_homomats = []
        w2tcp_homomats = w2tcp_homomats_345
        c2m_homomats = c2m_homomats_345
        for i in range(len(w2tcp_homomats)):
            w2tcp_mat = w2tcp_homomats[i]
            c2m_mat = c2m_homomats[i]
            r2c_mat = np.linalg.inv(w2tcp_mat).dot(w2m_mat).dot(np.linalg.inv(c2m_mat))
            r2c_homomats.append(r2c_mat)
            # gm.gen_frame(w2tcp_mat[:3, 3], w2tcp_mat[:3, :3]).attach_to(base)
            gm.gen_frame(r2c_mat[:3, 3], r2c_mat[:3, :3]).attach_to(base)
            print(r2c_mat)
        r2c_homomat_345 = r2c_homomats
        print(r2c_homomat_345)
    print("Result is :")
    if r2c_homomat_012 is not None and r2c_homomat_345 is not None:
        print(f"np.{repr(rm.homomat_average([*r2c_homomat_012, *r2c_homomat_345], bandwidth=10 / 1000))}")
    elif r2c_homomat_012 is not None:
        print(f"np.{repr(r2c_homomat_012)}")
    elif r2c_homomat_345 is not None:
        print(f"np.{repr(r2c_homomat_345)}")
    else:
        raise Exception
    # print(f"np.{repr(rm.homomat_average([*r2c_homomat_012, *r2c_homomat_345], bandwidth=10 / 1000))}")
    base.run()
