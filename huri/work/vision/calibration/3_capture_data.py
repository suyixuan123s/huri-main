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
    obs_jnts = [np.array([-1.498343, 0.61801, 0.553621, -3.175724, -0.124876, 3.272365]),
                np.array([1.627694, 0.641145, 0.526928, -2.148724, -0.056725, 2.181704])
                ]
    toggle_debug = False
    w2tcp_homomats_012 = []
    w2tcp_homomats_345 = []
    c2m_homomats_012 = []
    c2m_homomats_345 = []
    for ind, jnt in enumerate(obs_jnts):
        # rbtx.homeconf()
        rbtx.move_j(np.array([-0.159944, -0.491629, 0.711456, -3.156549, -1.164068, 2.725884]), speed=1, mvacc=2)
        rbtx.move_j(jnt, speed=1, mvacc=2)
        time.sleep(.5)
        rbt_pose = rbtx.get_pose()

        rots_candidate = np.array(rm.gen_icorotmats(icolevel=3,
                                                    rotation_interval=np.radians(360),
                                                    crop_normal=np.array([0, 0, -1]),
                                                    crop_angle=np.radians(15),
                                                    toggleflat=True))
        obs_homomats = []
        gm.gen_frame(*rbt_pose).attach_to(base)
        for rot in rots_candidate:
            obs_homomat = rm.homomat_from_posrot(rbt_pose[0],
                                                 np.dot(rbt_pose[1], rm.rotmat_from_axangle(np.array([0, 0, 1]),
                                                                                            np.radians(225)).dot(rot)))
            gm.gen_frame(obs_homomat[:3, 3], obs_homomat[:3, :3]).attach_to(base)
            obs_homomats.append(obs_homomat)
        # from robot_sim.robots.xarm_lite6_wrs.xarm_lite6_wrs import XArmLite6WRSGripper
        #
        # rbt = XArmLite6WRSGripper()
        # rbt.fk("arm", jnt)
        # rbt.gen_meshmodel().attach_to(base)
        # base.run()
        for p in obs_homomats:
            rbtx.move_p(pos=p[:3, 3], rot=p[:3, :3])
            time.sleep(.5)
            rbt_pose = rbtx.get_pose()
            # get data from D405
            pcd, pcd_color, depth_img, color_img = rs_pipe.get_pcd_texture_depth()
            # convert RGB pcd color to RGBA
            results: dict = rs_pipe.recognize_ar_marker(aruco_marker_size=.025)
            print(results)
            if 3 in results and 4 in results and 5 in results:
                mat_list = [results[3], results[4], results[5]]
                c2m_homomats = c2m_homomats_345
                w2tcp_homomats = w2tcp_homomats_345
            elif 0 in results and 1 in results and 2 in results:
                mat_list = [results[0], results[1], results[2]]
                c2m_homomats = c2m_homomats_012
                w2tcp_homomats = w2tcp_homomats_012
            else:
                continue
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
            print(ind, mat)
            print("-" * 10)

            if toggle_debug:
                pcd_color_rgba = np.append(pcd_color, np.ones((len(pcd_color), 1)), axis=1)
                gm.gen_pointcloud(pcd, rgbas=pcd_color_rgba).attach_to(base.run())
                gm.gen_frame(mat[:3, 3], mat[:3, :3]).attach_to(base.run())
            cv2.imshow("r", color_img)
            cv2.waitKey(100)

    fs.dump_pickle([w2tcp_homomats_012, w2tcp_homomats_345, c2m_homomats_012, c2m_homomats_345], 'obs_file.pkl')

    exit()
