"""
Example for utilize the D405 driver
Author: Chen Hao (chen960216@gmail.com), 20220916, osaka
"""
import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm
from d405_driver import RealSenseD405
import cv2

import basis.robot_math as rm

if __name__ == "__main__":
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, 0])
    rs_pipe = RealSenseD405()
    # the first frame contains no data information
    rs_pipe.get_pcd_texture_depth()
    while True:
        pcd, pcd_color, depth_img, color_img = rs_pipe.get_pcd_texture_depth()
        cv2.imshow("a",color_img)
        k = cv2.waitKey(1)
        if k == 27:
            break

    # get data from D405
    pcd, pcd_color, depth_img, color_img = rs_pipe.get_pcd_texture_depth()
    # convert RGB pcd color to RGBA
    pcd_color_rgba = np.append(pcd_color, np.ones((len(pcd_color), 1)), axis=1)

    m = rs_pipe.recognize_ar_marker(aruco_marker_size=100 / 1000, toggle_show=False)
    affine_mat = np.array([[ 0.34190478, -0.93820026,  0.05367861, -0.05125107],
       [-0.91520238, -0.34540205, -0.20761028, -0.00144602],
       [ 0.21332072,  0.02185616, -0.97673772,  0.29681173],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])

    pcd = rm.homomat_transform_points(np.linalg.inv(affine_mat),pcd)
    print(m)


    plot = [gm.gen_frame()]

    plot[0].remove()
    plot[0] = gm.gen_frame()
    # results: dict = rs_pipe.recognize_ar_marker(aruco_marker_size=.025)

    # mat = rm.homomat_average(list(results.values()))
    # for key, mat in results.items():
    # gm.gen_frame(mat[:3, 3], mat[:3, :3]).attach_to(plot[0])
    gm.gen_pointcloud(pcd, rgbas=pcd_color_rgba.tolist()).attach_to(plot[0])
    import huri.core.file_sys as fs

    fs.dump_pickle([pcd, pcd_color, color_img], "test.pkl")
    plot[0].attach_to(base)

    base.run()
