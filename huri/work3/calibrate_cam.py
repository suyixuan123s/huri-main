__VERSION__ = '0.0.1'

import cv2
import numpy as np

from robot_con.xarm_lite6.xarm_lite6_x import XArmLite6X
from robot_sim.robots.xarm_lite6_wrs import XArmLite6WRSGripper
from drivers.devices.realsense_d405.d405_driver import RealSenseD405
from huri.core.common_import import fs, wd, gm, rm
from huri.components.vision.tube_detector import TestTubeDetector, extract
from huri.math.math import combination
from vision_sys import RackLocatorDual, detect_tube
from huri.definitions.rack_def import Rack_Hard_Proto
#
from huri.core.base_boost import zoombase, boost_base

if __name__ == "__main__":
    base = wd.World(cam_pos=[0, 0, 1.5], lookat_pos=[0, 0, 0])
    base = boost_base(base)
    rs_pipe = RealSenseD405(toggle_new_process=True)
    rbt = XArmLite6WRSGripper(enable_cc=True)
    rbtx = XArmLite6X()
    component_name = "arm"
    detector = TestTubeDetector(affine_mat_path=None)
    # goto home configuration
    # rbtx.homeconf()
    gm.gen_frame().attach_to(base)
    rbt.fk(component_name, rbtx.get_jnt_values())
    rbt.gen_meshmodel().attach_to(base)

    # TODO write a program to self calibrate the camera
    hand_to_eye_pos = np.array([.006, 0.049, -.062])
    hand_to_eye_rot = rm.rotmat_from_axangle(np.array([0, 0, 1]), np.pi / 2)
    hand_to_eye_mat = rm.homomat_from_posrot(hand_to_eye_pos, hand_to_eye_rot)
    eye_to_hand_mat = np.linalg.inv(hand_to_eye_mat)

    rl = RackLocatorDual(rbtx, rs_pipe,
                         origin_obs_pose=np.array([0.25, 0., 0.15]),
                         eye_to_hand_mat=eye_to_hand_mat,
                         toggle_debug=False)

    c_pcd, pcds, imgs = rl.scan(toggle=False)

    gm.gen_rgb_pointcloud(c_pcd).attach_to(base)

    base.run()
