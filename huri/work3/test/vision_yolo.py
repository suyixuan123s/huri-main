__VERSION__ = '0.0.1'

import time

import cv2
import numpy as np

from robot_con.xarm_lite6.xarm_lite6_x import XArmLite6X
from robot_sim.manipulators.xarm_lite6 import XArmLite6
from huri.work3.dual_d405_manager import RealSenseD405
from huri.core.common_import import fs, wd, gm, rm
from huri.components.vision.tube_detector import TestTubeDetector, extract
from huri.math.math import combination
from huri.work3.vision_sys import RackLocator, detect_tube
from huri.definitions.rack_def import Rack_Hard_Proto
from huri.work3.constants20240215 import HEIGHT_RANGE, TCP2EYE_MAT_DEVICE_1, TCP2EYE_MAT_DEVICE_2, OBS_HEIGHT
from huri.core.base_boost import zoombase, boost_base

if __name__ == "__main__":
    base = wd.World(cam_pos=[0, 0, 1.5], lookat_pos=[0, 0, 0])
    base = boost_base(base)
    rs_pipe = RealSenseD405()
    rbt = XArmLite6(enable_cc=True)
    rbtx = XArmLite6X(ip='192.168.1.190', has_gripper=False)

    detector = TestTubeDetector(affine_mat_path=None)
    # goto home configuration
    gm.gen_frame().attach_to(base)
    rbt.fk(rbtx.get_jnt_values())
    rbt.gen_meshmodel().attach_to(base)
    rl = RackLocator(rbtx, rs_pipe,
                         origin_obs_pose=np.array([.27, 0.07, OBS_HEIGHT]),
                         hand_to_eye_mat=TCP2EYE_MAT_DEVICE_1,
                         toggle_debug=False)
    pcd_rgba, im = rl.get_pcd_im()
    rack, rack_tf = rl.locate_rack(pcd_rgba[:, :3], Rack_Hard_Proto,
                                   toggle_debug=True, height_range=HEIGHT_RANGE)
    # base.run()
    rack_instance, rack_tf = detect_tube(pcd_rgba[:, :3], im,
                                         rack_proto=Rack_Hard_Proto,
                                         rack_tf=rack_tf,
                                         detector=detector,
                                         toggle_yolo=True,
                                         toggle_detect_tube_pos=True)
    rack_instance.gen_mesh_model(gen_tube=True).attach_to(base)
    gm.gen_rgb_pointcloud(pcd_rgba).attach_to(base)
    base.run()