__VERSION__ = '0.0.1'

import time

import cv2
import numpy as np

from robot_con.xarm_lite6.xarm_lite6_x import XArmLite6X
from robot_sim.robots.xarm_lite6_wrs import XArmLite6WRSGripper
from drivers.devices.realsense_d405.d405_driver import RealSenseD405
from huri.core.common_import import fs, wd, gm, rm
from huri.components.vision.tube_detector import TestTubeDetector, extract
from huri.math.math import combination
from vision_sys import Rack_Locator, detect_tube, measure_pose_err
from huri.definitions.rack_def import Rack_Hard_Proto
#
from huri.core.base_boost import zoombase, boost_base

if __name__ == "__main__":
    APPROACH_DISTANCE_LIST = [.07, .12]
    base = wd.World(cam_pos=[0, 0, 1.5], lookat_pos=[0, 0, 0])
    base = boost_base(base)
    rs_pipe = RealSenseD405(toggle_new_process=True)
    rbt = XArmLite6WRSGripper(enable_cc=True)
    rbtx = XArmLite6X()

    detector = TestTubeDetector(affine_mat_path=None)
    # # goto home configuration
    # rbtx.homeconf()
    gm.gen_frame().attach_to(base)
    rbt.fk("arm", rbtx.get_jnt_values())
    rbt.gen_meshmodel().attach_to(base)

    # TODO write a program to self calibrate the camera
    hand_to_eye_pos = np.array([.006, 0.049, -.062])
    hand_to_eye_rot = rm.rotmat_from_axangle(np.array([0, 0, 1]), np.pi / 2)
    hand_to_eye_mat = rm.homomat_from_posrot(hand_to_eye_pos, hand_to_eye_rot)
    eye_to_hand_mat = np.linalg.inv(hand_to_eye_mat)
    hand2eye_mat = np.array([[-0.01247437, 0.99989549, 0.00730778, -0.05114001],
                             [-0.99991293, -0.01244247, -0.00439458, 0.01040267],
                             [-0.00430319, -0.00736196, 0.99996364, -0.08777642],
                             [0., 0., 0., 1.]])

    rl = Rack_Locator(rbtx, rs_pipe,
                      origin_obs_pose=np.array([0.25, 0., 0.145]),
                      hand_to_eye_mat=hand2eye_mat,
                      toggle_debug=False)

    HEIGHT_RANGE = (.055, 0.075)
    pcd_w, im, Rack_Hard_Proto, _rack_tf, detector, rack_tf = fs.load_pickle("pcd_debug_data")
    gm.gen_pointcloud(pcd_w).attach_to(base)
    _rack, _rack_tf, pcd_w, im = rl.track(Rack_Hard_Proto, rack_init_pose=rack_tf,
                                          height_range=HEIGHT_RANGE, toggle_pcd=False)
    _rack.gen_mesh_model().attach_to(base)
    base.run()
    _rack, __rack_tf = detect_tube(pcd_w, im,
                                   rack_proto=Rack_Hard_Proto,
                                   rack_tf=_rack_tf,
                                   detector=detector,
                                   toggle_yolo=True,
                                   toggle_detect_tube_pos=True)



    base.run()
