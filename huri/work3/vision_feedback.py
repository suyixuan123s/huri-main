__VERSION__ = '0.0.1'

import time

import cv2
import numpy as np

from robot_con.xarm_lite6.xarm_lite6_x import XArmLite6X
from robot_sim.robots.xarm_lite6_wrs import XArmLite6WRSGripper
from dual_d405_manager import RealSenseD405Dual
from huri.core.common_import import fs, wd, gm, rm
from huri.components.vision.tube_detector import TestTubeDetector, extract
from huri.math.math import combination
from vision_sys import RackLocatorDual, detect_tube
from huri.definitions.rack_def import Rack_Hard_Proto
from constants import HEIGHT_RANGE
import constants
from huri.core.base_boost import zoombase, boost_base

if __name__ == "__main__":
    APPROACH_DISTANCE_LIST = [.07, .12]
    base = wd.World(cam_pos=[0, 0, 1.5], lookat_pos=[0, 0, 0])
    base = boost_base(base)
    rs_pipe = RealSenseD405Dual()
    rbt = XArmLite6WRSGripper(enable_cc=True)
    rbtx = XArmLite6X()

    detector = TestTubeDetector(affine_mat_path=None)
    # # goto home configuration
    # rbtx.homeconf()
    gm.gen_frame().attach_to(base)
    rbt.fk("arm", rbtx.get_jnt_values())
    # rbt.gen_meshmodel().attach_to(base)

    # TODO write a program to self calibrate the camera
    hand_to_eye_pos = np.array([.006, 0.049, -.062])
    hand_to_eye_rot = rm.rotmat_from_axangle(np.array([0, 0, 1]), np.pi / 2)
    hand_to_eye_mat = rm.homomat_from_posrot(hand_to_eye_pos, hand_to_eye_rot)
    eye_to_hand_mat = np.linalg.inv(hand_to_eye_mat)
    # hand2eye_mat = np.array([[0., 1., 0., -0.04783408],
    #                          [-1., 0., 0., 0.01529641],
    #                          [0., 0., 1., -0.08705009],
    #                          [0., 0., 0., 1.]])
    hand2eye_mat = np.array([[0., 1., 0., -0.04383408],
                             [-1., 0., 0., 0.01479641],
                             [0., 0., 1., -0.08705009],
                             [0., 0., 0., 1.]])

    rl = RackLocatorDual(rbtx, rs_pipe,
                         origin_obs_pose=np.array([0.25, 0., 0.145]),
                         hand_to_eye_mat_1=constants.TCP2EYE_MAT_DEVICE_1,
                         hand_to_eye_mat_2=constants.TCP2EYE_MAT_DEVICE_2,
                         toggle_debug=False)

    motion_batch = fs.load_pickle(path="debug.pkl")
    el1 = motion_batch.elements[0]

    gm.gen_frame(el1.pick_approach[:3, 3], el1.pick_approach[:3, :3]).attach_to(base)
    print(el1.init_slot_id)
    # pcd, im = rl.get_pcd_im()
    # gm.gen_pointcloud(pcd).attach_to(base)
    # rbtx.move_p(el1.pick_approach[:3, 3], el1.pick_approach[:3, :3], speed=1000, mvacc=1000)
    # gm.gen_frame(el1.pick[:3, 3], el1.pick[:3, :3]).attach_to(base)

    grasp_cache_dict = {
        # purple ring
        1: fs.load_json(path=fs.workdir_data.joinpath("grasps", "lite6", "purple_ring_cap_tube_grasps.json")),
        # blue
        2: fs.load_json(path=fs.workdir_data.joinpath("grasps", "lite6", "blue_cap_tube_grasps.json")),
        # white
        3: fs.load_json(path=fs.workdir_data.joinpath("grasps", "lite6", "white_cap_tube_grasps.json")),
    }

    rack_status, rack_tf = fs.load_pickle("rack_states")

    a = time.time()
    _rack, _rack_tf, pcd_w, im = rl.track(Rack_Hard_Proto, rack_init_pose=rack_tf,
                                          height_range=HEIGHT_RANGE,
                                          rack_state=rack_status,
                                          toggle_pcd=True)

    base.run()
    rack_instance, rack_tf = detect_tube(pcd_w[:, :3], im,
                                         rack_proto=Rack_Hard_Proto,
                                         rack_tf=_rack_tf,
                                         detector=detector,
                                         toggle_yolo=False,
                                         toggle_detect_tube_pos=True)
    rack_instance.gen_mesh_model(gen_tube=True).attach_to(base)
    tube_homomat = _rack.get_slot_homomat(el1.init_slot_id)
    grasp = grasp_cache_dict[el1.tube.type][el1.grasp_id]
    print(repr(rack_instance.rack_status))
    init_jaw_center_pos = tube_homomat[:3, 3] + tube_homomat[:3, :3].dot(grasp[1])
    init_jaw_center_rotmat = tube_homomat[:3, :3].dot(grasp[2])

    b = time.time()

    print(b - a)
    err = measure_pose_err(_rack_tf, el1.rack_tf)

    #
    # rbtx.move_p(init_jaw_center_pos - init_jaw_center_rotmat[:3, 2] * APPROACH_DISTANCE_LIST[0], init_jaw_center_rotmat,
    #             speed=1000, mvacc=1000)
    # rbtx.move_p(init_jaw_center_pos, init_jaw_center_rotmat, speed=1000, mvacc=1000)
    # _rack.gen_mesh_model().attach_to(base)
    gm.gen_frame(*rbtx.get_pose()).attach_to(base)

    base.run()
