__VERSION__ = '0.0.1'

import time

import cv2
import numpy as np

from robot_con.xarm_lite6.xarm_lite6_x import XArmLite6X
from robot_sim.manipulators.xarm_lite6 import XArmLite6
from dual_d405_manager import RealSenseD405Dual
from huri.core.common_import import fs, wd, gm, rm
from huri.components.vision.tube_detector import TestTubeDetector, extract
from huri.math.math import combination
from vision_sys import RackLocatorDual, detect_tube
from huri.definitions.rack_def import Rack_Hard_Proto
from constants import HEIGHT_RANGE, TCP2EYE_MAT_DEVICE_1, TCP2EYE_MAT_DEVICE_2, OBS_HEIGHT
from huri.core.base_boost import zoombase, boost_base

if __name__ == "__main__":
    base = wd.World(cam_pos=[0, 0, 1.5], lookat_pos=[0, 0, 0])
    base = boost_base(base)
    rs_pipe = RealSenseD405Dual()
    rbt = XArmLite6(enable_cc=True)
    rbtx = XArmLite6X(ip='192.168.1.190', has_gripper=False)

    detector = TestTubeDetector(affine_mat_path=None)
    # goto home configuration
    gm.gen_frame().attach_to(base)
    rbt.fk(rbtx.get_jnt_values())
    rbt.gen_meshmodel().attach_to(base)

    is_scan = True

    rl = RackLocatorDual(rbtx, rs_pipe,
                         origin_obs_pose=np.array([.27, 0.07, OBS_HEIGHT]),
                         hand_to_eye_mat_1=TCP2EYE_MAT_DEVICE_1,
                         hand_to_eye_mat_2=TCP2EYE_MAT_DEVICE_2,
                         toggle_debug=False)

    # x_range = np.array([.35])
    # y_range = np.array([0.07])
    # z_range = np.array([.14])
    # r = combination([x_range, y_range, z_range]).tolist()
    # # print(r)
    # rl.add_obs_pos(obs_pos=r)
    if is_scan:
        rbtx.homeconf()
        c_pcd, pcds = rl.scan(toggle=False)

        gm.gen_rgb_pointcloud(c_pcd).attach_to(base)

        fs.dump_pickle([c_pcd, pcds], "pcd_data", reminder=False)

        gm.gen_rgb_pointcloud(c_pcd).attach_to(base)
        rack, rack_tf = rl.locate_rack(c_pcd, Rack_Hard_Proto,
                                       height_range=HEIGHT_RANGE,
                                       toggle_debug=True)

        # rack.gen_mesh_model().attach_to(base)

        # base.run()
        print(rack_tf)
        gm.gen_sphere([0.30035, 0.02837, .08]).attach_to(base)

        # pcd_rgba, im_center = rl.obs_in_rack_center(rack_tf, toggle=False)
        SPEED = 100
        ACC = 1500

        obs_p_1 = rack.get_slot_homomat((2, 0))
        new_rbt_obs_pos_1 = rl.get_obs_xy_point(obs_p_1[:2, 3], )
        rbtx.move_p(new_rbt_obs_pos_1[:3, 3],
                    new_rbt_obs_pos_1[:3, :3], speed=SPEED, mvacc=ACC)
        pcd_1, im_1 = rl.get_pcd_im_1()
        pcd_2, im_2 = rl.get_pcd_im_2()

        obs_p_2 = rack.get_slot_homomat((2, 9))
        new_rbt_obs_pos_2 = rl.get_obs_xy_point(obs_p_2[:2, 3], )
        rbtx.move_p(new_rbt_obs_pos_2[:3, 3],
                    new_rbt_obs_pos_2[:3, :3], speed=SPEED, mvacc=ACC)
        pcd_3, im_3 = rl.get_pcd_im_1()
        pcd_4, im_4 = rl.get_pcd_im_2()
        pcd_w = np.vstack((pcd_1, pcd_2, pcd_3, pcd_4))
        rack, rack_tf = rl.locate_rack(pcd_w, Rack_Hard_Proto, height_range=HEIGHT_RANGE)
        print("?")
        base.run()
        rack_instance = Rack_Hard_Proto
        for pcd, im in [(pcd_1, im_1), (pcd_2, im_2), (pcd_3, im_3), (pcd_4, im_4)]:
            rack_instance, rack_tf = detect_tube(pcd[:, :3], im,
                                                 rack_proto=rack_instance,
                                                 rack_tf=rack_tf,
                                                 detector=detector,
                                                 toggle_yolo=True,
                                                 toggle_detect_tube_pos=True)

    else:
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
    print(repr(rack_instance.rack_status))
    print(repr(rack_tf))
    # gm.gen_pointcloud(pcd_rgba[:, :3]).attach_to(base)
    rack_instance.gen_mesh_model(gen_tube=True).attach_to(base)

    base.run()

    show_animation = True
    if show_animation:
        pose_node = [None, None, None]


        def track_rack(rl: RackLocatorDual, rack_tf, task):
            if pose_node[0] is not None:
                pose_node[0].remove()
            if pose_node[1] is not None:
                pose_node[1].remove()
            a = time.time()
            rack, _rack_tf, pcd_w, im = rl.track(Rack_Hard_Proto, rack_init_pose=rack_tf,
                                                 height_range=HEIGHT_RANGE)
            err = measure_pose_err(_rack_tf, rack_tf)
            err_pos = err[:3]
            err_rot = err[3:6]
            errnorm_pos = err_pos.T.dot(err_pos)
            errnorm_rot = np.linalg.norm(err_rot)
            print("Error pos is", errnorm_pos)
            print("Error rot is", errnorm_rot)
            b = time.time()
            print(b - a)
            pose_node[0] = rack.gen_mesh_model()
            pose_node[0].attach_to(base)
            pose_node[1] = gm.gen_rgb_pointcloud(pcd_w)
            pose_node[1].attach_to(base)
            if errnorm_pos > 5e-6:
                print("moved")
                return task.done
            elif errnorm_rot > .25:
                print("moved")
                return task.done
            return task.again


        base.boost.add_task(track_rack, args=[rl, rack_tf], timestep=.7)

    base.run()
