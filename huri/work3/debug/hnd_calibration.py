from time import strftime

import matplotlib

matplotlib.use('TkAgg')

from huri.core.common_import import *
from huri.core.constants import SENSOR_INFO
# from huri.vision.phoxi_capture import vision_pipeline, SensorMarkerHandler, vision_read_data
import huri.components.utils.plot_projection as pp
from huri.components.vision.tube_detector import TestTubeDetector, extract
from huri.definitions.rack_def import Rack_Hard_Proto
import huri.vision.pnt_utils as pntu
import cv2

IP_ADR = SENSOR_INFO.IP_ADR_DEPTH_SENSOR_DEBUG
SAVE_PATH = fs.workdir / "data" / "vision_exp" / f"{strftime('%Y%m%d-%H%M%S')}.pkl"  # None: Do not save the data captured by phoxi
AFFINE_MAT_PATH = fs.workdir / "data/calibration/qaqqq.json"
DEBUG = False
from huri.work3.constants import (TCP2EYE_MAT_DEVICE_1,
                                  TCP2EYE_MAT_DEVICE_2,
                                  HEIGHT_RANGE,
                                  GOAL_PATTERN,
                                  GOAL_PATTERN1,
                                  GOAL_PATTERN2,
                                  GOAL_UP_DISTANCE,
                                  APPROACH_DISTANCE_LIST,
                                  DEPART_DISTANCE_LIST,
                                  SPEED,
                                  ACC,
                                  TOGGLE_RACK_LOCATOR,
                                  TOGGLE_YOLO,
                                  OBS_HEIGHT)

TOGGLE_VIS_DEBUG = True
from huri.work3.vision_sys import RackLocatorDual, detect_tube, is_pose_err
from huri.work3.utils import init_env_rbt, np, MotionElement, MotionBatch
from huri.work3.dual_d405_manager import RealSenseD405Dual
from robot_sim.end_effectors.gripper.lite6_wrs_gripper import Lite6WRSGripper2


def init_vision_system(rl, detector, rbtx, toggle_debug=False):
    # x_range = np.array([.3])
    # y_range = np.array([0.07])
    # z_range = np.array([OBS_HEIGHT])
    # r = combination([x_range, y_range, z_range]).tolist()
    # rl.add_obs_pos(obs_pos=r)
    rbtx.homeconf()
    c_pcd, pcds = rl.scan(toggle=False)
    rack, rack_tf = rl.locate_rack(c_pcd, Rack_Hard_Proto, height_range=HEIGHT_RANGE,
                                   toggle_debug=toggle_debug)
    # pcd_rgba, im_center = rl.obs_in_rack_center(rack_tf, toggle=TOGGLE_RACK_LOCATOR)

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

    rack_instance = Rack_Hard_Proto
    pcd_w = np.vstack((pcd_1, pcd_2, pcd_3, pcd_4))
    rack, rack_tf = rl.locate_rack(pcd_w, Rack_Hard_Proto, height_range=HEIGHT_RANGE, )
    for pcd, im in [(pcd_1, im_1), (pcd_2, im_2), (pcd_3, im_3), (pcd_4, im_4)]:
        rack_instance, rack_tf = detect_tube(pcd[:, :3], im,
                                             rack_proto=rack_instance,
                                             rack_tf=rack_tf,
                                             detector=detector,
                                             toggle_yolo=False,
                                             toggle_detect_tube_pos=True)
        gm.gen_rgb_pointcloud(pcd_w).attach_to(base)
    # if np.sum(E(rack_instance.rack_status, GOAL_PATTERN2)) > np.sum(E(rack_instance.rack_status, GOAL_PATTERN1)):
    #     rack_instance.rack_status = np.fliplr(rack_instance.rack_status)
    #     rack_instance.rack_status_confidence = np.fliplr(rack_instance.rack_status_confidence)
    #     rack_instance.tubes_pose = np.fliplr(rack_instance.tubes_pose)

    if TOGGLE_VIS_DEBUG:
        fs.dump_pickle([(pcd_1, im_1), (pcd_2, im_2), (pcd_3, im_3), (pcd_4, im_4)], "vis_calibration_debug",
                       reminder=False)
    return rack_instance, rack_tf


def grasp_check(init_pos, init_rotmat, grasp_dir, grasp_info_list, rbthp, tube_rack):
    cnt = 0
    grpr = Lite6WRSGripper2(enable_cc=True)
    for grasp_id, (jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat) in enumerate(grasp_info_list):
        cnt += 1
        # if cnt % 50 != 0:
        #     continue
        gic = grpr.copy()
        init_jaw_center_pos = init_pos + init_rotmat.dot(jaw_center_pos)
        # robot ee rot
        init_jaw_center_rotmat = init_rotmat.dot(jaw_center_rotmat)
        gic.grip_at_with_jcpose(gl_jaw_center_pos=init_jaw_center_pos,
                                gl_jaw_center_rotmat=init_jaw_center_rotmat,
                                jaw_width=.034)
        gm.gen_arrow(spos=init_jaw_center_pos, epos=init_jaw_center_pos - grasp_dir * .05).attach_to(base)
        # gic.gen_meshmodel().attach_to(base)
        # continue

        is_hnd_collision = gic.is_collided(tube_rack.gen_collision_model(gen_tube=True)[1].cm_list)
        if is_hnd_collision:
            # gic.gen_meshmodel(rgba=[1, 0, 0, .3]).attach_to(base)
            pass
        else:
            init_jaw_center_approach_pos = init_jaw_center_pos - init_jaw_center_rotmat[:3, 2] * \
                                           APPROACH_DISTANCE_LIST[3][
                                               0]
            # approach_jnt_values = rbthp.sim_rbt.ik("arm",
            #                                        init_jaw_center_approach_pos,
            #                                        init_jaw_center_rotmat,
            #                                        # seed_jnt_values=jnt_values.copy(),
            #                                        local_minima="end")
            approach_jnt_values = rbthp.ik(init_jaw_center_approach_pos,
                                           init_jaw_center_rotmat, )

            if approach_jnt_values is not None:
                rbthp.fk(approach_jnt_values)
                rbthp.gen_mesh_model().attach_to(base)
                # jnt_values = rbthp.sim_rbt.ik("arm",
                #                               init_jaw_center_pos,
                #                               init_jaw_center_rotmat,
                #                               seed_jnt_values=approach_jnt_values.copy(),
                #                               local_minima="end")
                jnt_values = rbthp.ik(init_jaw_center_pos,
                                      init_jaw_center_rotmat, )

                if jnt_values is not None:
                    gic.gen_meshmodel(rgba=[0, 1, 0, .3]).attach_to(base)
                    return grasp_id
                else:
                    gic.gen_meshmodel(rgba=[1, 1, 0, .3]).attach_to(base)
                    pass
            else:
                gic.gen_meshmodel(rgba=[1, 1, 0, .3]).attach_to(base)
                pass
    return None


def test(base):
    rbthp = init_env_rbt()
    rbtx = rbthp.rbt_con
    rs_pipe = RealSenseD405Dual()
    rl = RackLocatorDual(rbthp.rbt_con, rs_pipe,
                         origin_obs_pose=np.array([0.27, 0.07, OBS_HEIGHT]),
                         hand_to_eye_mat_1=TCP2EYE_MAT_DEVICE_1,
                         hand_to_eye_mat_2=TCP2EYE_MAT_DEVICE_2,
                         toggle_debug=False)
    detector = TestTubeDetector(affine_mat_path=None)
    tube_rack, rack_tf = init_vision_system(rl, detector, rbthp.rbt_con)
    rs_pipe.stop()
    rbthp.gen_mesh_model().attach_to(base)
    tube_rack.gen_mesh_model(gen_tube=True).attach_to(base)

    tube_id = np.array([2, 4])
    if tube_rack.rack_status[tuple(tube_id)] > 0:
        grasp_cache_dict = {
            # purple ring
            1: fs.load_json(path=fs.workdir_data.joinpath("grasps", "lite6", "purple_ring_cap_tube_grasps.json")),
            # blue
            2: fs.load_json(path=fs.workdir_data.joinpath("grasps", "lite6", "blue_cap_tube_grasps.json")),
            # white
            3: fs.load_json(path=fs.workdir_data.joinpath("grasps", "lite6", "white_cap_tube_grasps.json")),
        }
        grasp_info_list = tube_rack.get_grasps_local_state(tube_id,
                                                           grasp_cache_dict[tube_rack.rack_status[tuple(tube_id)]])
        tube_homomat = tube_rack.get_slot_homomat(tube_id)
        grasp_id = grasp_check(tube_homomat[:3, 3], tube_homomat[:3, :3], np.array([0, 0, 1]), grasp_info_list, rbthp,
                               tube_rack)
        if grasp_id is not None:
            grasp = grasp_cache_dict[tube_rack.rack_status[tuple(tube_id)]][grasp_id]
            init_jaw_center_pos = tube_homomat[:3, 3] + tube_homomat[:3, :3].dot(grasp[1])
            init_jaw_center_rotmat = tube_homomat[:3, :3].dot(grasp[2])
            rbtx.move_p(init_jaw_center_pos - init_jaw_center_rotmat[:3, 2] * APPROACH_DISTANCE_LIST[3][0],
                        init_jaw_center_rotmat,
                        speed=300, mvacc=ACC)
            rbtx.move_p(init_jaw_center_pos, init_jaw_center_rotmat, speed=SPEED, mvacc=ACC)

    tube_id = np.array([2, 5])
    if tube_rack.rack_status[tuple(tube_id)] > 0:
        grasp_cache_dict = {
            # purple ring
            1: fs.load_json(path=fs.workdir_data.joinpath("grasps", "lite6", "purple_ring_cap_tube_grasps.json")),
            # blue
            2: fs.load_json(path=fs.workdir_data.joinpath("grasps", "lite6", "blue_cap_tube_grasps.json")),
            # white
            3: fs.load_json(path=fs.workdir_data.joinpath("grasps", "lite6", "white_cap_tube_grasps.json")),
        }
        grasp_info_list = tube_rack.get_grasps_local_state(tube_id,
                                                           grasp_cache_dict[tube_rack.rack_status[tuple(tube_id)]])
        tube_homomat = tube_rack.get_slot_homomat(tube_id)
        grasp_id = grasp_check(tube_homomat[:3, 3], tube_homomat[:3, :3], np.array([0, 0, 1]), grasp_info_list, rbthp,
                               tube_rack)
        if grasp_id is not None:
            grasp = grasp_cache_dict[tube_rack.rack_status[tuple(tube_id)]][grasp_id]
            init_jaw_center_pos = tube_homomat[:3, 3] + tube_homomat[:3, :3].dot(grasp[1])
            init_jaw_center_rotmat = tube_homomat[:3, :3].dot(grasp[2])
            rbtx.move_p(init_jaw_center_pos - init_jaw_center_rotmat[:3, 2] * APPROACH_DISTANCE_LIST[3][0],
                        init_jaw_center_rotmat,
                        speed=300, mvacc=ACC)
            rbtx.move_p(init_jaw_center_pos, init_jaw_center_rotmat, speed=SPEED, mvacc=ACC)
    del rbtx
    base.run()


if __name__ == "__main__":
    from robot_sim.robots.xarm_lite6_wrs import XArmLite6WRSGripper

    # 3D environment
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    rbt = XArmLite6WRSGripper()
    rbt.gen_meshmodel().attach_to(base)
    test(base)
