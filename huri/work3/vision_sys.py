import time
from typing import Union
import logging

import numpy as np
import cv2

from huri.core.common_import import fs, rm, gm
from robot_con.xarm_lite6.xarm_lite6_x import XArmLite6X
import huri.vision.pnt_utils as pntu
from huri.components.vision.tube_detector import TestTubeDetector, extract
from huri.definitions.rack_def import Rack_Hard_Proto

logger = logging.getLogger(__name__)


class DumbRackDef(object):
    def __init__(self, pcd_template, rack):
        self._pcd_template = pcd_template
        self.rack = rack

    @property
    def _geom(self):
        return self.rack._geom

    def copy(self):
        return self.rack.copy()


class RackLocator(object):
    """
        Locate the pose of the rack
        """

    def __init__(self, arm_x: XArmLite6X,
                 sensor_handler: 'RealSenseD405',
                 origin_obs_pose: np.ndarray,
                 hand_to_eye_mat: np.ndarray,
                 toggle_debug=False):
        self._arm_x = arm_x
        self._sensor_hdl = sensor_handler
        # ensure to remove empty frame
        for _ in range(5):
            self._sensor_hdl.get_pcd_texture_depth()
        self._hand_to_eye_mat = hand_to_eye_mat

        # observation poses
        self.origin_obs_pos = origin_obs_pose
        self._obs_p_list = [origin_obs_pose]
        # self._rack2cam = rm.homomat_from_posrot(self._eye_to_hand_mat[:3, 3])

        self.toggle_debug = toggle_debug
        self._track_pose = None

    def _to_w_coord(self, hand_to_eye_mat: np.ndarray, pcd: np.ndarray, toggle_debug=False) -> np.ndarray:
        rbt_tcp_homomat = rm.homomat_from_posrot(*self._arm_x.get_pose())
        if toggle_debug:
            gm.gen_sphere(rbt_tcp_homomat[:3, 3], ).attach_to(base)
        trans = np.dot(rbt_tcp_homomat, hand_to_eye_mat)
        pcd_t = rm.homomat_transform_points(trans, pcd)
        if toggle_debug:
            gm.gen_sphere(trans[:3, 3], ).attach_to(base)
            gm.gen_frame(trans[:3, 3], trans[:3, :3]).attach_to(base)
        return pcd_t

    def to_w_coord(self, pcd: np.ndarray, toggle_debug=False) -> np.ndarray:
        return self._to_w_coord(self._hand_to_eye_mat, pcd, toggle_debug)

    def add_obs_pos(self, obs_pos: Union[list, np.ndarray]):
        """
        Add observation pose for the XARM
        :return:
        """
        if isinstance(obs_pos, list):
            self._obs_p_list.extend(obs_pos)
        else:
            self._obs_p_list.append(obs_pos)

    def scan(self, toggle=False):
        pcd_rgba_region = None
        pcds = []
        for i in range(len(self._obs_p_list)):
            obs_pos = self._obs_p_list[i]
            suc = self._arm_x.move_p(pos=obs_pos, rot=np.array([0, -3.1415926, 0]), speed=400, mvacc=1500)
            pcd, pcd_color, _, im = self._sensor_hdl.get_pcd_texture_depth()
            pcd_w = self.to_w_coord(pcd)
            pcd_rgba = np.concatenate((pcd_w, pcd_color), axis=1)
            if toggle:
                rgba = np.append(pcd_color, np.ones((len(pcd_color), 1)), axis=1)
                gm.gen_pointcloud(pcd_w[:, :3], rgbas=rgba).attach_to(base)
            if i >= 1:
                pcd_rgba_region = np.vstack((pcd_rgba_region, pcd_rgba))
            else:
                pcd_rgba_region = pcd_rgba
            pcds.append(pcd_rgba)
        return pcd_rgba_region, pcds

    def get_pcd_im(self):
        pcd, pcd_color, _, im = self._sensor_hdl.get_pcd_texture_depth()
        pcd_w = self.to_w_coord(pcd)
        return np.concatenate((pcd_w, pcd_color), axis=1), im

    def obs_in_rack_center(self, rack_tf: np.ndarray, toggle=False):
        rack_center_pos = rack_tf[:3, 3]

        angle = rm.angle_between_vectors(np.array([0, 1, 0]), rack_tf[:3, 1])
        rot2rack = rm.rotmat_from_axangle(np.array([0, 0, 1]), angle)
        # euler = rm.rotmat_to_euler()
        cam_obs_pos = np.zeros_like(rack_center_pos)
        cam_obs_pos[:2] = rack_center_pos[:2]
        # w2cam_mat = rm.homomat_from_posrot(*self._arm_x.get_pose()).dot(self._hand_to_eye_mat)
        w2cam_mat_rot = np.array([[3.4641e-07, -1, 1.2e-13],
                                  [-1, -3.4641e-07, -3.4641e-07],
                                  [3.4641e-07, 0, -1]])
        w2cam_obs_mat_rot = rot2rack.dot(w2cam_mat_rot)

        cam_obs_pos[2] = 0.237
        w2cam_obs_homo = rm.homomat_from_posrot(cam_obs_pos, w2cam_obs_mat_rot)
        w2r_obs_homo = w2cam_obs_homo.dot(np.linalg.inv(self._hand_to_eye_mat))
        # euler = np.array([0, 0, 0])
        suc = self._arm_x.move_p(pos=w2r_obs_homo[:3, 3], rot=w2r_obs_homo[:3, :3], speed=400, mvacc=1500)
        pcd, pcd_color, _, im = self._sensor_hdl.get_pcd_texture_depth()
        pcd_w = self.to_w_coord(pcd)
        pcd_rgba = np.concatenate((pcd_w, pcd_color), axis=1)
        if toggle:
            gm.gen_rgb_pointcloud(pcd_rgba).attach_to(base)
        return pcd_rgba, im

    def locate_rack(self, pcd_region: np.ndarray,
                    rack_proto,
                    height_range: Union[tuple, list] = (.055, 0.075),
                    rack_init_pose: np.ndarray = None,
                    toggle_debug: bool = False,
                    downsampling_voxelsize=.004,
                    maximum_distance=.003):
        pcd_ind = pntu.extract_pcd_by_range(pcd=pcd_region[:, :3],
                                            x_range=[.08, 1],
                                            z_range=height_range,
                                            toggle_debug=toggle_debug)
        raw_rack_pcd = pcd_region[:, :3][pcd_ind]
        if rack_init_pose is None:
            rack_transform = extract.oriented_box_icp(pcd=raw_rack_pcd,
                                                      pcd_template=rack_proto._pcd_template,
                                                      downsampling_voxelsize=downsampling_voxelsize,
                                                      toggle_debug=toggle_debug)
        else:
            rack_transform = extract.pcd2pcd_icp_match(rack_proto._pcd_template,
                                                       raw_rack_pcd,
                                                       rack_init_pose,
                                                       maximum_distance=maximum_distance)

        if toggle_debug:
            gm.gen_pointcloud(rm.homomat_transform_points(rack_transform, rack_proto._pcd_template),
                              rgbas=[[1, 0, 0, 1]]).attach_to(base)

        rack_height = rack_proto._geom.height
        rack_transform[:3, 3] = rack_transform[:3, 3] - rack_transform[:3, 2] * rack_height

        rack = rack_proto.copy()
        rack.set_homomat(rack_transform)
        return rack, rack_transform

    def get_obs_xy_point(self, xy_point: np.ndarray):
        # euler = rm.rotmat_to_euler()
        rbt_pose_homo = rm.homomat_from_posrot(*self._arm_x.get_pose())
        eye_homomat = np.dot(rbt_pose_homo, self._hand_to_eye_mat)
        eye_homomat[:2, 3] = xy_point
        new_tcp_pose = np.dot(eye_homomat, np.linalg.inv(self._hand_to_eye_mat))
        return new_tcp_pose

    def track(self, rack_proto, rack_init_pose, height_range, rack_state=None, toggle_pcd=False, maximum_distance=.003):
        # self._track_pose = rack_init_pose
        pcd, pcd_color, _, im = self._sensor_hdl.get_pcd_texture_depth()
        pcd_w = self.to_w_coord(pcd)

        if False:
            cam_pose = rm.homomat_from_posrot(*self._arm_x.get_pose()).dot(self._hand_to_eye_mat)
            wcam2rack_pose = np.linalg.inv(rack_init_pose).dot(cam_pose)
            uvw = rm.homomat_transform_points(np.dot(self._sensor_hdl.intr_mat, wcam2rack_pose[:3, :]),
                                              rack_proto._pcd_template + np.array([0, 0, rack_proto._geom.height]))
            uv = uvw[:, :2] * (1 / uvw[:, 2][0])
            rack_proto_pcd_v = rack_proto._pcd_template[
                (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < im.shape[1]) & (uv[:, 1] < im.shape[0])]

            slot_dim = rack_proto._geom.slots_geoms[0][0].dim + np.array([0.005, 0.005])
            for _ in np.vstack(np.where(rack_state > 0)).T:
                tube_slot_center = rack_proto.get_slot_homomat(tuple(_))[:2, 3]
                tube_slot_center_ul = tube_slot_center - slot_dim / 2
                tube_slot_center_br = tube_slot_center + slot_dim / 2

                rack_proto_pcd_v = rack_proto_pcd_v[~((rack_proto_pcd_v[:, 0] > tube_slot_center_ul[0]) & (
                        rack_proto_pcd_v[:, 0] < tube_slot_center_br[0]) & (
                                                              rack_proto_pcd_v[:, 1] > tube_slot_center_ul[1]) & (
                                                              rack_proto_pcd_v[:, 1] < tube_slot_center_br[1]))]
            dbrack = DumbRackDef(pcd_template=rack_proto_pcd_v, rack=rack_proto)
        rack_init_pose = rack_init_pose.copy()
        rack_init_pose[:3, 3] = rack_init_pose[:3, 3] + rack_init_pose[:3, 2] * rack_proto._geom.height
        if toggle_pcd:
            # gm.gen_frame(wcam2rack_pose[:3, 3], wcam2rack_pose[:3, :3]).attach_to(base)
            # gm.gen_pointcloud(rack_proto_pcd_v + np.array([0, 0, rack_proto._geom.height])).attach_to(base)
            # gm.gen_pointcloud(pcd).attach_to(base)
            pcd_rgba = np.concatenate((pcd_w, pcd_color), axis=1)
            gm.gen_rgb_pointcloud(pcd_rgba).attach_to(base)
            # gm.gen_pointcloud(rm.homomat_transform_points(rack_init_pose, rack_proto_pcd_v)).attach_to(base)
            # base.run()

        # rack, rack_tf = self.locate_rack(pcd_w, height_range=height_range,
        #                                  rack_proto=dbrack,
        #                                  rack_init_pose=rack_init_pose,
        #                                  toggle_debug=toggle_pcd)
        rack, rack_tf = self.locate_rack(pcd_w, height_range=height_range,
                                         rack_proto=rack_proto,
                                         rack_init_pose=rack_init_pose,
                                         toggle_debug=toggle_pcd,
                                         maximum_distance=maximum_distance)
        return rack, rack_tf, np.hstack((pcd_w, pcd_color)), im


class RackLocatorDual(object):
    """
    Locate the pose of the rack
    """

    def __init__(self, arm_x: XArmLite6X,
                 sensor_handler: 'RealSenseD405Dual',
                 origin_obs_pose: np.ndarray,
                 hand_to_eye_mat_1: np.ndarray,
                 hand_to_eye_mat_2: np.ndarray,
                 toggle_debug=False):
        self._arm_x = arm_x
        self._sensor_hdl = sensor_handler
        # ensure to remove empty frame
        for _ in range(5):
            self._sensor_hdl.get_pcd_texture_depth(device_id=1)
            self._sensor_hdl.get_pcd_texture_depth(device_id=2)
        self._hand_to_eye_mat_1 = hand_to_eye_mat_1
        self._hand_to_eye_mat_2 = hand_to_eye_mat_2

        # observation poses
        self.origin_obs_pos = origin_obs_pose
        self._obs_p_list = [origin_obs_pose]
        # self._rack2cam = rm.homomat_from_posrot(self._eye_to_hand_mat[:3, 3])

        self.toggle_debug = toggle_debug
        self._track_pose = None

    def _to_w_coord(self, hand_to_eye_mat: np.ndarray, pcd: np.ndarray, toggle_debug=False) -> np.ndarray:
        rbt_tcp_homomat = rm.homomat_from_posrot(*self._arm_x.get_pose())
        if toggle_debug:
            gm.gen_sphere(rbt_tcp_homomat[:3, 3], ).attach_to(base)
        trans = np.dot(rbt_tcp_homomat, hand_to_eye_mat)
        pcd_t = rm.homomat_transform_points(trans, pcd)
        if toggle_debug:
            gm.gen_sphere(trans[:3, 3], ).attach_to(base)
            gm.gen_frame(trans[:3, 3], trans[:3, :3]).attach_to(base)
        return pcd_t

    def to_w_coord_1(self, pcd: np.ndarray, toggle_debug=False) -> np.ndarray:
        return self._to_w_coord(self._hand_to_eye_mat_1, pcd, toggle_debug)

    def to_w_coord_2(self, pcd: np.ndarray, toggle_debug=False) -> np.ndarray:
        return self._to_w_coord(self._hand_to_eye_mat_2, pcd, toggle_debug)

    def add_obs_pos(self, obs_pos: Union[list, np.ndarray]):
        """
        Add observation pose for the XARM
        :return:
        """
        if isinstance(obs_pos, list):
            self._obs_p_list.extend(obs_pos)
        else:
            self._obs_p_list.append(obs_pos)

    def scan(self, toggle=False):
        pcd_rgba_region = None
        pcds = []
        for i in range(len(self._obs_p_list)):
            obs_pos = self._obs_p_list[i]
            suc = self._arm_x.move_p(pos=obs_pos, rot=np.array([0, -3.1415926, 0]), speed=400, mvacc=1500)
            pcd_1, pcd_color_1, _, im_1 = self._sensor_hdl.get_pcd_texture_depth(device_id=1)
            pcd_w_1 = self.to_w_coord_1(pcd_1)

            pcd_2, pcd_color_2, _, im_2 = self._sensor_hdl.get_pcd_texture_depth(device_id=2)
            pcd_w_2 = self.to_w_coord_2(pcd_2)
            pcd_w = np.vstack((pcd_w_1, pcd_w_2))
            pcd_color = np.vstack((pcd_color_1, pcd_color_2))
            pcd_rgba = np.concatenate((pcd_w, pcd_color), axis=1)
            if toggle:
                rgba = np.append(pcd_color, np.ones((len(pcd_color), 1)), axis=1)
                gm.gen_pointcloud(pcd_w[:, :3], rgbas=rgba).attach_to(base)
            if i >= 1:
                pcd_rgba_region = np.vstack((pcd_rgba_region, pcd_rgba))
            else:
                pcd_rgba_region = pcd_rgba
            pcds.append(pcd_rgba)
        return pcd_rgba_region, pcds

    def get_pcd_im_1(self):
        pcd, pcd_color, _, im = self._sensor_hdl.get_pcd_texture_depth(device_id=1)
        pcd_w = self.to_w_coord_1(pcd)
        return np.concatenate((pcd_w, pcd_color), axis=1), im

    def get_pcd_im_2(self):
        pcd, pcd_color, _, im = self._sensor_hdl.get_pcd_texture_depth(device_id=2)
        pcd_w = self.to_w_coord_2(pcd)
        return np.concatenate((pcd_w, pcd_color), axis=1), im

    def obs_in_rack_center(self, rack_tf: np.ndarray, toggle=False):
        rack_center_pos = rack_tf[:3, 3]

        angle = rm.angle_between_vectors(np.array([0, 1, 0]), rack_tf[:3, 1])
        rot2rack = rm.rotmat_from_axangle(np.array([0, 0, 1]), angle)
        # euler = rm.rotmat_to_euler()
        cam_obs_pos = np.zeros_like(rack_center_pos)
        cam_obs_pos[:2] = rack_center_pos[:2]
        # w2cam_mat = rm.homomat_from_posrot(*self._arm_x.get_pose()).dot(self._hand_to_eye_mat)
        w2cam_mat_rot = np.array([[3.4641e-07, -1, 1.2e-13],
                                  [-1, -3.4641e-07, -3.4641e-07],
                                  [3.4641e-07, 0, -1]])
        w2cam_obs_mat_rot = rot2rack.dot(w2cam_mat_rot)

        cam_obs_pos[2] = 0.237
        w2cam_obs_homo = rm.homomat_from_posrot(cam_obs_pos, w2cam_obs_mat_rot)
        w2r_obs_homo = w2cam_obs_homo.dot(np.linalg.inv(self._hand_to_eye_mat_1))
        # euler = np.array([0, 0, 0])
        suc = self._arm_x.move_p(pos=w2r_obs_homo[:3, 3], rot=w2r_obs_homo[:3, :3], speed=400, mvacc=1500)
        pcd, pcd_color, _, im = self._sensor_hdl.get_pcd_texture_depth()
        pcd_w = self.to_w_coord_1(pcd)
        pcd_rgba = np.concatenate((pcd_w, pcd_color), axis=1)
        if toggle:
            gm.gen_rgb_pointcloud(pcd_rgba).attach_to(base)
        return pcd_rgba, im

    def locate_rack(self, pcd_region: np.ndarray,
                    rack_proto,
                    height_range: Union[tuple, list] = (.055, 0.075),
                    rack_init_pose: np.ndarray = None,
                    toggle_debug: bool = False,
                    downsampling_voxelsize=.004,
                    maximum_distance=.003):
        pcd_ind = pntu.extract_pcd_by_range(pcd=pcd_region[:, :3],
                                            x_range=[.08, 1],
                                            z_range=height_range,
                                            toggle_debug=toggle_debug)
        raw_rack_pcd = pcd_region[:, :3][pcd_ind]
        if rack_init_pose is None:
            rack_transform = extract.oriented_box_icp(pcd=raw_rack_pcd,
                                                      pcd_template=rack_proto._pcd_template,
                                                      downsampling_voxelsize=downsampling_voxelsize,
                                                      toggle_debug=toggle_debug)
        else:
            rack_transform = extract.pcd2pcd_icp_match(rack_proto._pcd_template,
                                                       raw_rack_pcd,
                                                       rack_init_pose,
                                                       maximum_distance=maximum_distance)

        if toggle_debug:
            gm.gen_pointcloud(rm.homomat_transform_points(rack_transform, rack_proto._pcd_template),
                              rgbas=[[1, 0, 0, 1]]).attach_to(base)

        rack_height = rack_proto._geom.height
        rack_transform[:3, 3] = rack_transform[:3, 3] - rack_transform[:3, 2] * rack_height

        rack = rack_proto.copy()
        rack.set_homomat(rack_transform)
        return rack, rack_transform

    def get_obs_xy_point(self, xy_point: np.ndarray):
        # euler = rm.rotmat_to_euler()
        rbt_pose_homo = rm.homomat_from_posrot(*self._arm_x.get_pose())
        eye_homomat = np.dot(rbt_pose_homo, self._hand_to_eye_mat_1)
        eye_homomat[:2, 3] = xy_point
        new_tcp_pose = np.dot(eye_homomat, np.linalg.inv(self._hand_to_eye_mat_1))
        return new_tcp_pose

    def track(self, rack_proto, rack_init_pose, height_range, rack_state=None, toggle_pcd=False, maximum_distance=.003):
        # self._track_pose = rack_init_pose
        pcd, pcd_color_1, _, im1 = self._sensor_hdl.get_pcd_texture_depth(device_id=1)
        pcd_w_1 = self.to_w_coord_1(pcd)
        pcd, pcd_color_2, _, im2 = self._sensor_hdl.get_pcd_texture_depth(device_id=2)
        pcd_w_2 = self.to_w_coord_2(pcd)
        pcd_w = np.vstack((pcd_w_1, pcd_w_2))
        pcd_color = np.vstack((pcd_color_1, pcd_color_2))

        if False:
            cam_pose = rm.homomat_from_posrot(*self._arm_x.get_pose()).dot(self._hand_to_eye_mat)
            wcam2rack_pose = np.linalg.inv(rack_init_pose).dot(cam_pose)
            uvw = rm.homomat_transform_points(np.dot(self._sensor_hdl.intr_mat, wcam2rack_pose[:3, :]),
                                              rack_proto._pcd_template + np.array([0, 0, rack_proto._geom.height]))
            uv = uvw[:, :2] * (1 / uvw[:, 2][0])
            rack_proto_pcd_v = rack_proto._pcd_template[
                (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < im.shape[1]) & (uv[:, 1] < im.shape[0])]

            slot_dim = rack_proto._geom.slots_geoms[0][0].dim + np.array([0.005, 0.005])
            for _ in np.vstack(np.where(rack_state > 0)).T:
                tube_slot_center = rack_proto.get_slot_homomat(tuple(_))[:2, 3]
                tube_slot_center_ul = tube_slot_center - slot_dim / 2
                tube_slot_center_br = tube_slot_center + slot_dim / 2

                rack_proto_pcd_v = rack_proto_pcd_v[~((rack_proto_pcd_v[:, 0] > tube_slot_center_ul[0]) & (
                        rack_proto_pcd_v[:, 0] < tube_slot_center_br[0]) & (
                                                              rack_proto_pcd_v[:, 1] > tube_slot_center_ul[1]) & (
                                                              rack_proto_pcd_v[:, 1] < tube_slot_center_br[1]))]
            dbrack = DumbRackDef(pcd_template=rack_proto_pcd_v, rack=rack_proto)
        rack_init_pose = rack_init_pose.copy()
        rack_init_pose[:3, 3] = rack_init_pose[:3, 3] + rack_init_pose[:3, 2] * rack_proto._geom.height
        if toggle_pcd:
            # gm.gen_frame(wcam2rack_pose[:3, 3], wcam2rack_pose[:3, :3]).attach_to(base)
            # gm.gen_pointcloud(rack_proto_pcd_v + np.array([0, 0, rack_proto._geom.height])).attach_to(base)
            # gm.gen_pointcloud(pcd).attach_to(base)
            pcd_rgba = np.concatenate((pcd_w, pcd_color), axis=1)
            gm.gen_rgb_pointcloud(pcd_rgba).attach_to(base)
            # gm.gen_pointcloud(rm.homomat_transform_points(rack_init_pose, rack_proto_pcd_v)).attach_to(base)
            # base.run()

        # rack, rack_tf = self.locate_rack(pcd_w, height_range=height_range,
        #                                  rack_proto=dbrack,
        #                                  rack_init_pose=rack_init_pose,
        #                                  toggle_debug=toggle_pcd)
        rack, rack_tf = self.locate_rack(pcd_w, height_range=height_range,
                                         rack_proto=rack_proto,
                                         rack_init_pose=rack_init_pose,
                                         toggle_debug=toggle_pcd,
                                         maximum_distance=maximum_distance)
        return rack, rack_tf, np.hstack((pcd_w_1, pcd_color_1)), im1


def detect_tube(pcd: np.ndarray, img: np.ndarray, rack_proto, rack_tf, detector: TestTubeDetector = None,
                toggle_detect_tube_pos=False,
                toggle_yolo: bool = False):
    if detector is None:
        detector = TestTubeDetector(affine_mat_path=None)
    yolo_img, yolo_results = detector.yolo_detect(texture_img=img,
                                                  yolo_weights_path="E:\\huri_shared\\huri\\work3\\best.pt",
                                                  imgsz=(1280, 1280), )
    if toggle_yolo:
        cv2.imshow("YOLO Results", yolo_img)
        cv2.waitKey(0)
    detected_results, rack_instance, rack_tf = detector.analyze_tubes_given_rack_tf_yolo(rack_proto=rack_proto.copy(),
                                                                                         rack_tf=rack_tf,
                                                                                         pcd=pcd,
                                                                                         yolo_results=yolo_results,
                                                                                         yolo_img=yolo_img,
                                                                                         downsampling_voxelsize=.003,
                                                                                         toggle_detect_tube_pos=toggle_detect_tube_pos)
    return rack_instance, rack_tf


def is_pose_err(homomat1, homomat2):
    err = np.zeros(6)
    err[0:3] = (homomat1[:3, 3] - homomat2[:3, 3])
    err[3:6] = rm.deltaw_between_rotmat(homomat1[:3, :3], homomat1[:3, :3].T)

    err_pos = err[:3]
    err_rot = err[3:6]
    errnorm_pos = err_pos.T.dot(err_pos)
    errnorm_rot = np.linalg.norm(err_rot)
    print(errnorm_pos, errnorm_rot)
    if errnorm_pos > 1e-3:
        print("Pos error")
        return True
    elif errnorm_rot > 3.5:
        print("Rot error")
        return True

    return False
