import os
import time
from time import strftime
from typing import Union

import cv2
import cv2.aruco as aruco
import numpy as np

import drivers.rpc.phoxi.phoxi_client as pcdt
from huri.core.file_sys import dump_pickle, load_pickle, workdir_vision, Path


class PhoxiCapData(object):
    def __init__(self, pcd, texture, depth_img, rgb_texture=None, extcam_img=None):
        # necessary data
        self.pcd = pcd
        self.texture = texture
        self.depth_img = depth_img
        self._len = 3

        # external camera enabled
        self.rgb_texture = rgb_texture
        self.extcam_img = extcam_img
        if rgb_texture is not None:
            self._len += 1
        if self.extcam_img is not None:
            self._len += 1

    def __len__(self, ):
        return self._len


class SensorMarkerHandler(object):

    def __init__(self, ip_adr="192.168.125.100:18300"):
        self.aruco_parameters = aruco.DetectorParameters_create()
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        self.sensor_client = pcdt.PhxClient(host=ip_adr)
        self.aruco_target_id_list = [0, 1]
        self._texture_width = None
        self._texture_height = None

    def get_marker_center(self):
        self.sensor_client.triggerframe()
        img = self.sensor_client.gettextureimg()
        pcd = self.sensor_client.getpcd() * .001
        width = img.shape[1]
        # detect markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(cv2.equalizeHist(img), self.aruco_dict,
                                                              parameters=self.aruco_parameters)
        print(ids)
        if len(corners) < len(self.aruco_target_id_list) or len(ids) != len(self.aruco_target_id_list):
            return None
        if ids[0] not in self.aruco_target_id_list or ids[1] not in self.aruco_target_id_list:
            return None
        center = np.mean(np.mean(corners, axis=0), axis=1)[0].astype(np.int32)
        marker_pos_in_sensor = pcd[width * center[1] + center[0]]
        return marker_pos_in_sensor

    @property
    def texture_size(self):
        """
        Return the size of the texture
        :return: (height, width) of the texture
        """
        if self._texture_width is None or self._texture_height is None:
            texture = self.get_texture()
            height, width = texture.shape[:2]
            self._texture_width = width
            self._texture_height = height
        return (self._texture_height, self._texture_width)

    def get_pcd(self):
        self.sensor_client.triggerframe()
        pcd = self.sensor_client.getpcd() * .001
        return pcd

    def get_texture(self):
        self.sensor_client.triggerframe()
        texture = self.sensor_client.gettextureimg()
        return texture

    def get_depth_img(self):
        self.sensor_client.triggerframe()
        depth_img = self.sensor_client.getdepthimg()
        return depth_img

    def get_pcd_and_texuture(self):
        self.sensor_client.triggerframe()
        pcd = self.sensor_client.getpcd() * .001
        texture = self.sensor_client.gettextureimg()
        return pcd, texture

    def get_pcd_texture_depth(self, rgb_texture: bool = False, get_depth=False, toggle_time=False):
        if toggle_time:
            time_st = time.time()
        self.sensor_client.triggerframe()
        if toggle_time:
            time_trigger = time.time() - time_st
        pcd = self.sensor_client.getpcd() * .001
        texture = self.sensor_client.gettextureimg()
        if get_depth:
            depth_img = self.sensor_client.getdepthimg()
        else:
            depth_img = None
        if rgb_texture:
            rgb_texture = self.sensor_client.getrgbtextureimg()
            extcam_img = self.sensor_client.getextcamimg()
        else:
            rgb_texture = None
            extcam_img = None
        if toggle_time:
            time_get_data = time.time() - time_st - time_trigger
            print(f"Time consumption: Trigger Frame: {time_trigger}, Get Data: {time_get_data}")
        return pcd, texture, depth_img, rgb_texture, extcam_img

    def get_color_texture(self):
        return self.sensor_client.getrgbtextureimg()

    def set_led_power(self, ledpower=2000):
        self.sensor_client.changeledpower(ledpower)


def start_record(streamer, yolo=None):
    while True:
        pcd, texture_img = streamer.get_pcd_and_texuture()
        enhanced_image = cv2.equalizeHist(texture_img)
        if yolo is not None:
            image, results = yolo.detect(np.stack((enhanced_image,) * 3, axis=-1))
            cv2.imshow("captured image detected by yolo", image)
        else:
            image = enhanced_image
            cv2.imshow("captured image enhacned", image)
        pressed_key = cv2.waitKey(0)
        if pressed_key & 0xFF == ord('c'):
            dump_pickle([texture_img, enhanced_image, image, results, pcd],
                        workdir_vision / "data" / f"{strftime('%Y%m%d-%H%M%S')}.pkl")
        elif pressed_key & 0xFF == ord('s'):
            dump_pickle([texture_img, enhanced_image, image, results, pcd],
                        workdir_vision / "data" / f"{strftime('%Y%m%d-%H%M%S')}.pkl")
            exit(0)
        elif pressed_key & 0xFF == ord('q'):
            exit(0)


def vision_pipeline(streamer, dump_path=None, rgb_texture=False, get_depth=True, toggle_time=False):
    # streamer.set_led_power(ledpower)
    pcd, texture, depth_img, rgb_texture, extcam_img = streamer.get_pcd_texture_depth(rgb_texture=rgb_texture,
                                                                                      get_depth=get_depth,
                                                                                      toggle_time=toggle_time)
    if dump_path is not None:
        dump_path = str(dump_path)
        dump_pickle([pcd, texture, depth_img, rgb_texture, extcam_img], dump_path)
    return pcd, texture, depth_img, rgb_texture, extcam_img


def img_pipeline(streamer):
    texture = streamer.get_texture()
    return texture


def vision_read_data(filepath: Union[Path, str]):
    """
    Read the data saved by the function `vision_pipeline`.
    :param filepath: The path of data saved by the function `vision_pipeline`
    :return: point cloud data, gray texture of point cloud data, depth image, rgb texture of point cloud data
    """
    data = load_pickle(filepath)
    if isinstance(data, np.ndarray) or len(data) < 2:
        pcd, img = data, None
    else:
        pcd, img = data[0], data[1]
    if len(data) > 2:
        depth_img = data[2]
    else:
        depth_img = None
    if len(data) > 3:
        rgb_texture = data[3]
    else:
        rgb_texture = None
    if len(data) > 4:
        extcam_img = data[4]
    else:
        extcam_img = None
    return pcd, img, depth_img, rgb_texture, extcam_img


def depth2gray_map(depth_img, max_value=1300):
    depth_img_normalized = ((depth_img / max_value) * 255).astype('uint8').reshape(depth_img.shape[:2])
    # return cv2.applyColorMap(depth_img_normalized, cv2.COLORMAP_JET)
    return depth_img_normalized


def enhance_gray_img(gray_img):
    if len(gray_img.shape) == 2 or gray_img.shape[2] == 1:
        gray_img = gray_img.reshape(gray_img.shape[:2])
    return cv2.equalizeHist(gray_img)


def increase_brightness(img, value=30):
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


if __name__ == "__main__":
    import huri.vision.yolo.detect as yyd

    # streamer = SensorMarkerHandler()
    # # start_record(streamer, yyd)
    # depth_img = streamer.get_depth_img()
    # gray_img = streamer.get_texture()
    # depth_img_normalized = ((depth_img / depth_img.max()) * 255).astype('uint8')
    # cv.imshow("depth img", cv.applyColorMap(depth_img_normalized, cv.COLORMAP_JET))
    # cv.imshow("gray img", gray_img)
    # cv.waitKey(0)
    from huri.core.constants import SENSOR_INFO

    streamer = SensorMarkerHandler(SENSOR_INFO.IP_ADR_DEPTH_SENSOR_DEBUG)
    a = time.time()
    vision_pipeline(streamer, rgb_texture=False)
    b = time.time()
    print(f"Time consuming is {b - a}")
