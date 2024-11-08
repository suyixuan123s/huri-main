import time
from typing import Literal
import cv2
from drivers.devices.realsense_d405.d405_driver import RealSenseD405, find_devices

# device id. Use find_devices() to check device id
DEVICE_1_ID = '218622271111'  # right realsense
DEVICE_2_ID = '230322271843'  # left realsense


class RealSenseD405Dual(object):
    def __init__(self, device_1_id=DEVICE_1_ID, device_2_id=DEVICE_2_ID, resolution_1: Literal['mid', 'high'] = 'high',
                 resolution_2: Literal['mid', 'high'] = 'high'):
        self._rs1 = RealSenseD405(resolution=resolution_1, device=device_1_id)
        time.sleep(1)
        self._rs2 = RealSenseD405(resolution=resolution_2, device=device_2_id)
        self._devices = [self._rs1, self._rs2]

    def get_color_img(self, device_id=1):
        assert device_id == 1 or device_id == 2
        rs: RealSenseD405 = self._devices[device_id - 1]
        return rs.get_color_img()

    def get_pcd_texture_depth(self, device_id=1):
        assert device_id == 1 or device_id == 2
        rs: RealSenseD405 = self._devices[device_id - 1]
        return rs.get_pcd_texture_depth()

    def stop(self):
        self._rs1.stop()
        self._rs2.stop()


if __name__ == "__main__":
    rsd = RealSenseD405Dual()
    for i in range(20):
        rsd.get_pcd_texture_depth(device_id=1)
        time.sleep(.1)
        rsd.get_pcd_texture_depth(device_id=2)
        time.sleep(.1)
