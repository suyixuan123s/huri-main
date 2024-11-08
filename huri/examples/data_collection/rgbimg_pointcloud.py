"""
This program finds the corresponding area of a rgb image to the point cloud
Author: Hao Chen
Date: 20220321osaka
"""
import cv2

if __name__ == "__main__":
    from time import strftime

    from huri.vision.phoxi_capture import SensorMarkerHandler, vision_pipeline, vision_read_data, enhance_gray_img
    from huri.core.common_import import fs
    from huri.core.constants import SENSOR_INFO

    DEBUG = True
    SAVE_PATH = fs.workdir / "data" / "vision_exp" / f"{strftime('%Y%m%d-%H%M%S')}.pkl"  # None: Do not save the data captured by phoxi

    sensor_streamer = SensorMarkerHandler(SENSOR_INFO.IP_ADR_DEPTH_SENSOR_DEBUG)
    if not DEBUG:
        pcd, texutre, depth_img, rgb_texture = vision_pipeline(sensor_streamer, dump_path=SAVE_PATH)
    else:
        filepath = fs.workdir / "data" / "vision_exp" / "20220314-095527.pkl"
        pcd, texutre, depth_img, rgb_texture,_ = vision_read_data(filepath)
    cv2.imshow("rgb texture", rgb_texture)
    cv2.waitKey(0)