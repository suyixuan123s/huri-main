"""
This is an example to capture the image from phoxi and external rgb camera
Author: Chen Hao
Email: chen960216@gmail.com
"""
from typing import Union
import cv2
import os

import huri.core.file_sys as fs
from huri.core.constants import SENSOR_INFO
from huri.components.utils.img_utils import scale_img
from huri.vision.phoxi_capture import vision_pipeline


def capture_images(image_save_path: Union[str, fs.Path],
                   depth_sensor_streamer=None,
                   rgb_cam_streamer=None,
                   capture_interval=1000,  # ms
                   clear_folder=True,
                   ):
    """
    Capture images from depth sensor and webcam
    :param image_save_path: The path to save the image
    :param rgb_cam_streamer:  RGB camera streamer
    :param capture_interval: Time inverval between two captures
    :param clear_folder: Clear images in the save path folder
    :return:
    """
    if depth_sensor_streamer is None:
        raise Exception("No streamer for the RGB camera")
    if rgb_cam_streamer is None:
        raise Exception("No streamer for the RGB camera")
    image_save_path = fs.Path(image_save_path)
    # remove old images in the dir
    if clear_folder:
        images_files = image_save_path.glob("**/*")
        for file_path in images_files:
            if not file_path.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                continue
            try:
                os.remove(file_path)
            except:
                print("Error while deleting file : ", file_path)

    # add new images
    save_counter = 1
    while True:
        # depth sensor acquire data
        _, gray_texture, _, _, _ = vision_pipeline(depth_sensor_streamer, dump_path=None, rgb_texture=False)
        # rgb camera acquire data
        rgb_img = rgb_cam_streamer.get_img()
        cv2.imshow('gray image caputured from depth sensor', scale_img(enhance_gray_img(gray_texture), scale=.4))
        cv2.imshow('rgb image caputured from webcam', scale_img(rgb_img, scale=.4))

        k = cv2.waitKey(0)
        if k == 27:  # Esc to quit
            break
        if k == 32:  # Space to save image
            cv2.imwrite(str(image_save_path / f"frame{save_counter}.jpg"), gray_texture)
            cv2.imwrite(str(image_save_path / f"image{save_counter}.jpg"), rgb_img)
            print(f"Save {save_counter} image")
            save_counter += 1
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import drivers.rpc.extcam.cameras as camera
    from huri.vision.phoxi_capture import SensorMarkerHandler, enhance_gray_img

    depth_sensor_streamer = SensorMarkerHandler(SENSOR_INFO.IP_ADR_DEPTH_SENSOR_DEBUG)
    # remember to set `calibpath` of phoxi server to empty to suppress using the RGB cam
    rgb_cam_streamer = camera.ExtCam()
    #
    image_save_path = fs.Path(__file__).parent / "data"
    capture_images(image_save_path=image_save_path,
                   depth_sensor_streamer=depth_sensor_streamer,
                   rgb_cam_streamer=rgb_cam_streamer, )

    # image_save_path = fs.Path(image_save_path)
    # images_files = image_save_path.glob("**/*")
    # for file_path in images_files:
    #     if not file_path.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
    #         continue
    #     if "frame" in file_path.name:
    #         img = enhance_gray_img(cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE))
    #         # img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
    #         cv2.imwrite(str(image_save_path / file_path.name), img)
