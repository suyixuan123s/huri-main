"""
This is an example to capture RGB image from webcam
"""
from typing import Union
from time import strftime
import cv2
import os

import huri.core.file_sys as fs


def capture_images(image_save_path: Union[str, fs.Path],
                   rgbcam_streamer=None,
                   capture_interval=1000,  # ms
                   clear_folder = True,
                   ):
    """
    Capture images from webcam
    :param image_save_path: The path to save the image
    :param rgbcam_streamer:  RGB camera streamer
    :param capture_interval: Time inverval between two captures
    :param clear_folder: Clear images in the save path folder
    :return:
    """
    if rgbcam_streamer is None:
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
    while True:
        rgbcam_img = rgbcam_streamer.get_img()
        cv2.imshow('rgb image caputured from webcam', rgbcam_img)
        k = cv2.waitKey(capture_interval)
        if k == 27:  # Esc to quit
            break
        cv2.imwrite(str(image_save_path / f"{strftime('%Y%m%d-%H%M%S')}.jpg"), rgbcam_img)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import drivers.rpc.extcam.cameras as camera

    image_save_path = fs.Path(__file__).parent / "data"
    capture_images(image_save_path=image_save_path,
                   rgbcam_streamer=camera.ExtCam())
