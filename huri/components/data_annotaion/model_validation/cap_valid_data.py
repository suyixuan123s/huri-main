import time
import argparse

import cv2

import huri.core.file_sys as fs
from huri.core.constants import SENSOR_INFO
from huri.components.data_annotaion import *
from huri.components.utils.img_utils import crop_center
from huri.vision.phoxi_capture import vision_pipeline, SensorMarkerHandler


def cap_data(save_path=None, is_color_img=False, ledpowers=(2000,)):
    while True:
        # Capture img
        for ledpower in ledpowers:
            color_img = None
            pcd, texture, depth_img, rgb_texture, extcam_img = vision_pipeline(
                streamer=SensorMarkerHandler(SENSOR_INFO.IP_ADR_DEPTH_SENSOR_DEBUG),
                rgb_texture=is_color_img,
                dump_path=save_path.joinpath(f"{time.strftime('%Y%m%d-%H%M%S')}_ledpower{ledpower}.pkl"),
                ledpower=ledpower)

            # Increase Brightness before display
            disp_img = cv2.equalizeHist(texture)
            if is_color_img:
                color_img = crop_center(extcam_img, 2064, 1544)[0]
            # Show
            # _image = cv2.putText(img, str("a: New Image || q: Quit"), (10, 100), cv2.FONT_HERSHEY_PLAIN, 6,
            #                      (100, 200, 200), 4)
            cv2.imshow(f"Captured Result_GRAY_IMG_ORIGIN {ledpower}.", texture)
            cv2.imshow(f"Captured Result_GRAY_IMG_EQHIST {ledpower}.", disp_img)
            if color_img is not None:
                cv2.imshow(f"Captured Result_COLOR_IMG {ledpower}.", color_img)
            # Keyboard Input
        k = cv2.waitKey(0)
        if k == ord("a"):  # key "a" reduce -1
            continue
        elif k == ord("q"):
            exit(0)
        else:
            pass


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_color_img', type=bool, default=False, help='Whether to capture color image')
    parser.add_argument('--save_path', type=str, default='blue_valid_data',
                        help='The path to save the file. It will be saved in huri/data/data_annotation/$SAVE_PATH')
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_opt()
    # path = DATA_ANNOT_PATH.joinpath(fs.Path(opt.save_path).resolve())
    path = DATA_ANNOT_PATH.joinpath("valid", "color_gray_blue_white_20220616")
    if not path.exists():
        path.mkdir()
    cap_data(save_path=path, is_color_img=True)


if __name__ == "__main__":
    # default command to save to
    # python cap_valid_data.py --save_path=tab_color_valid/blue_valid_data --is_color_img=True
    # python cap_valid_data.py --save_path=blue_valid_data --is_color_img=False
    main()
