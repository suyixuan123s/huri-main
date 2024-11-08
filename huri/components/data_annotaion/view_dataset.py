import copy

import cv2
import numpy as np

from basis.data_adapter import gen_colorarray

from huri.vision.phoxi_capture import (vision_pipeline,
                                       SensorMarkerHandler,
                                       vision_read_data,
                                       depth2gray_map,
                                       enhance_gray_img, )
from huri.core.common_import import *
from huri.components.utils.panda3d_utils import img_to_n_channel
from huri.vision.yolov6.detect import detect
from huri.test.vision.test_calibration import increase_brightness_constrast
from huri.components.utils.annotation_utils import read_yolo_annotation_file, bboxes_xywh2xyxy, scale_label_format
from huri.components.utils.img_utils import letterbox, crop_center


def model_comp(data_path_list: list,
               img_type="gray",):

    print(len(data_path_list))
    for iii, data_path in enumerate(data_path_list):
        # if iii < 7:
        #     continue
        if fs.Path(data_path).name.split(".")[-1] == "pkl":
            pcd, img, depth_img, _, ext_cam_img = vision_read_data(data_path)
            if img_type == "gray":
                img = img_to_n_channel(enhance_gray_img(img.reshape(img.shape[:2])))
            elif "color" in img_type:
                img = crop_center(ext_cam_img, *(2064, 1544))[0]
                if "gray" in img_type:
                    img = img_to_n_channel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            elif img_type == "depth":
                img = img_to_n_channel(depth2gray_map(depth_img))
            else:
                raise Exception("Image type only support gray and color!!")
        else:
            img = cv2.imread(str(data_path))
        # img = increase_brightness_constrast(img, beta=30)
        cv2.imshow("img", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    is_analy_mode = True
    if is_analy_mode:
        # root = fs.workdir_data.joinpath("data_annotation", "blue_valid_data").glob("**/*")
        # root = fs.workdir.joinpath("components","exe","debug_data").glob("*.pkl")
        root = fs.workdir_data.joinpath("data_annotation", "blue_white_valid_data").glob("**/*")
        # root = fs.workdir_data.joinpath("data_annotation", "tab_color_valid", "blue_valid_data").glob("**/*")
        # root = fs.workdir_data.joinpath("data_annotation", "tab_color_valid", "blue_white_valid_data").glob("**/*")
        # root = fs.workdir_data.joinpath("data_annotation", "blue_white_purple_valid_data").glob("**/*")
        # root = list(fs.Path("D:\\chen\\yolo_huri\\paper\\dataset9\\images").glob("**/2022*"))
        data_path_list = list(root)
        print(data_path_list)
        res_analy = model_comp(data_path_list=data_path_list,)
