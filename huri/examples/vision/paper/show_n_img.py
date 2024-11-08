import matplotlib
from time import strftime

from huri.core.file_sys import workdir
from huri.core.base_boost import render_img, spawn_window, gen_img_texture_on_render, set_img_texture
from huri.vision.phoxi_capture import (vision_pipeline,
                                       SensorMarkerHandler,
                                       vision_read_data,
                                       depth2gray_map,
                                       enhance_gray_img, )
from huri.core.common_import import *
from huri.core.utils import img_to_n_channel

matplotlib.use('TkAgg')

IP_ADR = "192.168.125.100:18300"
SAVE_PATH = workdir / "data" / "vision_exp" / f"{strftime('%Y%m%d-%H%M%S')}.pkl"  # None: Do not save the data captured by phoxi
AFFINE_MAT_PATH = workdir / "data/calibration/qaqqq.json"

from huri.vision.yolov6.detect import detect


def scale_img(img, scale=1.):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized_img


def show_depth_img():
    FILENAMES = fs.workdir.resolve(["data", "vision_exp"]).glob("**/202203*.pkl")
    # FILENAMES = ["20220227-150428.pkl",
    #              "20220227-150458.pkl",
    #              "20220227-150542.pkl",
    #              "20220227-150607.pkl",
    #              "20220227-150644.pkl",
    #              "20220227-150715.pkl",
    #              "20220227-151234.pkl",
    #              "20220227-151302.pkl",
    #              "20220227-151323.pkl",
    #              "20220227-151356.pkl",
    #              "20220227-205406.pkl",
    #              "20220227-205428.pkl",
    #              "20220227-205512.pkl",
    #              "20220227-205602.pkl",
    #              "20220227-205626.pkl",
    #              "20220226-231353.pkl",
    #              "20220226-232151.pkl",
    #              "20220227-143956.pkl",
    #              "20220227-144026.pkl",
    #              "20220227-144343.pkl",
    #              "20220227-144418.pkl",
    #              "20220227-144609.pkl",
    #              "20220227-144654.pkl",
    #              "20220227-144718.pkl",
    #              "20220227-144746.pkl",
    #              "20220227-144809.pkl",
    #              "20220227-145535.pkl",
    #              "20220227-150330.pkl",
    #              "20220227-150400.pkl", ]
    for filename in FILENAMES:
        filepath = workdir / "data" / "vision_exp" / filename
        pcd, img, depth_img = vision_read_data(filepath)
        if img is None:
            continue
        yolo_img, yolo_result = detect(
            weights=fs.workdir_vision / "yolov6" / "bluecap_1376_richdata.pt",  # model.pt path(s)
            # weights=fs.workdir_vision / "yolov6" / "bluecap_1280_richdata.pt",  # model.pt path(s)
            # data=fs.workdir_vision / "yolov6" / 'tube3.yaml',
            # conf_thres=.5  ,
            # imgsz=(1280, 1280),
            # iou_thres=.03,
            source=img_to_n_channel(enhance_gray_img(img.reshape(img.shape[:2]))))


        # yolo_img2, yolo_result = detect(
        #     # weights=fs.workdir_vision / "yolov6" / "bluecap_1280_2.pt",  # model.pt path(s)
        #     weights=fs.workdir_vision / "yolov6" / "bluecap_1280_richdata.pt",  # model.pt path(s)
        #     # data=fs.workdir_vision / "yolov6" / 'tube3.yaml',
        #     # conf_thres=.5  ,
        #     imgsz=(1280, 1280),
        #     # iou_thres=.03,
        #     source=img_to_n_channel(enhance_gray_img(img.reshape(img.shape[:2]))))

        # if len(yolo_result) == 0:
        #     continue
        # cv2.imshow("depth", depth2gray_map(depth_img))
        # cv2.imshow("gray", enhance_gray_img(img))
        # yolo_img2, yolo_result = detect(
        #     weights=fs.workdir_vision / "yolov6" / "best_depth2.pt",  # model.pt path(s)
        #     # data=fs.workdir_vision / "yolov6" / 'tube3.yaml',
        #     conf_thres=.5,
        #     iou_thres=.03,
        #     source=img_to_n_channel(depth2gray_map(depth_img).reshape(*depth_img.shape[:2])))

        cv2.imshow("yolo", scale_img(yolo_img, .7))
        # cv2.imshow("yolo2", scale_img(yolo_img2, .7))
        cv2.waitKey(0)


if __name__ == "__main__":
    show_depth_img()
