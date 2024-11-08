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

IP_ADR = "127.0.0.1:18300"
SAVE_PATH = workdir / "data" / "vision_exp" / f"{strftime('%Y%m%d-%H%M%S')}.pkl"  # None: Do not save the data captured by phoxi
AFFINE_MAT_PATH = workdir / "data/calibration/qaqqq.json"
DEBUG = False


def show_img():
    # 3D environment
    # base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    # init the simulation robot
    # yumi_robot = ym.Yumi(enable_cc=True)
    # yumi_robot.gen_meshmodel().attach_to(base)
    # Get Data From Camera
    if DEBUG:
        filename = workdir / "data" / "vision_exp" / "20220309-141828.pkl"
        pcd, img, depth_img = vision_read_data(filename)
    else:
        pcd, img, depth_img = vision_pipeline(streamer=SensorMarkerHandler(IP_ADR),
                                              dump_path=SAVE_PATH)

    affine_mat = np.asarray(
        fs.load_json(AFFINE_MAT_PATH)['affine_mat'])
    pcd_r = rm.homomat_transform_points(affine_mat, points=pcd)

    from huri.vision.yolov6.detect import detect
    from huri.test.vision.test_calibration import increase_brightness_constrast

    img = increase_brightness_constrast(img, beta= 30)

    img = img_to_n_channel(enhance_gray_img(img.reshape(img.shape[:2])))
    print(img.shape)
    # img = increase_brightness_constrast(img, beta=70)
    # img = img_to_n_channel(img.reshape(img.shape[:2]))
    yolo_img, yolo_result = detect(
        # weights=fs.workdir_vision / "yolov6" / "blue_white_cap_1280.pt",  # model.pt path(s)
        data=fs.workdir_vision / "yolov6" / 'tube.yaml',
        # conf_thres=.1,
        # conf_thres=.1,
        # imgsz=(1280, 1280),
        # iou_thres=.05,
        source=img)

    # cv2.imshow("depth", depth2gray_map(depth_img))
    # cv2.imshow("gray", enhance_gray_img(img))
    cv2.imshow("yolo", yolo_img)



    yolo_img, yolo_result = detect(
        # weights=fs.workdir_vision / "yolov6" / "whitecap_1376_richdata.pt",  # model.pt path(s)
        weights=fs.workdir_vision / "yolov6" / "best_onrack.pt",  # model.pt path(s)
        # data=fs.workdir_vision / "yolov6" / 'tube.yaml',
        # conf_thres=.4,
        # imgsz=(1280, 1280),
        line_thickness=1,
        iou_thres=.4,
        source=img)

    # cv2.imshow("depth2", depth2gray_map(depth_img))
    # cv2.imshow("gray2", enhance_gray_img(img))
    cv2.imshow("best on rack", yolo_img)

    cv2.waitKey(0)

    exit(0)



if __name__ == "__main__":
    show_img()
