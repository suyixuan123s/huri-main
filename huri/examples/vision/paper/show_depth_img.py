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
DEBUG = False


def show_depth_img():
    # 3D environment
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    # init the simulation robot
    yumi_robot = ym.Yumi(enable_cc=True)
    # yumi_robot.gen_meshmodel().attach_to(base)
    # Get Data From Camera
    if DEBUG:
        filename = workdir / "data" / "vision_exp" / "20220227-144809.pkl"
        pcd, img, depth_img = vision_read_data(filename)
    else:
        pcd, img, depth_img = vision_pipeline(streamer=SensorMarkerHandler(IP_ADR),
                                              dump_path=SAVE_PATH)

    affine_mat = np.asarray(
        fs.load_json(AFFINE_MAT_PATH)['affine_mat'])
    pcd_r = rm.homomat_transform_points(affine_mat, points=pcd)

    from huri.vision.yolov6.detect import detect
    yolo_img, yolo_result = detect(
        weights=fs.workdir_vision / "yolov6" / "best_depth_bluecap.pt",  # model.pt path(s)
        data=fs.workdir_vision / "yolov6" / 'tube.yaml',
        conf_thres=.6,
        source=img_to_n_channel(depth2gray_map(depth_img).reshape(*depth_img.shape[:2])))

    cv2.imshow("depth", depth2gray_map(depth_img))
    cv2.imshow("gray", enhance_gray_img(img))
    cv2.imshow("yolo", yolo_img)
    cv2.waitKey(0)

    exit(0)
    depth_img_render, _, _, = spawn_window(base)
    gray_img_render, _, _, = spawn_window(base)



    depth_img_tx = gen_img_texture_on_render(depth_img_render)
    set_img_texture(depth2gray_map(depth_img), depth_img_tx)

    gray_img_tx = gen_img_texture_on_render(gray_img_render)
    set_img_texture(img_to_n_channel(enhance_gray_img(img)), gray_img_tx)

    gm.gen_pointcloud(pcd_r, [[0, 0, 0, .3]]).attach_to(base)

    base.run()


if __name__ == "__main__":
    show_depth_img()
