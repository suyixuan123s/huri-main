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
               model_paths: dict,
               conf_thres=.7,
               analysis_tube: list = None,
               img_type="gray",
               show=True) -> list:
    if analysis_tube is None:
        analysis_tube = []
    # result analysis
    res_analy = {}
    for model_name in model_paths.keys():
        res_analy[model_name] = []

    print(len(data_path_list))
    blue = 0
    white = 0
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

        # v_1 = [len(res_analy[model_name]) for model_name in model_paths.keys()]
        res_stat = {model_name: {t: 0 for t in analysis_tube} for model_name in model_paths.keys()}
        for model_name, model_path in model_paths.items():
            tmp_cnt = res_stat[model_name]
            yolo_img, yolo_result = detect(
                weights=model_path,
                source=img,
                conf_thres=conf_thres,
                # device='cpu',
                visualize=False)
            if model_name not in res_analy:
                res_analy[model_name] = []
            for tube_name in yolo_result.keys():
                if tube_name.split(" ")[0] in analysis_tube:
                    tmp_cnt[tube_name.split(" ")[0]] += 1
                    res_analy[model_name].append(yolo_result[tube_name]['pos'])
            if show:
                cv2.imwrite("r.jpg", yolo_img)
                # cv2.imshow(model_name, letterbox(yolo_img, new_shape=(1500, 1500))[0])
                cv2.imshow(model_name, yolo_img)
                # cv2.imshow(f"{model_name}_org", img)
        print(res_stat)
        # blue += res_stat['20220624_blue_cap_human_grasp']['blue']
        # white += res_stat['20220616_sd_r_color_bg_iter_100']['white']
        # [print(f"{model_name} has {len(res_analy[model_name]) - v_1[i_]} tubes") for i_, model_name in
        #  enumerate(model_paths.keys())]
        if show:
            print(f"{iii} is shown")
            cv2.waitKey(0)
    print(f"blue {blue}, white {white}")
    return res_analy


def plot_analy_res(res_analy: dict, bgnd_img: np.ndarray, color=None, show=True):
    if color is None:
        color = np.array(gen_colorarray(len(res_analy)))[..., :3] * 255
    for i, model_name in enumerate(res_analy.keys()):
        print(f"Number of {model_name} is {len(res_analy[model_name])}")
        average_bbox_size = 0
        num_bbox = 0
        for data in res_analy[model_name]:
            bgnd_img = cv2.rectangle(bgnd_img, data[0], data[1], color=color[i], thickness=2)
            average_bbox_size += np.prod(data[1] - data[0])
            num_bbox += 1
        print(f"average bbox size of {model_name} is {np.sqrt(average_bbox_size / num_bbox)}")
    if show:
        cv2.imshow("res", bgnd_img)
        cv2.waitKey(0)
    return bgnd_img


def plot_annot_distr(label_path_list: list, bgnd_img: np.ndarray, color=None, analysis_tube: list = None, show=True):
    if color is None:
        color = np.array(gen_colorarray(len(res_analy)))[..., :3] * 255
    if analysis_tube is None:
        analysis_tube = []
    average_bbox_size = 0
    num_bbox = 0
    for label_p in label_path_list:
        labels, bboxes_xywh = read_yolo_annotation_file(label_p)
        bboxes_xyxy = scale_label_format(bboxes_xywh2xyxy(bboxes_xywh), bgnd_img.shape, op="scale").astype(int)
        bboxes_xywh = scale_label_format(bboxes_xywh, bgnd_img.shape, op="scale").astype(int)
        for n in range(len(bboxes_xyxy)):
            if bboxes_xywh[n][2] * bboxes_xywh[n][3] < 3000:
                bgnd_img = cv2.rectangle(bgnd_img, bboxes_xyxy[n][0:2], bboxes_xyxy[n][2:4], color=color,
                                         thickness=2)
            average_bbox_size += bboxes_xywh[n][2] * bboxes_xywh[n][3]
            num_bbox += 1
    print(f"average bbox size is {np.sqrt(average_bbox_size / num_bbox)}")
    if show:
        cv2.imshow("res", bgnd_img)
        cv2.waitKey(0)
    return bgnd_img


if __name__ == "__main__":
    is_analy_mode = True
    if is_analy_mode:
        # root = fs.workdir_data.joinpath("data_annotation", "blue_valid_data").glob("**/*")
        # root = fs.workdir.joinpath("components","exe","debug_data").glob("*.pkl")
        # root = fs.workdir_data.joinpath("data_annotation", "blue_white_valid_data").glob("**/*")
        # root = fs.workdir_data.joinpath("data_annotation", "tab_color_valid", "blue_valid_data").glob("**/*")
        root = fs.workdir_data.joinpath("data_annotation", "tab_color_valid", "blue_white_valid_data").glob("**/*")
        # root = fs.workdir_data.joinpath("data_annotation", "blue_white_purple_valid_data").glob("**/*")
        # root = list(fs.Path("D:\\chen\\yolo_huri\\paper\\dataset9\\images").glob("**/2022*"))
        data_path_list = list(root)
        print(data_path_list)
        models = {
            # "baseline": fs.workdir_vision.joinpath("yolov6", "best2.pt"),
            # "blue_annotation_rack": fs.workdir_vision.joinpath("yolov6", "weights", "blue_cap_rack.pt"),
            # "blue_annotation_rack_noracklabel": fs.workdir_vision.joinpath("yolov6", "weights", "blue_cap_norack.pt"),
            # "blue_annotation": fs.workdir_vision.joinpath("yolov6", "weights", "blue_cap.pt"),
            # "blue_annotation_copy_paste_aug": fs.workdir_vision.joinpath("yolov6", "weights",
            #                                                              "blue_cap_copy_paste_aug.pt"),
            # "blue_annotation_copy_paste_aug_occlusion": fs.workdir_vision.joinpath("yolov6", "weights",
            #                                                                        "blue_cap_cp_aug_occlusion.pt"),
            # "color_blue_annotation_synthetic":fs.workdir_vision.joinpath("yolov6", "weights",
            #                                                                        "color_blue_cap_synthetic.pt"),
            # "blue_annotation_synthetic_data": fs.workdir_vision.joinpath("yolov6", "weights",
            #                                                              "blue_cap_synthetic_data.pt"),

            # "20220608_sd_r_iter_100": fs.workdir_vision.joinpath("yolov6", "weights",
            #                                                   "20220608",
            #                                                   "sd_100.pt"),
            # "20220608_ssd_r_iter_100": fs.workdir_vision.joinpath("yolov6", "weights",
            #                                                   "20220608",
            #                                                   "ssd_r_100.pt"),
            # "20220608_sd_r_nbg_iter_100": fs.workdir_vision.joinpath("yolov6", "weights",
            #                                                       "20220608",
            #                                                       "sd_r_nbg_100.pt"),
            # "20220608_sd_r_bg_iter_100": fs.workdir_vision.joinpath("yolov6", "weights",
            #                                                          "20220608",
            #                                                          "sd_r_bg100.pt"),
            # "20220608_sd_r_baseline_iter_100": fs.workdir_vision.joinpath("yolov6", "weights",
            #                                                               "20220609",
            #                                                               "sd_r_baseline_100.pt"),
            "20220608_sd_r_rand_occ_iter_100": fs.workdir_vision.joinpath("yolov6", "weights",
                                                                          "20220609",
                                                                          "sd_r_rand_occ_100.pt"),
            # "20220610_sd_r_no_bg_iter_100": fs.workdir_vision.joinpath("yolov6", "weights",
            #                                                               "20220610",
            #                                                               "sd_r_no_bg.pt"),
            # "20220616_sd_r_fixed_bg_iter_100": fs.workdir_vision.joinpath("yolov6", "weights",
            #                                                            "20220616",
            #                                                            "sd_r_fixed_bg.pt"),
            # "20220616_sd_r_white_bg_iter_100": fs.workdir_vision.joinpath("yolov6", "weights",
            #                                                               "20220616",
            #                                                               "sd_r_white_bg.pt"),

            # "20220616_sd_r_color_bg_iter_100": fs.workdir_vision.joinpath("yolov6", "weights",
            #                                                               "20220616",
            #                                                               "sd_r_color.pt"),

            # "20220624_blue_cap_human_grasp": fs.workdir_vision.joinpath("yolov6", "weights",
            #                                                               "20220624",
            #                                                               "blue_cap_human_grasp.pt"),
            # "blue_annotation_synthetic_data_negative": fs.workdir_vision.joinpath("yolov6", "weights",
            #                                                              "blue_cap_cp_aug_negative_enhance.pt"),
            # "blue_annotation_synthetic_data_again_occlusion_include_real_obj": fs.workdir_vision.joinpath("yolov6", "weights",
            #                                                                       "blue_annotation_synthetic_data_against_occlusion_real_obj_include.pt"),
            # "blue_annotation_color": fs.workdir_vision.joinpath("yolov6", "weights", "blue_cap_color.pt"),
            # "blue_inhnd_annotation_color": fs.workdir_vision.joinpath("yolov6", "weights", "blue_cap_inhand_color.pt"),
            # "white_inhnd_annotation_79_epoch": fs.workdir_vision.joinpath("yolov6", "weights", "white_cap_79_epoch.pt"),
            # "yolov5.pt": fs.Path("D:\chen\yolo_huri\yolov5x.pt")
            # "blue_cap_depth": fs.workdir_vision.joinpath("yolov6", "weights", "blue_cap_depth.pt"),
        }
        res_analy = model_comp(data_path_list=data_path_list, model_paths=models, analysis_tube=["blue", ],
                               img_type="gray",
                               show=True, )
        fs.dump_pickle(res_analy, "res_vaild.pkl", reminder=False)
    else:
        bg_img = np.ones((1544, 2064, 3)) * 255
        # bg_img = cv2.imread("20220421-211641.jpg")

        root = list(fs.Path("D:\\chen\\yolo_huri\\paper\\dataset10\\labels").glob("**/2022*"))
        # bg_img = plot_annot_distr(root, bg_img, color=(0, 255, 0), show=False)

        res_analy = fs.load_pickle("res_vaild.pkl")
        bg_img = plot_analy_res(res_analy, bgnd_img=bg_img, color=[(0, 0, 255), (255, 0, 0)], show=True)

        from huri.components.utils.img_utils import draw_grid

        bg_img = draw_grid(bg_img, line_space=32)

        cv2.imwrite("result.jpg", bg_img)
