import numpy as np

from huri.core.common_import import *
from _constants import *  # import constants and logging conf
import huri.vision.yolo.detect as yyd
import cv2 as cv
from time import strftime
from pathlib import Path
from tqdm import tqdm
from huri.vision.phoxi_capture import (depth2gray_map, )
from huri.components.utils.panda3d_utils import img_to_n_channel
from huri.components.utils.img_utils import crop_center
from huri.core.constants import ANNOTATION

ANNOTAION_FORMAT = ANNOTATION.IN_HAND_ANNOTATION_SAVE_FORMAT
ANNOTAION_LABEL = ANNOTATION.LABEL
BBOX_LABEL = ANNOTATION.BBOX_XYXY


def to_path(p):
    if isinstance(p, str):
        return Path(p)
    elif isinstance(p, Path):
        return p
    else:
        raise Exception("Unsupport path format")


def check_id(name, label_info):
    for _n in label_info.keys():
        if _n in name:
            return label_info[_n]


def boudingbox2yololabel(size, box: np.ndarray):
    """
    size: the size of the image (w, h)
    box: the bounding box contains four values (x_min, x_max, y_min, y_max)
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def crop_label_xyxy(crop_start_coord, xyxy):
    xyxy = xyxy.copy()
    xyxy[..., [0, 2]] -= crop_start_coord[0]
    xyxy[..., [1, 3]] -= crop_start_coord[1]
    return xyxy


def auto_labeling(file_dir: str,
                  label_info: int,
                  save_img_path: str = None,
                  save_label_path: str = None,
                  is_gray_img=True,
                  img_type="gray",
                  toggle_debug=False):
    file_dir = to_path(file_dir)
    if save_img_path is not None:
        save_img_path = to_path(save_img_path)

    if save_label_path is not None:
        save_label_path = to_path(save_label_path)

    sampled_file_paths = list(file_dir.glob("**/*"))
    for idx, file_path in enumerate(tqdm(sampled_file_paths, desc='Scanning images', total=len(sampled_file_paths))):
        print(f"The {idx + 1} sample: {file_path.name.lower()}")
        if file_path.name.lower().endswith(('pkl', 'pickle')):
            data = fs.load_pickle(file_path)
            annotations = None
            if len(data) == 6:
                pcd, img, depth_img, rbt_tcp_pos, rbt_tcp_rot, box_labels = data[0], data[1], data[2], data[3], \
                    data[4], data[5]
                has_color_img = False
                print("No color image in the data")
            elif len(data) == 5:
                pcd, img, rgb_img, depth_img, box_labels = data[0], data[1], data[2], data[3], \
                    data[4]
            elif len(data) == 9:
                annotations = ANNOTAION_FORMAT(*data)
                if img_type == "gray":
                    saved_img = cv.equalizeHist(annotations.gray_img)
                elif img_type == "depth":
                    saved_img = depth2gray_map(annotations.depth_img)
                elif img_type == "color":
                    saved_img = saved_img, crop_start_coord = crop_center(annotations.extcam_img, 2064, 1544,
                                                                          offsetx=300, )
                else:
                    raise Exception(f"Unsupported generated image type: {img_type}")
                box_labels = [ANNOTAION_LABEL(*_).bboxes for _ in annotations.annotations]
            else:
                pcd, img, rgb_img, depth_img, rbt_tcp_pos, rbt_tcp_rot, box_labels = data[0], data[1], data[2], data[3], \
                    data[4], data[5], data[6]
                has_color_img = True
            if img_type == "gray" and not annotations:
                saved_img = cv.equalizeHist(img)
            elif img_type == "depth":
                saved_img = depth2gray_map(depth_img)
            elif img_type == "color" and has_color_img:
                saved_img, crop_start_coord = crop_center(rgb_img, 2064, 1544, offsetx=300, )
            else:
                if not annotations:
                    raise Exception(f"Unsupported generated image type: {img_type}")
        else:
            print("Unsupport format")
            continue
            # raise Exception("Unsupport format")
        labeled_img = img_to_n_channel(saved_img)
        if box_labels is None:
            continue
        for box_label in box_labels:
            if img_type == "gray":
                if isinstance(box_label, list) or isinstance(box_label,tuple):
                    (h1, w1, h2, w2, idx) = box_label
                else:
                    (w1, h1, w2, h2) = box_label.bbox_img
            else:
                (w1, h1, w2, h2) = box_label.bbox_img
            labeled_img = cv2.rectangle(labeled_img, (w1, h1), (w2, h2),
                                        color=(255, 0, 0), thickness=1)
        # img_name = strftime('%Y%m%d-%H%M%S')
        img_name = file_path.name.split(".")[0]
        img_shape = saved_img.shape
        x_px, y_px = img_shape[1], img_shape[0]
        if save_label_path is not None:
            if save_img_path is not None:
                label_name = img_name
            else:
                label_name = file_path.name.split(".")[0]
            with open(str(save_label_path / f"{label_name}.txt"), "w") as f:
                for box_label in box_labels:
                    if img_type == "gray":
                        if isinstance(box_label, list):
                            (h1, w1, h2, w2, idx) = box_label
                        else:
                            (w1, h1, w2, h2) = box_label.bbox_img
                            label_name = box_label.label_name
                    else:
                        (w1, h1, w2, h2) = box_label.bbox_img
                        w1, h1, w2, h2 = crop_label_xyxy(crop_start_coord=crop_start_coord,
                                                         xyxy=np.array([w1, h1, w2, h2]))

                    _x, _y, _w, _h = boudingbox2yololabel(size=(x_px, y_px), box=np.array([w1, w2, h1, h2]))
                    f.writelines(f"{label_info[label_name]} {_x} {_y} {_w} {_h}\n")
        if save_img_path is not None:
            cv.imwrite(str(save_img_path / f"{img_name}.jpg"), saved_img)

        if toggle_debug:
            cv2.imshow(f"YOLO detection result.", labeled_img)
            cv2.waitKey(0)


if __name__ == "__main__":
    # setup the file sampled
    # file_dir = fs.workdir / "data" / "vision_exp" / "exp_20211222193921"
    # file_dir = DATA_ANNOT_PATH.joinpath("EXP", "WHITE_CAP")

    TUBE_TYPE = 'RED_CAP'
    DATA_PATH = fs.Path(r'H:\data')
    file_dir = DATA_PATH.joinpath(TUBE_TYPE)
    # file_dir = fs.workdir_data / "tase_paper"

    # "D:\chen\yolo_huri\imgs\imgs2"
    # save_img_path = fs.workdir / "data" / "training_data" / "train_imgs"
    # save_label_path = fs.workdir / "data" / "training_data" / "labels"
    # save_img_path = "D:\chen\yolo_huri\paper\\tase_paper"
    save_img_path = "H:\data\saved_data2\images"
    save_label_path = "H:\data\saved_data2\labels"
    label_info = {
        'rack': 0, 'WHITE_CAP': 1, 'WHITE_TUBE_3': 2, 'RED_CAP': 3,
        'CUSTOM_CAP': 4, 'WHITE_CAP_1': 5,
    }

    auto_labeling(file_dir=file_dir,
                  label_info=label_info,
                  save_img_path=save_img_path,
                  save_label_path=save_label_path,
                  img_type="gray",
                  toggle_debug=False)
