import numpy as np

from huri.core.common_import import *
import huri.vision.yolo.detect as yyd
import cv2 as cv
from time import strftime
from pathlib import Path
from tqdm import tqdm
from huri.components.utils.panda3d_utils import img_to_n_channel
from huri.vision.phoxi_capture import (depth2gray_map, )


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


def auto_labeling(file_dir: str,
                  label_info: dict,
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
            pcd, img, depth_img, box_labels = data[0], data[1], data[2], data[3]
            if img_type == "gray":
                saved_img = cv.equalizeHist(img)
            elif img_type == "depth":
                saved_img = depth2gray_map(depth_img)
            else:
                raise Exception(f"Unsupported generated image type: {img_type}")
        else:
            print("Unsupport format")
            continue
            # raise Exception("Unsupport format")
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
                for (annotation_name, h1, w1, h2, w2, idx) in box_labels:
                    _x, _y, _w, _h = boudingbox2yololabel(size=(x_px, y_px), box=np.array([w1, w2, h1, h2]))
                    f.writelines(f"{label_info[annotation_name]} {_x} {_y} {_w} {_h}\n")
        if save_img_path is not None:
            cv.imwrite(str(save_img_path / f"{img_name}.jpg"), saved_img)

        if toggle_debug:
            labeled_img = img_to_n_channel(saved_img)
            for (label_name, h1, w1, h2, w2, idx) in box_labels:
                labeled_img = cv2.rectangle(labeled_img, (w1, h1), (w2, h2),
                                            color=(255, 0, 0), thickness=1)
            cv2.imshow(f"YOLO detection result.", labeled_img)
            cv2.waitKey(0)


if __name__ == "__main__":
    # setup the file sampled
    # file_dir = fs.workdir / "data" / "vision_exp" / "exp_20211222193921"
    file_dir = fs.workdir / "examples" / "vision" / "paper" / "data" / "dataset_bluecap_withrack"
    TUBE_TYPE = 'blue cap'
    # "D:\chen\yolo_huri\imgs\imgs2"
    # save_img_path = fs.workdir / "data" / "training_data" / "train_imgs"
    # save_label_path = fs.workdir / "data" / "training_data" / "labels"
    save_img_path = "D:\chen\yolo_huri\\paper\dataset10\images"
    # save_img_path = None
    save_label_path = "D:\chen\yolo_huri\\paper\dataset10\labels"
    # save_label_path = None
    # label_info = {
    #     "blue cap": 0,
    #     "rack": 1,
    #     "purple cap": 2,
    #     "white cap": 3,
    #     "red cap": 4,
    #     "purple ring": 5,
    #     "white cap small": 6,
    # }

    label_info = {
        'rack': 0, 'blue cap': 1, 'purple cap': 2, 'white cap': 3, 'purple ring': 4, 'white cap small': 5
    }

    auto_labeling(file_dir=file_dir,
                  label_info=label_info,
                  save_img_path=save_img_path,
                  save_label_path=save_label_path,
                  img_type="gray",
                  toggle_debug=False)
