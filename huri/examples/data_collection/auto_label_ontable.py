import numpy as np

from huri.core.common_import import *
import huri.vision.yolo.detect as yyd
import cv2 as cv
from time import strftime
from pathlib import Path
from tqdm import tqdm
# from huri.core.utils import img_to_n_channel
from huri.vision.phoxi_capture import (depth2gray_map, )
from huri.examples.data_collection.on_table_labeling import Label
from huri.components.utils.annotation_utils import scale_label_format, bboxes_xyxy2xywh
from huri.components.utils.img_utils import crop_center


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


def crop_label_xyxy(crop_start_coord, xyxy):
    xyxy = xyxy.copy()
    xyxy[..., [0, 2]] -= crop_start_coord[0]
    xyxy[..., [1, 3]] -= crop_start_coord[1]
    return xyxy


def auto_labeling(file_dir: str,
                  label_info: dict,
                  save_img_path: str = None,
                  save_label_path: str = None,
                  img_type="gray",
                  crop2size=(2064, 1544),
                  cropoffset=(0, 0),
                  annot_type="robot",
                  toggle_debug=False):
    assert annot_type in ['robot', 'rack']
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
            if annot_type == "rack":
                pcd, texture, img, depth_img, box_labels = data[0], data[1], data[2], data[3], data[4]
            else:
                pcd, texture, extcam_img, depth_img, rbt_tcp_pos, rbt_tcp_rot, box_labels = data[0], data[1], data[2], \
                                                                                            data[3], \
                                                                                            data[4], data[5], data[6]
            if img_type == "gray":
                saved_img = cv.equalizeHist(texture)
            elif img_type == "depth":
                saved_img = depth2gray_map(depth_img)
            elif img_type == "color":
                saved_img = extcam_img
            else:
                raise Exception(f"Unsupported generated image type: {img_type}")
        else:
            print("Unsupport format")
            continue
            # raise Exception("Unsupport format")
        # img_name = strftime('%Y%m%d-%H%M%S')

        img_name = file_path.name.split(".")[0]
        cropped_saved_img, crop_start_coord = crop_center(saved_img, cropx=crop2size[0], cropy=crop2size[1],
                                                          offsetx=cropoffset[0], offsety=cropoffset[1])
        img_shape = cropped_saved_img.shape
        x_px, y_px = img_shape[1], img_shape[0]
        if save_label_path is not None:
            if save_img_path is not None:
                anon_name = img_name
            else:
                anon_name = file_path.name.split(".")[0]
            with (save_label_path / f"{anon_name}.txt").open("w") as f:
                if box_labels is None:
                    box_labels = []
                for _label in box_labels:
                    _label_name = _label.label_name
                    _w1, _h1, _w2, _h2 = _label.bbox_img
                    _xxyy_cropped = crop_label_xyxy(crop_start_coord=crop_start_coord,
                                                    xyxy=np.array([_w1, _h1, _w2, _h2]))
                    # change to YOLO annotation format (x,y,w,h)
                    _bbox_xywh = bboxes_xyxy2xywh(bboxes_xyxy=_xxyy_cropped)
                    ## normalize bbox
                    _bbox_xywh_normalized = scale_label_format(_bbox_xywh, img_size=img_shape, op="normalize")
                    f.writelines(f"{label_info[_label_name]} {' '.join(_bbox_xywh_normalized.astype(str))}\n")
        if save_img_path is not None:
            cv.imwrite(str(save_img_path / f"{img_name}.jpg"), cropped_saved_img)

        if toggle_debug:
            labeled_img = cropped_saved_img.copy()
            for _label in box_labels:
                _label_name = _label.label_name
                _w1, _h1, _w2, _h2 = _label.bbox_extcam
                _xxyy_cropped = crop_label_xyxy(crop_start_coord=crop_start_coord,
                                                xyxy=np.array([_w1, _h1, _w2, _h2]))
                labeled_img = cv2.rectangle(labeled_img, (_xxyy_cropped[0], _xxyy_cropped[1]),
                                            (_xxyy_cropped[2], _xxyy_cropped[3]),
                                            color=(255, 0, 0), thickness=1)
            cv2.imshow(f"YOLO detection result.", labeled_img)
            cv2.waitKey(0)


if __name__ == "__main__":
    # setup the file sampled
    # file_dir = fs.workdir / "data" / "vision_exp" / "exp_20211222193921"
    # file_dir = fs.workdir / "examples" / "data_collection" / "data" / "bluecap"
    file_dir = fs.workdir_data / "data_annotation" / "inhnd_color_annot"
    TUBE_TYPE = 'blue cap'
    # "D:\chen\yolo_huri\imgs\imgs2"
    # save_img_path = fs.workdir / "data" / "training_data" / "train_imgs"
    # save_label_path = fs.workdir / "data" / "training_data" / "labels"
    save_img_path = "D:\chen\yolo_huri\\paper\dataset13\images"
    # save_img_path = None
    save_label_path = "D:\chen\yolo_huri\\paper\dataset13\labels"
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
                  cropoffset=(200, 0),
                  img_type="color",
                  annot_type='robot',
                  toggle_debug=False)
