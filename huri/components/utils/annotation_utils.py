"""
Utility functions to operate annotations of image
"""
from typing import Tuple, Union, Literal

import cv2
import numpy as np

from huri.core.file_sys import Path


def bboxes_xywh2xyxy(bboxes_xywh: np.ndarray) -> np.ndarray:
    """
    Change format of bounding box from (x,y,w,h) to the (x1, y1, x1, y2)
    :param bboxes_xywh: np array in nx4 dimension. n is number of bounding boxes
    :return: nx4 np.array bounding boxes in (x1, y1, x1, y2) format
    """
    box_xyxy = bboxes_xywh.copy()
    box_xyxy[..., 0] = (bboxes_xywh[..., 0] - bboxes_xywh[..., 2] / 2.)
    box_xyxy[..., 1] = (bboxes_xywh[..., 1] - bboxes_xywh[..., 3] / 2.)
    box_xyxy[..., 2] = (bboxes_xywh[..., 0] + bboxes_xywh[..., 2] / 2.)
    box_xyxy[..., 3] = (bboxes_xywh[..., 1] + bboxes_xywh[..., 3] / 2.)
    return box_xyxy


def bboxes_xyxy2xywh(bboxes_xyxy) -> np.ndarray:
    """
    Change format of bounding box from (x1, y1, x1, y2) to the (x,y,w,h)
    :param bboxes_xyxy: np array in nx4 dimension. n is number of bounding boxes
    :return: nx4 np.array bounding boxes in (x,y,w,h) format
    """
    box_xywh = bboxes_xyxy.copy()
    box_xywh[..., 0] = (bboxes_xyxy[..., 0] + bboxes_xyxy[..., 2]) / 2.
    box_xywh[..., 1] = (bboxes_xyxy[..., 1] + bboxes_xyxy[..., 3]) / 2.
    box_xywh[..., 2] = bboxes_xyxy[..., 2] - bboxes_xyxy[..., 0]
    box_xywh[..., 3] = bboxes_xyxy[..., 3] - bboxes_xyxy[..., 1]
    return box_xywh


def scale_label_format(bboxes_anno: np.ndarray, img_size: np.ndarray,
                       op: Literal["normalize", "scale"] = "normalize", ):
    """
    Scale the label
    :param bboxes_anno: the bbox annotations
    :param img_size:
    :param op: Two operations: `normalize` or `scale`.
            - `normalize` will make labels into normalized label.
            - `scale` will make normalized labels back to origin scale
    :return:
    """
    assert op in ("normalize", "scale")
    h, w = img_size[:2]
    bboxes_anno = bboxes_anno.copy().astype(float)
    op_func = lambda x, s: x * s
    if op == "normalize":
        op_func = lambda x, s: x / s
    bboxes_anno[..., [0, 2]] = op_func(bboxes_anno[..., [0, 2]], w)
    bboxes_anno[..., [1, 3]] = op_func(bboxes_anno[..., [1, 3]], h)
    return bboxes_anno


def read_yolo_annotation_file(file_path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read YOLO annotation file
    :param file_path: path of the YOLO annotation file
    :return: labels and normalized bounding box annotation
    """
    annotation_file_path = Path(file_path)
    if not annotation_file_path.exists():
        return None, None
    annotation_raw_string = annotation_file_path.read_text().rstrip().lstrip()
    labels = []
    bboxes_xywh = []
    for annotation_string in annotation_raw_string.split("\n"):
        if len(annotation_string) < 1:
            continue
        data = annotation_string.rstrip().lstrip().split()
        label = int(data[0])
        bbox_xywh = np.array([float(data[1]), float(data[2]), float(data[3]), float(data[4])])
        labels.append(label)
        bboxes_xywh.append(bbox_xywh)
    return np.array(labels), np.array(bboxes_xywh)


def write_yolo_annotation_file(labels: np.ndarray, bboxes: np.ndarray, save_path: Union[(Path, str)]) -> bool:
    """
    Write YOLO annotation file
    :param labels: 1xn array to represent labels' indices
    :param bboxes: nx4 array to represent the bounding boxes in form of [x y w h] for labels
    :param save_path: save path for the annotation file
    :return: save success or not
    """
    save_path = Path(save_path)
    labels = np.asarray(labels)
    bboxes = np.asarray(bboxes)
    with save_path.open("w") as f:
        if save_path.exists():
            labels = labels.flatten()
            if len(labels) > 0:
                for label_id in range(len(labels)):
                    f.writelines(f"{int(labels[label_id])} {' '.join(bboxes[label_id].astype(str))}\n")
                return True
            else:
                return False
        else:
            return False


if __name__ == "__main__":
    labels, bboxes_xywh = read_yolo_annotation_file(file_path="20210622-220424.txt")
    img = cv2.imread("20210622-220424.jpg")

    bboxes_xyxy = bboxes_xywh2xyxy(bboxes_xywh)
    for (x1, y1, x2, y2) in scale_label_format(bboxes_xyxy, img_size=img.shape[:2], op="scale").astype(int):
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), )
    cv2.imshow("sfsd", img)
    cv2.waitKey(0)
