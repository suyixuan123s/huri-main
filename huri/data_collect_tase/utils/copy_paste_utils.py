from typing import Literal
from huri.core.common_import import fs
from huri.data_collect_tase.constant import ANNOTATION

LABEL = ANNOTATION.LABEL
BBOX = ANNOTATION.BBOX_XYXY


def parse_collected_data(data_path, data_type: Literal["in_hand", "on_table"]):
    if data_type == "in_hand":
        ANNOTATION_FORMAT = ANNOTATION.IN_HAND_ANNOTATION_SAVE_FORMAT
    else:
        ANNOTATION_FORMAT = ANNOTATION.ON_TABLE_ANNOTATION_SAVE_FORMAT
    try:
        data = fs.load_pickle(data_path)
    except Exception:
        print(data_path)
        return None, None
    annotations = ANNOTATION_FORMAT(*data)
    img = annotations.gray_img
    masks = []
    for box_label in annotations.annotations:
        box_label = LABEL(*box_label)
        masks.append(box_label.polygons)
    return img, masks


def parse_collected_data_rich(data_path, data_type: Literal["in_hand", "on_table"]):
    if data_type == "in_hand":
        ANNOTATION_FORMAT = ANNOTATION.IN_HAND_ANNOTATION_SAVE_FORMAT
    else:
        ANNOTATION_FORMAT = ANNOTATION.ON_TABLE_ANNOTATION_SAVE_FORMAT
    try:
        data = fs.load_pickle(data_path)
    except Exception:
        print(data_path)
        return None, None
    annotations = ANNOTATION_FORMAT(*data)
    img = annotations.gray_img
    masks = []
    for box_label in annotations.annotations:
        box_label = LABEL(*box_label)
        masks.append(box_label.polygons)
    return img, masks, annotations
