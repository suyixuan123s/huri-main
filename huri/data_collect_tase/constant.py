import logging
from collections import namedtuple
from typing import Literal

import huri.core.file_sys as fs

# logging configuration
# LOGGER_LEVEL = logging.DEBUG
# logging.basicConfig(level=LOGGER_LEVEL)

# Setup
# TUBE_NAME: str = 'WHITE_CAP'
# TUBE_ID: int = 3
# WORK_ARM: Literal['rgt_arm', 'lft_arm', 'both'] = 'both'
# SAVE_PATH: str = r'E:/huri_shared/huri/data/data_annotation/EXP/WHITE_CAP'
# IMG_TYPE: Literal['gray', 'color'] = 'gray'  # Use `gray` only if an external camera exists
# TOGGLE_DEBUG = False

# Setup
# TUBE_NAME: str = 'RED_CAP'
# TUBE_ID: int = 3
# WORK_ARM: Literal['rgt_arm', 'lft_arm', 'both'] = 'both'
# SAVE_PATH: str = r'E:/huri_shared/huri/data/data_annotation/EXP/RED_CAP'
# IMG_TYPE: Literal['gray', 'color'] = 'gray'  # Use `gray` only if an external camera exists
# TOGGLE_DEBUG = False

# TUBE_NAME: str = 'WHITE_TUBE_3'
# TUBE_ID: int = 4
# WORK_ARM: Literal['rgt_arm', 'lft_arm', 'both'] = 'both'
# SAVE_PATH: str = r'E:/huri_shared/huri/data/data_annotation/EXP/WHITE_TUBE_3'
# IMG_TYPE: Literal['gray', 'color'] = 'gray'  # Use `gray` only if an external camera exists
# TOGGLE_DEBUG = False

# setup
TUBE_NAME: str = 'CUSTOM_CAP'
TUBE_ID: int = 5
WORK_ARM: Literal['rgt_arm', 'lft_arm', 'both'] = 'both'
SAVE_PATH: str = r'E:/huri_shared/huri/data/data_annotation/EXP/CUSTOM_CAP'
IMG_TYPE: Literal['gray', 'color'] = 'gray'  # Use `gray` only if an external camera exists
TOGGLE_DEBUG = False


# Constants
DATA_ANNOT_PATH = fs.workdir_data.joinpath("data_annotation")
DATA_ANNOT_ONRACK_PATH = DATA_ANNOT_PATH.joinpath("tab_color_valid")
SEL_PARAM_PATH = fs.Path(__file__).parents[0].joinpath("params")


class ANNOTATION:
    VERSION = "0.0.1"

    class WORK_ARM_ID:
        RIGHT = 0
        LEFT = 1

    LABEL = namedtuple("LABEL", ["label_name",
                                 "version",
                                 "img_type",
                                 "bboxes",
                                 "polygons",
                                 "extracted_pcd_idx", ])

    BBOX_XYXY = namedtuple("BBOX_XYXY", ["w1", "w2", "h1", "h2"])
    IN_HAND_ANNOTATION_SAVE_FORMAT = namedtuple("IN_HAND_ANNOTATION_SAVE_FORMAT",
                                                ["version",
                                                 "pcd",
                                                 "gray_img",
                                                 "extcam_img",
                                                 "depth_img",
                                                 "rbt_tcp_pos",
                                                 "rbt_tcp_rot",
                                                 "rbt_joints",
                                                 "annotations"])

    ON_TABLE_ANNOTATION_SAVE_FORMAT = namedtuple("ON_TABLE_ANNOTATION_SAVE_FORMAT", ["version",
                                                                                     "pcd",
                                                                                     "gray_img",
                                                                                     "extcam_img",
                                                                                     "depth_img",
                                                                                     "annotations"])


class ANNOTATION_0_0_2:
    VERSION = "0.0.2"

    class WORK_ARM_ID:
        RIGHT = 0
        LEFT = 1

    LABEL = namedtuple("LABEL", ["label_name",
                                 "version",
                                 "img_type",
                                 "bboxes",
                                 "polygons",
                                 "extracted_pcd_idx", ])

    BBOX_XYXY = namedtuple("BBOX_XYXY", ["w1", "w2", "h1", "h2"])
    IN_HAND_ANNOTATION_SAVE_FORMAT = namedtuple("IN_HAND_ANNOTATION_SAVE_FORMAT",
                                                ["version",
                                                 "pcd",
                                                 "pcd_color",
                                                 "gray_img",
                                                 "color_img",
                                                 "depth_img",
                                                 "rbt_tcp_pos",
                                                 "rbt_tcp_rot",
                                                 "rbt_joints",
                                                 "annotations"])

    ON_TABLE_ANNOTATION_SAVE_FORMAT = namedtuple("ON_TABLE_ANNOTATION_SAVE_FORMAT", ["version",
                                                                                     "pcd",
                                                                                     "gray_img",
                                                                                     "extcam_img",
                                                                                     "depth_img",
                                                                                     "annotations"])
