from pathlib import Path
from collections import namedtuple

workdir = Path(__file__).parent.parent  # work directory
workdir_model = workdir.joinpath("models")  # model
workdir_learning = workdir.joinpath("learning")
workdir_data = workdir.joinpath("data")

DATA_ANNOT_PATH = workdir_data.joinpath("data_annotation")


class SENSOR_INFO:
    IP_ADR_DEPTH_SENSOR = "127.0.0.1:18300"
    IP_ADR_DEPTH_SENSOR_DEBUG = "127.0.0.1:18300"
    PNT_CLD_CALIBR_MAT_PATH = workdir_data.joinpath("calibration", "qaqqq3.json")


DATA_IDX = 1
EXP = 'exp1'


# class DATA_ANNOT:
#     SAVE_PATH = workdir_data.joinpath("data_annotation")
# SEL_PARAM_PATH = fs.Path(__file__).parents[0].joinpath("params")

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
