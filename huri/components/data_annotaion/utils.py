from collections import namedtuple
from typing import Union, Literal

import cv2
from tqdm import tqdm

from basis.data_adapter import gen_colorarray

import huri.core.utils as hcu
from huri.core.common_import import fs, gm, rm
from huri.components.gui.tk_gui.base import GuiFrame
from huri.components.utils.panda3d_utils import ExtraWindow, ImgOnscreen, img_to_n_channel
from huri.vision.phoxi_capture import vision_pipeline, SensorMarkerHandler, vision_read_data
from huri.examples.data_collection.phoxi_calib.utils import read_calibration_info
from huri.core.constants import ANNOTATION_0_0_2
import open3d as o3d
import numpy as np
import scipy

Label = namedtuple("Label", ["label_name",
                             "bbox_depth",
                             "bbox_img",
                             "extracted_pcd_idx", ])


def phoxi_map_pcd_2_poly_mask(extracted_pcd_idx, img_sz, conv_area=True):
    h, w = img_sz[0], img_sz[1]
    idx_in_pixel = np.unravel_index(extracted_pcd_idx, (h, w))
    vertices_candidate = np.vstack((idx_in_pixel[1], idx_in_pixel[0])).T
    if conv_area:
        vertices = vertices_candidate[scipy.spatial.ConvexHull(vertices_candidate).vertices]
        return cv2.fillPoly(np.zeros(img_sz[:2]), pts=[vertices], color=(1))
    else:
        return cv2.fillPoly(np.zeros(img_sz[:2]), pts=[vertices_candidate], color=(1))


def phoxi_map_pcd_2_img_pixels(extracted_pcd_idx, img_sz):
    h, w = img_sz[0], img_sz[1]
    idx_in_pixel = np.unravel_index(extracted_pcd_idx, (h, w))
    vertices_candidate = np.vstack((idx_in_pixel[1], idx_in_pixel[0])).T
    return vertices_candidate


def mask_2_bbox_xyxy(mask):
    idx_in_pixel = np.where(mask)
    h1, h2 = idx_in_pixel[0].min(), idx_in_pixel[0].max()
    w1, w2 = idx_in_pixel[1].min(), idx_in_pixel[1].max()
    return w1, h1, w2, h2


def highlight_mask(img, mask, color=(0, 255, 0), show_bbox=False, show_contour=False):
    # color to fill
    color = np.array(color, dtype='uint8')

    # equal color where mask, else image
    # this would paint your object silhouette entirely with `color`
    masked_img = np.where(mask[..., None], color, img)

    # use `addWeighted` to blend the two images
    # the object will be tinted toward `color`
    out = cv2.addWeighted(img, 0.8, masked_img, 0.2, 0)

    if show_bbox:
        idx_in_pixel = np.where(mask)
        h1, h2 = idx_in_pixel[0].min(), idx_in_pixel[0].max()
        w1, w2 = idx_in_pixel[1].min(), idx_in_pixel[1].max()
        out = cv2.rectangle(out, (w1, h1), (w2, h2),
                            color=color.tolist(), thickness=2)
    if show_contour:
        contours = cv2.findContours(cv2.inRange(mask, 1, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        big_contour = max(contours, key=cv2.contourArea)
        # draw filled contour on black background
        cv2.drawContours(out, [big_contour], 0, color.tolist(), 3, )
    # merge the channels back together
    return out


class SelectRegionBase:
    """
    Select a region in the scene
    """

    def __init__(self, showbase, param_path: Union[str, fs.Path] = "", ):
        # showbase
        self._showbase = showbase

        # some parameters
        self._param_path = fs.Path(param_path)
        if self._param_path.exists():
            self._params = fs.load_json(param_path)  # load previous select region as a parameters
        else:
            self._create_params()
        self._color_list = gen_colorarray(ncolors=100, nonrandcolor=False)  # generate 100 different random color

        # init gui
        self._panel = GuiFrame(showbase.tkRoot, hidden=False)

    def _create_params(self):
        raise NotImplementedError

    def _init_store_sensor_data(self):
        raise NotImplementedError

    def _init_gui(self):
        raise NotImplementedError

    def _read_offline_sensor_data(self):
        raise NotImplementedError


class SelectRegionBasePhoxiStreamer(SelectRegionBase):
    def __init__(self, showbase,
                 param_path,
                 vision_server_ip,
                 pcd_affine_matrix=None,
                 img_type="color",
                 toggle_debug=False,
                 debug_path=None,
                 is_depth_wd=False):
        assert img_type in ["gray", "color"]
        self._img_type = img_type

        super(SelectRegionBasePhoxiStreamer, self).__init__(showbase=showbase, param_path=param_path)
        self._vision_server_ip = vision_server_ip

        # initialize attributes to store sensor data
        self._init_store_sensor_data()
        # initialize attributes to store pcd transformation if calibration is done
        self._init_calibration_info()

        # load calibration data of external camera
        if img_type == "color":
            self._cam_mat, self._dist_coef, self._homo_mat, self._cam_res = read_calibration_info(
                "D:\chen\phoxi_server_tst\calib_external_cam_custom\calibration.txt")

        # boost base and nodepath of pcd
        # Open two new window
        img_wd = ExtraWindow(base=showbase)
        self._img_tx = ImgOnscreen((1920, 1080), parent_np=img_wd)
        if is_depth_wd:
            depth_img_wd = ExtraWindow(base=showbase)
            self._depth_img_tx = ImgOnscreen((1920, 1080), parent_np=depth_img_wd)

        # attribute to store the drawing of pcd in the simulator
        self._np_pcd = None

        # load pcd affine matrix
        self._pcd_affine_mat = pcd_affine_matrix

        if toggle_debug:
            if debug_path is None:
                raise Exception("Define the toggle path first")
            self._pcd, self._texture, self._depth_img, self._rgb_texture, self._extcam_img = fs.load_pickle(debug_path)
            self._assign_img()
            if self._pcd_affine_mat is not None:
                self._pcd_aligned = rm.homomat_transform_points(self._pcd_affine_mat, points=self._pcd)
                self._render_pcd(self._pcd_aligned)

    def _create_params(self):
        empty_params = {
            "last_selection_val_x": [0, 0, 0, 0],
            "last_selection_val_y": [0, 0, 0, 0],
            "last_selection_val_z": [0, 0, 0, 0],
        }
        fs.dump_json(self._param_path, empty_params)

    def _init_store_sensor_data(self):
        # raw sensor data
        self._streamer = SensorMarkerHandler(self._vision_server_ip)
        self._pcd = None
        self._np_pcd = None
        self._texture = None
        self._depth_img = None
        self._rgb_texture = None
        self._extcam_img = None

        self._img = None

    def _render_pcd(self, pcd, color=((0, 0, 0, .7),)):
        if pcd is None:
            print("Input pcd is None")
            return
        if self._np_pcd is not None:
            self._np_pcd.remove()
        self._np_pcd = gm.gen_pointcloud(pcd, rgbas=color, pntsize=3)
        self._np_pcd.attach_to(self._showbase)

    def _render_img(self, img):
        if img is None:
            print("Input img is None")
            return
        self._img_tx.update_img(img)

    def _render_depth_img(self, img):
        if img is None:
            print("Input img is None")
            return
        self._depth_img_tx.update_img(img)

    def _init_calibration_info(self):
        self._pcd_affine_mat = None
        self._pcd_aligned = None

    def _assign_img(self):
        if self._img_type == "color":
            self._img = self._extcam_img
        elif self._img_type == "gray":
            self._img = self._texture


def save_exp_data(file_dir: str,
                  save_data_path: str = None,
                  data_type: Literal["gray", "color", "pcd"] = "gray", ):
    file_dir = fs.Path(file_dir)
    if save_data_path is not None:
        save_data_path = fs.Path(save_data_path)
    else:
        return False

    is_img_data = False

    sampled_file_paths = list(file_dir.glob("**/*"))
    for idx, file_path in enumerate(tqdm(sampled_file_paths, desc='Scanning images', total=len(sampled_file_paths))):
        print(f"The {idx + 1} sample: {file_path.name.lower()}")
        if file_path.name.lower().endswith(('pkl', 'pickle')):
            pcd, img, depth_img, rgb_texture, extcam_img = vision_read_data(file_path)
            if data_type == "gray":
                saved_data = cv2.equalizeHist(img)
                is_img_data = True
            elif data_type == "color":
                saved_data = extcam_img
                is_img_data = True
            elif data_type == "pcd":
                saved_data = pcd
            else:
                raise Exception(f"Unsupported generated image type: {data_type}")
        else:
            print("Unsupport format")
            continue
        data_name = file_path.name.split(".")[0]

        if save_data_path is not None:
            if is_img_data:
                cv2.imwrite(str(save_data_path / f"{data_name}.jpg"), saved_data)
            else:
                fs.dump_ply(saved_data, path=str(save_data_path / f"{data_name}.ply"), reminder=False)


def output_data_collection(file_dir: str,
                           save_data_path: str = None,
                           data_type: Literal["gray", "color", "pcd"] = "gray", ):
    file_dir = fs.Path(file_dir)
    if save_data_path is not None:
        save_data_path = fs.Path(save_data_path)
    else:
        return False

    is_img_data = False

    sampled_file_paths = list(file_dir.glob("**/*"))
    for idx, file_path in enumerate(tqdm(sampled_file_paths, desc='Scanning images', total=len(sampled_file_paths))):
        print(f"The {idx + 1} sample: {file_path.name.lower()}")
        if file_path.name.lower().endswith(('pkl', 'pickle')):
            data = ANNOTATION_0_0_2.IN_HAND_ANNOTATION_SAVE_FORMAT(*fs.load_pickle(file_path))
            if data_type == "color":
                saved_data = data.color_img
                is_img_data = True
            elif data_type == "pcd":
                saved_data = data.pcd
            else:
                raise Exception(f"Unsupported generated image type: {data_type}")
        else:
            print("Unsupport format")
            continue
        data_name = file_path.name.split(".")[0]

        if save_data_path is not None:
            if is_img_data:
                cv2.imwrite(str(save_data_path / f"{data_name}.jpg"), saved_data)
            else:
                fs.dump_ply(saved_data, path=str(save_data_path / f"{data_name}.ply"), reminder=False)


if __name__ == "__main__":
    # save_exp_data(
    #     file_dir=fs.workdir_data.joinpath("data_annotation", "tab_color_valid", "blue_white_valid_data"),
    #     save_data_path="D:\chen\yolo_huri\paper\\tase_paper\\robot_bg",
    #     data_type="color",
    # )

    # save_exp_data(
    #     file_dir=fs.workdir_data.joinpath("data_annotation", "human_grasp", "blue_tube"),
    #     save_data_path="D:\chen\yolo_huri\paper\dataset15\images",
    #     data_type="gray",
    # )

    output_data_collection(
        file_dir="C:\\Users\\WRS\\Desktop\\img20220927",
        save_data_path="C:\\Users\\WRS\\Desktop\\data20220927",
        data_type="color",
    )
