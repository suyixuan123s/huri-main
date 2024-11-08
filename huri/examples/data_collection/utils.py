from collections import namedtuple

from basis.data_adapter import gen_colorarray

import huri.core.utils as hcu
from huri.core.common_import import fs, gm, rm
from huri.components.gui.tk_gui.base import GuiFrame
from huri.components.utils.panda3d_utils import ExtraWindow, ImgOnscreen, img_to_n_channel
from huri.vision.phoxi_capture import vision_pipeline, SensorMarkerHandler
from huri.examples.data_collection.phoxi_calib.utils import read_calibration_info

Label = namedtuple("Label", ["label_name",
                             "bbox_depth",
                             "bbox_img",
                             "extracted_pcd_idx", ])


class SelectRegionBase:
    """
    Select a region in the scene
    """

    def __init__(self, showbase, param_path="", ):
        # showbase
        self._showbase = showbase

        # some parameters
        self._params = fs.load_json(param_path)  # load previous select region as a parameters
        self._color_list = gen_colorarray(ncolors=100, nonrandcolor=False) * 255  # generate 100 different random color

        # init gui
        self._panel = GuiFrame(showbase.tkRoot, hidden=False)

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
                 debug_path=None):
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
        depth_img_wd = ExtraWindow(base=showbase)
        self._img_tx = ImgOnscreen((1920, 1080), parent_np=img_wd)
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

    def _init_store_sensor_data(self):
        # raw sensor data
        self._streamer = SensorMarkerHandler(self._vision_server_ip),
        self._pcd = None
        self._texture = None
        self._depth_img = None
        self._rgb_texture = None
        self._extcam_img = None

        self._img = None

    def _render_pcd(self, pcd):
        if pcd is None:
            print("Input pcd is None")
            return
        if self._np_pcd is not None:
            self._np_pcd.remove()
        self._np_pcd = gm.gen_pointcloud(pcd)
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

    def _acquire_sensor_data(self):
        """
        1. Acquire the raw sensor data: 1)pcd 2)texture 3) depth image
        """
        if self._streamer is None:
            self._pcd, self._texture, self._depth_img, self._rgb_texture, self._extcam_img = None, None, None, None, None
            print("Cannot acquire data")
        else:
            self._pcd, self._texture, self._depth_img, \
            self._rgb_texture, self._extcam_img = vision_pipeline(self._streamer)
            self._assign_img()

    def _assign_img(self):
        if self._img_type == "color":
            self._img = self._extcam_img
        elif self._img_type == "gray":
            self._img = self._texture
