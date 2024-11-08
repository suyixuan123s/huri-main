import argparse
from typing import List
from time import strftime

import cv2
import numpy as np
import trimesh

import basis.trimesh as tm
import motion.probabilistic.rrt_connect as rrtc

from _constants import *  # import constants and logging conf
from huri.components.data_annotaion.rgb_phoxi_calib.utils import read_calibration_info
from huri.components.gui.tk_gui.base import GuiFrame
from huri.components.utils.panda3d_utils import ImgOnscreen, ExtraWindow
from rgb_phoxi_calib.utils import project_pcd2extcamimg
from utils import SelectRegionBasePhoxiStreamer, img_to_n_channel, Label
from huri.core.file_sys import workdir
from huri.core.constants import SENSOR_INFO
from huri.core._logging import colorstr
from huri.definitions.tube_def import TubeType
from huri.vision.pnt_utils import extc_fgnd_pcd
from huri.vision.phoxi_capture import depth2gray_map, enhance_gray_img
import vision.depth_camera.util_functions as dcuf
import scipy as sp
import scipy.ndimage

logger = logging.getLogger(__file__)



def flood_fill(test_array,h_max=255):
    if len(test_array) > 2 and test_array.shape[2] > 1:
        test_array = cv2.cvtColor(test_array, cv2.COLOR_BGR2GRAY)
    else:
        test_array = test_array.copy().reshape(test_array.shape[:2])
    input_array = np.copy(test_array)
    el = sp.ndimage.generate_binary_structure(2,2).astype(int)
    inside_mask = sp.ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask]=h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)
    el = sp.ndimage.generate_binary_structure(2,1).astype(int)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(input_array,sp.ndimage.grey_erosion(output_array, size=(3,3), footprint=el))
    return output_array


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--color_img', type=bool, default=False, help='0 for gray image, 1 for color image')
    opt = parser.parse_args()
    opt.img_type = ["gray", "color"][opt.color_img]
    logger.info(colorstr(opt))
    return opt


class ViewAnnotationData():
    @staticmethod
    def _read_color_collected_data(path: fs.Path):
        data = fs.load_pickle(path)
        pcd, gray_im, rgb_im, depth_im, rbt_tcp_pos, rbt_tcp_rot, annotaions = data
        return {"pcd": pcd,
                "gray_im": gray_im,
                "rgb_im": rgb_im,
                "depth_im": depth_im,
                "rbt_tcp_pos": rbt_tcp_pos,
                "rbt_tcp_rot": rbt_tcp_rot,
                "annotaions": annotaions}

    @staticmethod
    def _read_gray_collected_data(path: fs.Path):
        data = fs.load_pickle(path)
        pcd, gray_im, depth_im, rbt_tcp_pos, rbt_tcp_rot, annotation_infos, annotaion_id = data

        annotaions = []
        for annotation_info in annotation_infos:
            annotaions.append(Label(label_name=annotaion_id,
                                    bbox_depth=(
                                        annotation_info[1], annotation_info[0], annotation_info[3], annotation_info[2]),
                                    bbox_img=(
                                        annotation_info[1], annotation_info[0], annotation_info[3], annotation_info[2]),
                                    extracted_pcd_idx=annotation_info[4]))

        return {"pcd": pcd,
                "gray_im": gray_im,
                "rgb_im": None,
                "depth_im": depth_im,
                "rbt_tcp_pos": rbt_tcp_pos,
                "rbt_tcp_rot": rbt_tcp_rot,
                "annotaions": annotaions}

    def __init__(self,
                 showbase,
                 data_path=DATA_ANNOT_PATH,
                 pcd_affine_matrix=None,
                 ext_cam_calib_path="",
                 img_type="gray"):

        self.data_file_paths = []
        data_path = fs.Path(data_path)
        if data_path.is_file():
            self.data_file_paths.append(data_path)
        else:
            for p in data_path.glob("*"):
                if p.name.rstrip().endswith(".pkl"):
                    self.data_file_paths.append(p)

        self._cached_file = {}
        self._file_ind = 1

        # variables to save drawining
        img_wd = ExtraWindow(base=showbase)
        self._img_tx = ImgOnscreen((1920, 1080), parent_np=img_wd)
        self._pcd_draw = None
        self._showbase = showbase

        # load pcd affine matrix
        self._pcd_affine_mat = pcd_affine_matrix

        # load calibration data of external camera
        self._img_type = img_type
        if "color" in img_type:
            self._cam_mat, self._dist_coef, self._homo_mat, self._cam_res = read_calibration_info(ext_cam_calib_path)
            self._read_collected_data = self._read_color_collected_data
        elif "gray" in img_type:
            self._read_collected_data = self._read_gray_collected_data
        else:
            raise Exception("Unsupport image type")

        # init gui
        self._panel = GuiFrame(showbase.tkRoot, hidden=False)
        self._panel.add_button(text="Show Previous File", command=lambda: self._render(self._file_ind - 1),
                               pos=(0, 0))
        self._panel.add_button(text="Show Next File", command=lambda: self._render(self._file_ind + 1), pos=(0, 1))
        self._panel.show()

        self._render(1)

    def _render_pcd(self, pcd, color=((0, 0, 0, .7))):
        if pcd is None:
            print("Input pcd is None")
            return
        if self._pcd_draw is not None:
            self._pcd_draw.remove()
        self._pcd_draw = gm.gen_pointcloud(pcd, rgbas=list(color))
        self._pcd_draw.attach_to(self._showbase)

    def _render_img(self, img):
        if img is None:
            print("Input img is None")
            return
        self._img_tx.update_img(img)

    def _render(self, file_ind):
        file_ind = min(max(file_ind, 0), len(self.data_file_paths) - 1)
        if file_ind in self._cached_file:
            collected_data = self._cached_file[file_ind]
        else:
            collected_data = self._read_collected_data(self.data_file_paths[file_ind])
            self._cached_file[file_ind] = collected_data
        self._file_ind = file_ind
        pcd_aligned = rm.homomat_transform_points(self._pcd_affine_mat, points=collected_data["pcd"])
        if self._img_type == "color":
            color_ind, _ = project_pcd2extcamimg(pcd_raw=collected_data["pcd"],
                                                 phoxi2extcam_homo_mat=np.linalg.inv(
                                                     self._homo_mat),
                                                 cam_mat=self._cam_mat,
                                                 extcam_res=self._cam_res,
                                                 dist_coef=self._dist_coef, )
            color = collected_data["rgb_im"][color_ind[:, 1], color_ind[:, 0]]
            color_c4 = np.ones((len(collected_data["pcd"]), 4), dtype=float)
            # color_c4[..., :3] *= 1
            color_c4[_, :3] = color[..., ::-1] / 255
            collected_im = collected_data["rgb_im"]
        else:
            # TODO Finish the code for gray image coloring
            collected_im = enhance_gray_img(collected_data["gray_im"])
            collected_im = cv2.cvtColor(collected_im, cv2.COLOR_GRAY2BGR)
            color_c3 = collected_im.copy().reshape(-1, 3)
            color_c4 = np.ones((len(color_c3), 4), dtype=float)
            color_c4[..., :3] = color_c3 / 255
        self._render_pcd(pcd_aligned, color_c4)
        self._render_img(collected_im)
        img_labeled = collected_im.copy()

        for label in collected_data["annotaions"]:
            label_name = label.label_name
            img_bbox = label.bbox_img
            extract_pcd_idx = label.extracted_pcd_idx
            img_labeled = cv2.rectangle(img_labeled, (img_bbox[0], img_bbox[1]), (img_bbox[2], img_bbox[3]),
                                        color=(0, 255, 0), thickness=3)
            if self._img_type == "gray":
                mask = np.zeros_like(collected_im).reshape(-1, 3)
                mask[extract_pcd_idx] = 1
                mask = mask.reshape(collected_im.shape)
                mask = mask[img_bbox[1]:img_bbox[3] + 1, img_bbox[0]:img_bbox[2] + 1, :]
                mask = flood_fill(mask, h_max=1)
                ssssm = collected_im[img_bbox[1]:img_bbox[3] + 1, img_bbox[0]:img_bbox[2] + 1, :]
                ssssm2 = cv2.cvtColor(ssssm, cv2.COLOR_BGR2GRAY)
                ssssm3 = ssssm2*mask
                cv2.imwrite("mask.jpg", mask*255)
                cv2.imwrite("img.jpg", collected_im[img_bbox[1]:img_bbox[3] + 1, img_bbox[0]:img_bbox[2] + 1, :])
                cv2.imwrite("imaaag.jpg", ssssm3)
                # cv2.waitKey(0)
            if self._pcd_draw is not None:
                pcd_trimesh = tm.Trimesh(vertices=dcuf.remove_outlier(src_nparray=pcd_aligned[extract_pcd_idx].copy(),
                                                                      downsampling_voxelsize=0.002,
                                                                      radius=0.006))
                obb_gm = gm.GeometricModel(initor=pcd_trimesh.bounding_box_oriented)
                obb_gm.set_rgba([0, 1, 0, .3])
                obb_gm.attach_to(self._pcd_draw)

        self._render_img(img_labeled)


if __name__ == "__main__":
    from huri.core.common_import import *
    from huri.components.yumi_control.yumi_con import YumiController

    # par
    opt = parse_opt()
    # 3D environment
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    # init the simulation robot
    yumi_s = ym.Yumi(enable_cc=True)

    # affine matrix
    affine_mat = np.asarray(
        fs.load_json(SENSOR_INFO.PNT_CLD_CALIBR_MAT_PATH)['affine_mat'])

    al = ViewAnnotationData(showbase=base,
                            pcd_affine_matrix=affine_mat,
                            data_path=fs.Path("D:\chen\huri_shared\huri\examples\\vision\paper\data\dataset_bluecap_2"),
                            ext_cam_calib_path="D:\chen\huri_shared\huri\data\data_annotation\inhnd_color_annot\calibration.txt",
                            # img_type=opt.img_type,
                            img_type="gray")

    base.startTk()
    base.tkRoot.withdraw()
    base.run()
