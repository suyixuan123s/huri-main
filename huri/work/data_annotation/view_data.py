from huri.components.data_annotaion._constants import *  # import constants and logging conf
from huri.components.gui.tk_gui.base import GuiFrame
from huri.components.utils.panda3d_utils import ImgOnscreen, ExtraWindow
import huri.core.file_sys as fs
import scipy as sp
from huri.core.constants import ANNOTATION_0_0_2
import huri.vision.pnt_utils as pntu
from huri.components.vision.tube_detector import extract
from huri.components.data_annotaion.utils import highlight_mask
import vision.depth_camera.util_functions as dcuf
from huri.vision.pnt_utils import cluster_pcd

ANN_FORMAT = ANNOTATION_0_0_2.IN_HAND_ANNOTATION_SAVE_FORMAT


def flood_fill(test_array, h_max=255):
    if len(test_array) > 2 and test_array.shape[2] > 1:
        test_array = cv2.cvtColor(test_array, cv2.COLOR_BGR2GRAY)
    else:
        test_array = test_array.copy().reshape(test_array.shape[:2])
    input_array = np.copy(test_array)
    el = sp.ndimage.generate_binary_structure(2, 2).astype(np.int)
    inside_mask = sp.ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask] = h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)
    el = sp.ndimage.generate_binary_structure(2, 1).astype(np.int)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(input_array, sp.ndimage.grey_erosion(output_array, size=(3, 3), footprint=el))
    return output_array


def map_pcd_img2d(extracted_pcd_idx, img_sz):
    h, w = img_sz[0], img_sz[1]
    idx_in_pixel = np.unravel_index(extracted_pcd_idx, (h, w))
    h1, h2 = idx_in_pixel[0].min(), idx_in_pixel[0].max()
    w1, w2 = idx_in_pixel[1].min(), idx_in_pixel[1].max()
    return h1, w1, h2, w2


def extract_pixel(pcd,
                  img_sz,
                  extract_area=((-.03, .03), (.01, .05), (-.03, .03)), ):
    # Transform and Plot Point Clouds
    extracted_pcd_idx = np.where((pcd[:, 0] > extract_area[0][0]) & (pcd[:, 0] < extract_area[0][1])
                                 & (pcd[:, 1] > extract_area[1][0]) & (pcd[:, 1] < extract_area[1][1])
                                 & (pcd[:, 2] > extract_area[2][0]) & (pcd[:, 2] < extract_area[2][1]))[
        0]
    if len(extracted_pcd_idx) < 1:
        return None
    h1, w1, h2, w2 = map_pcd_img2d(extracted_pcd_idx, img_sz)
    ploygon_mask = phoxi_map_pcd_2_poly_mask(extracted_pcd_idx, img_sz, conv_area=True)
    return (h1, w1, h2, w2), ploygon_mask, extracted_pcd_idx


class ViewAnnotationData():
    @staticmethod
    def _read_collected_data(path: fs.Path) -> ANN_FORMAT:
        data = fs.load_pickle(path)
        return ANN_FORMAT(*data)

    def __init__(self,
                 showbase,
                 rbt_s=None,
                 data_path=DATA_ANNOT_PATH,
                 pcd_affine_matrix=None, ):
        # load data
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

        # sim robot
        self._rbt_s = rbt_s
        self._rbt_s_draw = None

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
        self._pcd_draw = gm.gen_pointcloud(pcd, rgbas=color, pntsize=5)
        self._pcd_draw.attach_to(self._showbase)

    def _render_sim_rbt(self, jnt_val):
        if self._rbt_s is None:
            return
        self._rbt_s.fk(jnt_val)
        if self._rbt_s_draw is not None:
            self._rbt_s_draw.remove()
        self._rbt_s_draw = self._rbt_s.gen_meshmodel()
        self._rbt_s_draw.attach_to(self._showbase)

    def _render_img(self, img):
        if img is None:
            print("Input img is None")
            return
        self._img_tx.update_img(img)

    def extract_pcd_by_height(self, pcd_region, height_range=(.045, 0.072), toggle_debug=False):
        pcd_ind = pntu.extract_pcd_by_range(pcd=pcd_region[:, :3],
                                            z_range=height_range,
                                            toggle_debug=toggle_debug)
        return pcd_ind

    def _render(self, file_ind):
        file_ind = min(max(file_ind, 0), len(self.data_file_paths) - 1)
        if file_ind in self._cached_file:
            collected_data = self._cached_file[file_ind]
        else:
            collected_data = self._read_collected_data(self.data_file_paths[file_ind])
            self._cached_file[file_ind] = collected_data
        self._file_ind = file_ind

        img_labeled = collected_data.color_img.copy()

        # align pcd
        trans = np.dot(rm.homomat_from_posrot(pos=collected_data.rbt_tcp_pos, rot=collected_data.rbt_tcp_rot),
                       self._pcd_affine_mat)
        pcd_aligned = rm.homomat_transform_points(trans, points=collected_data.pcd)

        # render pcd
        color_c4 = np.ones((len(collected_data.pcd_color), 4), dtype=float)
        color_c4[:, :3] = collected_data.pcd_color

        rack_pcd_id = self.extract_pcd_by_height(pcd_region=pcd_aligned, height_range=(.045, 0.072), toggle_debug=False)
        r_pcd = pcd_aligned[rack_pcd_id]
        r_color_c4 = color_c4[rack_pcd_id]
        r, iid = dcuf.remove_outlier(r_pcd, downsampling_voxelsize=None, nb_points=50, radius=.0015)
        r_pcd, r_color_c4 = r_pcd[iid], r_color_c4[iid]
        # cluster_label = cluster_pcd(r_pcd, is_remove_outlier=False)
        # cluster_label[cluster_label < 0] = 0
        #
        # f_id = np.argmax(np.bincount(cluster_label))
        # c_id = f_id == cluster_label
        # r_pcd, r_color_c4 = r_pcd[c_id], r_color_c4[c_id]
        #

        self._render_pcd(r_pcd, r_color_c4)

        # tube_pcd_id = self.extract_pcd_by_height(pcd_region=pcd_aligned,
        #                                          height_range=(.073, .13),
        #                                          toggle_debug=True)
        # r2_pcd = pcd_aligned[tube_pcd_id]
        # r2_color_c4 = color_c4[tube_pcd_id]
        #
        # cluster_label = cluster_pcd(r2_pcd,
        #                             nb_distance=.01,
        #                             min_points=80,
        #                             is_remove_outlier=False,
        #                             nb_points=50,
        #                             radius=.0015)
        # print(f"There are {cluster_label.max() + 1} tubes")
        # for i in range(cluster_label.max() + 1):
        #     random_color = rm.random_rgba(False)
        #     random_color[:3] = random_color[:3] * 255
        #     cluster_index = cluster_label == i
        #     croped_pcd = r2_pcd[cluster_index]
        #     extracted_pcd_idx = tube_pcd_id[cluster_index]
        #     _h1, _w1, _h2, _w2 = map_pcd_img2d(extracted_pcd_idx, collected_data.color_img.shape)
        #
        #     if (_w2 - _w1) * (_h2 - _h1) < 1000:
        #         continue
        #
        #     gm.gen_pointcloud(croped_pcd,
        #                       rgbas=[[1, 0, 0, 1]]).attach_to(self._pcd_draw)
        #     _h_e_1, _w_e_1, _h_e_2, _w_e_2 = _h1, _w1, _h2, _w2
        #     img_labeled = cv2.rectangle(img_labeled, (_w_e_1, _h_e_1), (_w_e_2, _h_e_2),
        #                                 color=random_color[:3][::-1], thickness=3)
        # self._render_pcd(r2_pcd, r2_color_c4)

        #
        c_img = collected_data.color_img
        idx_in_pixel = np.unravel_index(rack_pcd_id, c_img.shape[:2])
        vertices_candidate = np.vstack((idx_in_pixel[1], idx_in_pixel[0])).T
        mask = cv2.fillPoly(np.zeros(c_img.shape[:2]), pts=[vertices_candidate], color=(1))
        idx_in_pixel = np.where(mask)
        h1, h2 = idx_in_pixel[0].min(), idx_in_pixel[0].max()
        w1, w2 = idx_in_pixel[1].min(), idx_in_pixel[1].max()
        out = cv2.rectangle(img_labeled, (w1, h1), (w2, h2),
                            color=(0, 255, 0), thickness=2)

        # render img
        self._render_img(out)

        # render sim robot
        self._render_sim_rbt(collected_data.rbt_joints)

        # img_labeled = collected_im.copy()
        #
        # for label in collected_data["annotaions"]:
        #     label_name = label.label_name
        #     img_bbox = label.bbox_img
        #     extract_pcd_idx = label.extracted_pcd_idx
        #     img_labeled = cv2.rectangle(img_labeled, (img_bbox[0], img_bbox[1]), (img_bbox[2], img_bbox[3]),
        #                                 color=(0, 255, 0), thickness=3)
        #     if self._img_type == "gray":
        #         mask = np.zeros_like(collected_im).reshape(-1, 3)
        #         mask[extract_pcd_idx] = 1
        #         mask = mask.reshape(collected_im.shape)
        #         mask = mask[img_bbox[1]:img_bbox[3] + 1, img_bbox[0]:img_bbox[2] + 1, :]
        #         mask = flood_fill(mask, h_max=1)
        #         ssssm = collected_im[img_bbox[1]:img_bbox[3] + 1, img_bbox[0]:img_bbox[2] + 1, :]
        #         ssssm2 = cv2.cvtColor(ssssm, cv2.COLOR_BGR2GRAY)
        #         ssssm3 = ssssm2 * mask
        #         cv2.imwrite("mask.jpg", mask * 255)
        #         cv2.imwrite("img.jpg", collected_im[img_bbox[1]:img_bbox[3] + 1, img_bbox[0]:img_bbox[2] + 1, :])
        #         cv2.imwrite("imaaag.jpg", ssssm3)
        #         # cv2.waitKey(0)
        #     if self._pcd_draw is not None:
        #         pcd_trimesh = tm.Trimesh(vertices=dcuf.remove_outlier(src_nparray=pcd_aligned[extract_pcd_idx].copy(),
        #                                                               downsampling_voxelsize=0.002,
        #                                                               radius=0.006))
        #         obb_gm = gm.GeometricModel(initor=pcd_trimesh.bounding_box_oriented)
        #         obb_gm.set_rgba([0, 1, 0, .3])
        #         obb_gm.attach_to(self._pcd_draw)
        #
        # self._render_img(img_labeled)


if __name__ == "__main__":
    from huri.core.common_import import *
    from robot_sim.manipulators.xarm_lite6 import XArmLite6

    # 3D environment
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    # init the simulation robot
    rbt_s = XArmLite6(enable_cc=True)

    # affine matrix
    eye_to_hand_pos = np.array([0.065, 0, 0.028])
    eye_to_hand_rot = rm.rotmat_from_axangle(np.array([0, 0, 1]), np.pi / 2)
    eye_to_hand_mat = rm.homomat_from_posrot(eye_to_hand_pos, eye_to_hand_rot)

    al = ViewAnnotationData(showbase=base,
                            rbt_s=rbt_s,
                            pcd_affine_matrix=eye_to_hand_mat,
                            data_path=fs.Path("C:\\Users\\WRS\\Desktop\\r"), )

    base.startTk()
    base.tkRoot.withdraw()
    base.run()
