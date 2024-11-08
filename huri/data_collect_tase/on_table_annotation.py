from time import strftime

from constant import *
from huri.core.file_sys import workdir
from huri.core.common_import import *
from huri.core.constants import SENSOR_INFO, ANNOTATION
from huri.components.data_annotaion.utils import SelectRegionBasePhoxiStreamer, img_to_n_channel, Label
from utils.gui_utils import phoxi_map_pcd_2_poly_mask, highlight_mask
from huri.vision.phoxi_capture import depth2gray_map, enhance_gray_img, vision_pipeline

VERSION = ANNOTATION.VERSION
Label = ANNOTATION.LABEL
Bbox = ANNOTATION.BBOX_XYXY
Save_format = ANNOTATION.ON_TABLE_ANNOTATION_SAVE_FORMAT


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


def map_pcd_img2d(extracted_pcd_idx, img_sz):
    h, w = img_sz[0], img_sz[1]
    idx_in_pixel = np.unravel_index(extracted_pcd_idx, (h, w))
    h1, h2 = idx_in_pixel[0].min(), idx_in_pixel[0].max()
    w1, w2 = idx_in_pixel[1].min(), idx_in_pixel[1].max()
    return h1, w1, h2, w2


def plot_dash_frame(frame_size=np.array([.1, .1, .01]),
                    x_range=np.array([0, 1]),
                    y_range=np.array([-.5, .5]),
                    z_range=np.array([0, .1])):
    x_grid = np.arange(x_range[0], x_range[1], frame_size[0])
    y_grid = np.arange(y_range[0], y_range[1], frame_size[1])
    z_grid = np.arange(z_range[0], z_range[1], frame_size[2])
    _grid = np.asarray(np.meshgrid(x_grid, y_grid, z_grid)).T.reshape(-1, 3)
    [gm.gen_frame_box(extent=frame_size, homomat=
    rm.homomat_from_posrot(_pos), rgba=np.array([0, 1, 0, .4])).attach_to(base) for _pos in _grid]


class SelectRegion(SelectRegionBasePhoxiStreamer):
    def __init__(self, showbase,
                 save_path="./",
                 pcd_affine_matrix=None,
                 streamer_ip=None,
                 debug_path=None,
                 toggle_debug=False,
                 img_type="gray"):

        super(SelectRegion, self).__init__(showbase,
                                           param_path=SEL_PARAM_PATH.joinpath("params_sr.json"),
                                           vision_server_ip=streamer_ip,
                                           pcd_affine_matrix=pcd_affine_matrix,
                                           img_type=img_type,
                                           toggle_debug=toggle_debug,
                                           debug_path=debug_path)
        # initialize
        self._init_gui()

        # label info and save path
        self.save_path = fs.Path(save_path)

        if toggle_debug:
            self._update(None)

    def _init_gui(self):
        self._x_bound = np.array([-.3, 1])
        self._y_bound = np.array([-.6, .6])
        self._z_bound = np.array([-.1, .9])
        _row = 1
        self._xrange = [
            self._panel.add_scale(f"x-", default_value=self._params['last_selection_val_x'][0], command=self._update,
                                  val_range=[self._x_bound[0], self._x_bound[1]],
                                  pos=(_row, 0)),
            self._panel.add_scale(f"x+", default_value=self._params['last_selection_val_x'][1], command=self._update,
                                  val_range=[self._x_bound[0], self._x_bound[1]],
                                  pos=(_row, 1))]
        self._yrange = [
            self._panel.add_scale(f"y-", default_value=self._params['last_selection_val_y'][0], command=self._update,
                                  val_range=[self._y_bound[0], self._y_bound[1]],
                                  pos=(_row + 1, 0)),
            self._panel.add_scale(f"y+", default_value=self._params['last_selection_val_y'][1], command=self._update,
                                  val_range=[self._y_bound[0], self._y_bound[1]],
                                  pos=(_row + 1, 1))]
        # init z direction in tcp coordinate

        self._zrange = [
            self._panel.add_scale(f"z-", default_value=self._params['last_selection_val_z'][0], command=self._update,
                                  val_range=[self._z_bound[0], self._z_bound[1]],
                                  pos=(_row + 2, 0)),
            self._panel.add_scale(f"z+", default_value=self._params['last_selection_val_z'][1], command=self._update,
                                  val_range=[self._z_bound[0], self._z_bound[1]],
                                  pos=(_row + 2, 1))]
        _row += 3

        self._panel.add_button(text="Get data and render", command=self._render_acquire_data, pos=(_row, 0))
        self._panel.add_button(text="Get data and render and save", command=self._render_acquire_data_and_save,
                               pos=(_row, 1))

    def _render_acquire_data(self):
        # acquire vision data
        if self._streamer is None:
            self._pcd, self._texture, self._depth_img, self._rgb_texture, self._extcam_img = None, None, None, None, None
            print("Cannot acquire data")
        else:
            self._pcd, self._texture, self._depth_img, \
                self._rgb_texture, self._extcam_img = vision_pipeline(self._streamer,
                                                                      rgb_texture=True if self._img_type == "color" else False,
                                                                      get_depth=True)
            self._assign_img()

        if self._pcd is None or self._depth_img is None or self._img is None:
            return

        if self._pcd_affine_mat is not None:
            self._pcd_aligned = rm.homomat_transform_points(self._pcd_affine_mat, points=self._pcd)

        return self._update(None)

    def _render_acquire_data_and_save(self):
        annotaions = self._render_acquire_data()
        fs.dump_pickle(tuple(Save_format(version=VERSION,
                                         pcd=self._pcd,
                                         gray_img=self._texture,
                                         extcam_img=self._extcam_img,
                                         depth_img=self._depth_img,
                                         annotations=annotaions)),
                       self.save_path / f"{strftime('%Y%m%d-%H%M%S')}.pkl")

    def _update(self, x):
        bad_img = False

        x_range = np.array([_.get() for _ in self._xrange])
        y_range = np.array([_.get() for _ in self._yrange])
        z_range = np.array([_.get() for _ in self._zrange])

        # depth_img_labeled = img_to_n_channel(depth2gray_map(self._depth_img.copy()))
        if self._img_type == "gray":
            img_labeled = img_to_n_channel(enhance_gray_img(self._img.copy()))
        else:
            img_labeled = self._img.copy()

        # render pcd
        self._render_pcd(self._pcd_aligned if self._pcd_aligned is not None else self._pcd)
        labels = []
        # for the rack
        data = extract_pixel(self._pcd_aligned, img_sz=self._texture.shape, extract_area=(x_range[0:2],
                                                                                          y_range[0:2],
                                                                                          z_range[0:2]))
        if data is None:
            bad_img = True
        else:
            label_name = "rack"
            (h1, w1, h2, w2), polygon_mask, extracted_pcd_idx = data

            label_info = tuple(Label(label_name=label_name,
                                     version=VERSION,
                                     img_type="gray",
                                     bboxes=tuple(Bbox(w1=w1, h1=h1, w2=w2, h2=h2)),
                                     polygons=polygon_mask,
                                     extracted_pcd_idx=extracted_pcd_idx, ))
            labels.append(label_info)
            # depth_img_labeled = cv2.rectangle(depth_img_labeled, (w1, h1), (w2, h2),
            #                                   color=(255, 0, 0), thickness=3)
            img_labeled = cv2.rectangle(img_labeled, (w1, h1), (w2, h2),
                                        color=(255, 0, 0), thickness=3)
            img_labeled = highlight_mask(img_labeled, polygon_mask)
            gm.gen_pointcloud(
                self._pcd_aligned[extracted_pcd_idx] if self._pcd_aligned is not None else self._pcd[extracted_pcd_idx],
                rgbas=[[1, 0, 0, 1]]).attach_to(self._np_pcd)

        self._render_img(img_labeled)

        if self._np_pcd is not None:
            trans = np.eye(4)
            x_offset = np.sum(self._x_bound[:2])
            y_offset = np.sum(self._y_bound[:2])
            z_offset = np.sum(self._z_bound[:2])
            print(x_offset, y_offset, z_offset)
            trans[:3, 3] += (x_offset + (x_range[1] - self._x_bound[1]) - (self._x_bound[0] - x_range[0])) * trans[:3,
                                                                                                             0] / 2
            trans[:3, 3] += (y_offset + (y_range[1] - self._y_bound[1]) - (self._y_bound[0] - y_range[0])) * trans[:3,
                                                                                                             1] / 2
            trans[:3, 3] += (z_offset + (z_range[1] - self._z_bound[1]) - (self._z_bound[0] - z_range[0])) * trans[:3,
                                                                                                             2] / 2
            gm.gen_box([x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]],
                       homomat=trans, rgba=(1, 0, 1, .2)).attach_to(self._np_pcd)
        self._save_params()
        if bad_img:
            return None
        return labels

    def _save_params(self):
        x_range = np.array([_.get() for _ in self._xrange])
        y_range = np.array([_.get() for _ in self._yrange])
        z_range = np.array([_.get() for _ in self._zrange])
        for i in range(len(x_range)):
            self._params["last_selection_val_x"][i] = x_range[i]
        for i in range(len(y_range)):
            self._params["last_selection_val_y"][i] = y_range[i]
        for i in range(len(z_range)):
            self._params["last_selection_val_z"][i] = z_range[i]
        fs.dump_json(self._params, "params/params_sr.json", reminder=False)


def test(save_path,
         debug=False):
    # 3D environment
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    # std_out = RenderController(base.tkRoot, base)
    std_out = None
    # Get Data From Camera

    # Init detector
    affine_mat = np.asarray(fs.load_json(SENSOR_INFO.PNT_CLD_CALIBR_MAT_PATH)["affine_mat"])
    gm.gen_frame().attach_to(base)
    sr = SelectRegion(showbase=base,
                      pcd_affine_matrix=affine_mat,
                      streamer_ip=SENSOR_INFO.IP_ADR_DEPTH_SENSOR_DEBUG,
                      save_path=save_path,
                      debug_path=None,
                      toggle_debug=debug,
                      img_type="gray")

    base.startTk()
    base.tkRoot.withdraw()
    base.run()


if __name__ == "__main__":
    SAVE_PATH = DATA_ANNOT_PATH.joinpath("EXP", "RACK")
    test(save_path=SAVE_PATH)
