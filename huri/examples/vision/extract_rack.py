from huri.core.file_sys import workdir
from huri.components.pipeline.data_pipeline import RenderController
from huri.vision.phoxi_capture import vision_pipeline, SensorMarkerHandler
from huri.core.common_import import *
import huri.components.vision.extract as extract
from huri.components.vision.tube_detector import TestTubeDetector
import matplotlib
from huri.core.constants import SENSOR_INFO
import huri.core.utils as hcu

matplotlib.use('TkAgg')

SAVE_PATH = None  # None: Do not save the data captured by phoxi
# Colors
color_group = [hcu.color_hex2oct(color) for color in hcu.color_hex["Beach Towels"]]


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


def test():
    DEBUG = False
    # 3D environment
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    # std_out = RenderController(base.tkRoot, base)
    std_out = None
    # Get Data From Camera
    if DEBUG:
        filename = workdir / "data" / "vision_exp" / "20220613-052249.pkl"
        pcd, texture, depth_img, rgb_texture, extcam_img = fs.load_pickle(filename)
    else:
        pcd, texture, depth_img, rgb_texture, extcam_img = vision_pipeline(SensorMarkerHandler(ip_adr=SENSOR_INFO.IP_ADR_DEPTH_SENSOR_DEBUG),
                                   dump_path=SAVE_PATH, rgb_texture=False)
    # Init detector
    detector = TestTubeDetector(affine_mat_path=SENSOR_INFO.PNT_CLD_CALIBR_MAT_PATH)
    # Detect
    yolo_img, yolo_results = detector._yolo_detect(texture_img=texture, toggle=True)
    # if there are more than one rack, choose the rack with higher confident
    pcd_calibrated = rm.homomat_transform_points(homomat=detector.affine_mat, points=pcd)
    # plot pcd
    # gm.gen_pointcloud(pcd_calibrated, ).attach_to(base)

    UPPER_BOUND = 0.05 + .015
    LOWER_BOUND = 0.02 + .02
    # plot a dash coordinate frame for reference
    plot_dash_frame(z_range=np.array([LOWER_BOUND, UPPER_BOUND]))
    rack_trans, rack_pcd, outliner = extract.extrack_rack(pcd=pcd_calibrated,
                                                          results=yolo_results,
                                                          img_shape=texture.shape,
                                                          std_out=std_out,
                                                          height_lower=LOWER_BOUND,
                                                          height_upper=UPPER_BOUND)
    color_group[0] = [0,0,0,1]
    gm.gen_pointcloud(rack_pcd, rgbas=[color_group[0]]).attach_to(base)
    gm.gen_pointcloud(outliner, rgbas=[color_group[0]]).attach_to(base)
    # gm.gen_pointcloud(rack_pcd, rgbas=[color_group[2]]).attach_to(base)

    base.run()


if __name__ == "__main__":
    test()
