from huri.core.file_sys import workdir
from huri.components.pipeline.data_pipeline import RenderController
from huri.vision.phoxi_capture import vision_pipeline, SensorMarkerHandler
from huri.core.common_import import *
import huri.components.utils.plot_projection as pp
from huri.components.vision.tube_detector import TestTubeDetector
import matplotlib
from time import strftime

matplotlib.use('TkAgg')

IP_ADR = "192.168.125.100:18300"
SAVE_PATH = workdir / "data" / "vision_exp" / f"{strftime('%Y%m%d-%H%M%S')}.pkl"  # None: Do not save the data captured by phoxi
AFFINE_MAT_PATH = workdir / "data/calibration/qaqqq.json"
DEBUG = True


def test():
    # 3D environment
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    # init the simulation robot
    yumi_robot = ym.Yumi(enable_cc=True)
    # yumi_robot.gen_meshmodel().attach_to(base)

    std_out = RenderController(base.tkRoot, base)
    # 2D canvas to show the projection of point clouds
    canvas = pp.Plot(x_size=500)
    # Get Data From Camera
    if DEBUG:
        # filename = workdir / "data" / "vision_exp" / "20220127-195849.pkl"
        filename = workdir / "data" / "vision_exp" / "20220127-171553.pkl"
        # filename = workdir / "data" / "vision_exp" / "20220201-032152.pkl"
        # filename = workdir / "data" / "vision_exp" / "20220201-041049.pkl"
        # filename = fs.workdir / "data" / "vision_exp" / "4_4_4_2" / "exp_20211223-040745.pkl"
        pcd, img = fs.load_pickle(filename)
    else:
        pcd, img = vision_pipeline(streamer=SensorMarkerHandler(IP_ADR),
                                   dump_path=SAVE_PATH)
    # Init detector
    detector = TestTubeDetector(affine_mat_path=AFFINE_MAT_PATH)
    # detector = TestTubeDetector(affine_mat_path=AFFINE_MAT_PATH, use_last_available_rack_yolo_pos=False,
    #                              rack_height_lower=.01, rack_height_upper=.03)
    affine_mat = np.asarray(
        fs.load_json(AFFINE_MAT_PATH)['affine_mat'])
    pcd_r = rm.homomat_transform_points(affine_mat, points=pcd)
    gm.gen_pointcloud(pcd_r, [[0, 0, 0, .3]]).attach_to(base)
    # Detect
    detector.analyze_scene(pcd=pcd, texture_img=img[:, :, 0], std_out=std_out, canvas=canvas, toggle_yolo=True)

    # Show results
    canvas.show()

    r_img= canvas.get_img()
    r_img

    # base.startTk()
    # base.tkRoot.withdraw()
    # base.run()


if __name__ == "__main__":
    test()
