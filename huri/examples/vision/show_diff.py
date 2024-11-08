from huri.core.file_sys import workdir
from huri.components.pipeline.data_pipeline import RenderController
from huri.vision.phoxi_capture import vision_pipeline, SensorMarkerHandler
from huri.core.common_import import *
import huri.components.utils.plot_projection as pp
from huri.components.vision.tube_detector import TestTubeDetector
import matplotlib
from time import strftime
import huri.components.vision.extract as extract

matplotlib.use('TkAgg')

SAVE_PATH = workdir / "data" / "vision_exp" / f"{strftime('%Y%m%d-%H%M%S')}.pkl"  # None: Do not save the data captured by phoxi
AFFINE_MAT_PATH = workdir / "data/calibration/qaqqq.json"


def test():
    # 3D environment
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    # init the simulation robot
    yumi_robot = ym.Yumi(enable_cc=True)
    # yumi_robot.gen_meshmodel().attach_to(base)

    std_out = RenderController(base.tkRoot, base)
    # 2D canvas to show the projection of point clouds
    canvas = pp.Plot(x_size=500)
    # Init detector
    detector = TestTubeDetector(affine_mat_path=AFFINE_MAT_PATH)

    UPPER_BOUND = 0.05 + .015
    LOWER_BOUND = 0.02 + .02
    # Get Data From Camera
    filename = workdir / "data" / "vision_exp" / "20220114-175147.pkl"
    pcd_1, img_1 = fs.load_pickle(filename)
    yolo_img, yolo_results = detector._yolo_detect(texture_img=img_1, toggle=False)
    pcd_calibrated_1 = rm.homomat_transform_points(homomat=detector.affine_mat, points=pcd_1)
    # Detect
    tf1, rack_pcd1, outliner = extract.extrack_rack(pcd=pcd_calibrated_1,
                                                   results=yolo_results,
                                                   img_shape=img_1.shape,
                                                   std_out=None,
                                                   height_lower=LOWER_BOUND,
                                                   height_upper=UPPER_BOUND)
    gm.gen_frame(tf1[:3,3], tf1[:3,:3]).attach_to(base)
    gm.gen_pointcloud(rack_pcd1, rgbas=[[1,0,0,1]]).attach_to(base)
    filename = workdir / "data" / "vision_exp" / "20220114-175233.pkl"
    pcd_2, img_2 = fs.load_pickle(filename)
    yolo_img, yolo_results = detector._yolo_detect(texture_img=img_2, toggle=False)
    pcd_calibrated_2 = rm.homomat_transform_points(homomat=detector.affine_mat, points=pcd_2)
    tf2, rack_pcd2, outliner = extract.extrack_rack(pcd=pcd_calibrated_2,
                                                   results=yolo_results,
                                                   img_shape=img_2.shape,
                                                   std_out=None,
                                                   height_lower=LOWER_BOUND,
                                                   height_upper=UPPER_BOUND)
    gm.gen_frame(tf2[:3, 3], tf2[:3, :3]).attach_to(base)
    gm.gen_pointcloud(rack_pcd2, rgbas=[[0, 0, 1, 1]]).attach_to(base)
    print(tf1[:3, 3])
    print(tf2[:3, 3])
    print(np.linalg.norm(tf2[:3, 3] - tf1[:3, 3]))
    #
    # base.startTk()
    # base.tkRoot.withdraw()
    base.run()


if __name__ == "__main__":
    test()
