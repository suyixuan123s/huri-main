from time import strftime

import matplotlib
import numpy as np

matplotlib.use('TkAgg')

from huri.core.common_import import *
from huri.core.constants import SENSOR_INFO
from huri.components.pipeline.data_pipeline import RenderController
from huri.vision.phoxi_capture import vision_pipeline, SensorMarkerHandler, vision_read_data
import huri.components.utils.plot_projection as pp
from huri.components.vision.tube_detector import TestTubeDetector
from huri.definitions.rack_def import Rack_Hard_Proto

IP_ADR = SENSOR_INFO.IP_ADR_DEPTH_SENSOR_DEBUG
SAVE_PATH = fs.workdir / "data" / "vision_exp" / f"{strftime('%Y%m%d-%H%M%S')}.pkl"  # None: Do not save the data captured by phoxi
AFFINE_MAT_PATH = fs.workdir / "data/calibration/qaqqq.json"
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
        # filename = workdir / "data" / "vision_exp" / "20220104-185633.pkl"
        # filename = workdir / "data" / "vision_exp" / "20220201-032152.pkl"
        # filename = workdir / "data" / "vision_exp" / "20220201-041049.pkl"
        filename = fs.workdir / "data" / "vision_exp" / "20231130-155725.pkl"
        # filename = fs.workdir / "data" / "vision_exp" / "6_6_6_2" / "exp_20211223-053040.pkl"
        pcd, img, depth_img, _, _ = vision_read_data(filename)
    else:
        pcd, img, depth_img, _, _ = vision_pipeline(streamer=SensorMarkerHandler(IP_ADR),
                                                    dump_path=SAVE_PATH, rgb_texture=False, )
    # Init detector
    detector = TestTubeDetector(affine_mat_path=AFFINE_MAT_PATH)
    # detector = TestTubeDetector(affine_mat_path=AFFINE_MAT_PATH, use_last_available_rack_yolo_pos=False,
    #                              rack_height_lower=.01, rack_height_upper=.03)
    affine_mat = np.asarray(
        fs.load_json(AFFINE_MAT_PATH)['affine_mat'])
    pcd_r = rm.homomat_transform_points(affine_mat, points=pcd)
    enhanced_image = cv2.equalizeHist(img)
    enhanced_image = np.repeat(enhanced_image[..., None], 3, axis=-1)
    enhanced_image = enhanced_image.reshape(-1, 3)
    enhanced_image = np.concatenate((enhanced_image / 255, np.ones((len(enhanced_image), 1,))), axis=1)
    gm.gen_pointcloud(pcd_r, enhanced_image).attach_to(base)
    base.run()
    # Detect
    # for i in range(10):
    detected_test_tubes, tube_rack, rack_tf, yolo_img = detector.analyze_scene(rack_proto=Rack_Hard_Proto, pcd=pcd,
                                                                               texture_img=img[:, :, 0],
                                                                               std_out=std_out, canvas=canvas,
                                                                               save_detect=True,
                                                                               toggle_detect_tube_pos=True,
                                                                               toggle_yolo=True)
    print(repr(rack_tf))
    gm.gen_frame(rack_tf[:3, 3], rack_tf[:3, :3]).attach_to(base)
    # detector.analyze_scene_2(pcd=pcd, texture_img=img[:, :, 0], std_out=std_out, canvas=canvas, toggle_yolo=True)
    tube_rack.gen_mesh_model().attach_to(base)
    # Show results
    # canvas.show()
    # base.startTk()
    # base.tkRoot.withdraw()
    base.run()


if __name__ == "__main__":
    test()
