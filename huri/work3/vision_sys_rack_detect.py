from time import strftime

import matplotlib

matplotlib.use('TkAgg')

from huri.core.common_import import *
from huri.core.constants import SENSOR_INFO
# from huri.vision.phoxi_capture import vision_pipeline, SensorMarkerHandler, vision_read_data
import huri.components.utils.plot_projection as pp
from huri.components.vision.tube_detector import TestTubeDetector, extract
from huri.definitions.rack_def import Rack_Hard_Proto
import huri.vision.pnt_utils as pntu
import cv2

IP_ADR = SENSOR_INFO.IP_ADR_DEPTH_SENSOR_DEBUG
SAVE_PATH = fs.workdir / "data" / "vision_exp" / f"{strftime('%Y%m%d-%H%M%S')}.pkl"  # None: Do not save the data captured by phoxi
AFFINE_MAT_PATH = fs.workdir / "data/calibration/qaqqq.json"
DEBUG = False
from constants import HEIGHT_RANGE

def test(base):
    # 2D canvas to show the projection of point clouds
    canvas = pp.Plot(x_size=500)
    # Get Data From Camera
    # Init detector
    detector = TestTubeDetector(affine_mat_path=None)
    # detector = TestTubeDetector(affine_mat_path=AFFINE_MAT_PATH, use_last_available_rack_yolo_pos=False,
    #                              rack_height_lower=.01, rack_height_upper=.03)
    # c_pcd, pcds = fs.load_pickle("pcd_data")
    c_pcd, pcds = fs.load_pickle("pcd_data_debug")
    # gm.gen_pointcloud(c_pcd, [[0, 0, 0, .3]]).attach_to(base)
    # Detect
    gm.gen_rgb_pointcloud(c_pcd).attach_to(base)
    pcd_ind = pntu.extract_pcd_by_range(pcd=c_pcd[:, :3], z_range=HEIGHT_RANGE,
                                        toggle_debug=True)

    raw_rack_pcd = c_pcd[pcd_ind]
    rack_transform = extract.oriented_box_icp(pcd=raw_rack_pcd,
                                              pcd_template=Rack_Hard_Proto._pcd_template,
                                              downsampling_voxelsize=.007,
                                              toggle_debug=False)
    rack_height = Rack_Hard_Proto._geom.height

    template_pcd_cm = cm.CollisionModel(gm.gen_pointcloud(Rack_Hard_Proto._pcd_template))
    template_pcd_cm.set_homomat(rack_transform)
    template_pcd_cm.attach_to(base)
    base.run()
    rack_transform[:3, 3] = rack_transform[:3, 3] - rack_transform[:3, 2] * rack_height

    rack = Rack_Hard_Proto.copy()
    rack.set_homomat(rack_transform)
    rack_mdl = rack.gen_mesh_model()
    rack_mdl.set_rgba([0, 0, 1, 0.3])
    rack_mdl.attach_to(base)
    print("The rack transformation is: ", rack_transform)

    # gm.gen_rgb_pointcloud(c_pcd[pcd_ind]).attach_to(base)

    import huri.vision.yolov6.detect as yyd

    for i, img in enumerate(imgs):
        pcd = pcds[i]
        yolo_img, yolo_results = detector.yolo_detect(texture_img=img,
                                                      yolo_weights_path="best.pt")
        cv2.imshow("aaa", yolo_img)
        cv2.waitKey(0)
        detected_results, rack_instance, rack_tf = detector.analyze_tubes_given_rack_tf_yolo(rack_proto=Rack_Hard_Proto,
                                                                                             rack_tf=rack_transform,
                                                                                             pcd=pcd,
                                                                                             yolo_results=yolo_results,
                                                                                             yolo_img=yolo_img,
                                                                                             downsampling_voxelsize=.001,
                                                                                             toggle_detect_tube_pos=True)
        measure_pose_err(rack_tf,rack_transform)
        rack_instance.gen_mesh_model(gen_tube=True).attach_to(base)
        base.run()
    # print(repr(rack_tf))
    gm.gen_frame().attach_to(base)
    # detector.analyze_scene_2(pcd=pcd, texture_img=img[:, :, 0], std_out=std_out, canvas=canvas, toggle_yolo=True)

    # Show results
    canvas.show()
    base.run()


if __name__ == "__main__":
    from robot_sim.robots.xarm_lite6_wrs import XArmLite6WRSGripper

    # 3D environment
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    rbt = XArmLite6WRSGripper()
    rbt.gen_meshmodel().attach_to(base)
    test(base)
