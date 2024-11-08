import numpy as np

from huri.core.common_import import *
import huri.vision.pnt_utils as pntu
from huri.core.file_sys import Path, load_pickle, workdir_model, workdir_vision, dump_pickle
import vision.depth_camera.depth_calibrator as dcdc
import vision.depth_camera.util_functions as dcuf

affine_mat, _, _ = dcdc.load_calibration_data(str(workdir_vision / "calibration" / "depth_sensor_calib_mat.pkl"))


def gen_rack_template(template_path):
    _, _, img, result, pcd = load_pickle(template_path)
    for label in result.keys():
        if "rack" in label:
            lt_pos, rb_pos = result[label]
            break
    else:
        return False
    pcd = rm.homomat_transform_points(affine_mat, pcd)
    rack_pcd_template = pntu.extract_pcd_by_yolo(pcd, img.shape,
                                                 bound_lt=lt_pos, bound_rb=rb_pos,
                                                 enlarge_detection_ratio=.2)
    rack_pcd_template = rack_pcd_template[np.where((rack_pcd_template[:, 2] > 0.03)
                                                   & (rack_pcd_template[:, 2] < 0.05))]
    gm.gen_pointcloud(rack_pcd_template, [[0, 1, 1, .3]]).attach_to(base)
    return rack_pcd_template


if __name__ == "__main__":
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

    tube_rack_cm = cm.CollisionModel(str(workdir_model / "tubestand_light_thinner.stl"))
    tube_rack_cm.set_rgba([0, .3, 1, 1])
    tube_rack_cm.attach_to(base)
    # tube_rack_cm.show_localframe()
    # base.run()
    # exit(0)
    # keypoints = pntu.detect_keypoint_iss_from_model(str(workdir_model / "tubestand_light_thinner.stl"))
    #
    keypoints = tube_rack_cm.sample_surface(radius=0.0001, nsample=4000)[0]
    keypoints = keypoints[np.where((keypoints[:, 2] > 0.055))]
    gm.gen_pointcloud(keypoints, rgbas=[[1, 0, 0, 1]], pntsize=5).attach_to(base)
    rack_pcd_template = gen_rack_template(workdir_vision / "data" / "20210707-214502.pkl")
    transformation = pntu.global_registration(
        rack_pcd_template,
        keypoints,
    ).copy()
    print(len(rack_pcd_template))
    rack_pcd_template = dcuf.remove_outlier(rack_pcd_template, downsampling_voxelsize=None, radius=0.1)
    print(len(rack_pcd_template))
    transformation[:3, 3] = transformation[:3, 3].copy() - np.array([0, 0, 0.055])
    gm.gen_pointcloud(rm.homomat_transform_points(transformation, rack_pcd_template), ).attach_to(base)
    # dump_pickle(rm.homomat_transform_points(transformation, rack_pcd_template),
    #             path=workdir_vision / "template" / "rack_1")
    base.run()
