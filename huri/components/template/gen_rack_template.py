from huri.vision.phoxi_capture import vision_read_data
from huri.components.vision.tube_detector import TestTubeDetector
from huri.core.constants import SENSOR_INFO
from huri.core.common_import import rm, np, gm, fs
import huri.vision.pnt_utils as pntu
import cv2


def gen_rack_template(data_path):
    detector = TestTubeDetector(affine_mat_path=SENSOR_INFO.PNT_CLD_CALIBR_MAT_PATH, )
    pcd, img, depth_img, rgb_texture, extcam_img = vision_read_data(data_path)

    yolo_img, yolo_results = detector._yolo_detect(texture_img=img[:, :, 0], toggle=True)

    rack_yolo_info = yolo_results[yolo_results[:, 0] == 0]
    if len(rack_yolo_info) < 1:
        raise Exception("No yolo information")
    lt_pos, rb_pos = rack_yolo_info.ravel()[1:5].reshape(2, 2)

    pcd = rm.homomat_transform_points(detector.affine_mat, pcd)
    rack_pcd_template = pntu.extract_pcd_by_yolo(pcd, img.shape,
                                                 bound_lt=lt_pos, bound_rb=rb_pos,
                                                 enlarge_detection_ratio=.2)
    rack_pcd_template = rack_pcd_template[np.where((rack_pcd_template[:, 2] > 0.03)
                                                   & (rack_pcd_template[:, 2] < 0.05))]
    gm.gen_pointcloud(rack_pcd_template, [[0, 1, 1, .3]]).attach_to(base)
    return rack_pcd_template


if __name__ == "__main__":
    from huri.definitions.rack_geom import SlotGeom, RackGeom, Mm, rack_soft_geom
    from huri.components.vision.extract import oriented_box_icp
    from huri.core.common_import import wd, cm

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

    tube_rack_cm = cm.CollisionModel(str(fs.workdir_model / "tubestand_light_thinner.stl"))
    tube_rack_cm.set_rgba([0, .3, 1, 1])
    tube_rack_cm.attach_to(base)
    # tube_rack_cm.show_localframe()
    # exit(0)
    # keypoints = pntu.detect_keypoint_iss_from_model(str(workdir_model / "tubestand_light_thinner.stl"))
    #
    keypoints = rack_soft_geom.pcd
    # keypoints = keypoints[np.where((keypoints[:, 2] > 0.055))]
    gm.gen_pointcloud(keypoints, rgbas=[[1, 0, 0, 1]], pntsize=5).attach_to(base)

    rack_transform = oriented_box_icp(pcd=keypoints,
                                      downsampling_voxelsize=0.003)
    # rack_pcd_template = gen_rack_template(workdir_vision / "data" / "20210707-214502.pkl")
    rack_pcd_template = gen_rack_template(fs.workdir_data / "vision_exp" / "20220805-011940.pkl")

    gm.gen_pointcloud(rm.homomat_transform_points(rack_transform, rack_pcd_template), ).attach_to(base)
    # transformation = pntu.global_registration(
    #     rack_pcd_template,
    #     keypoints,
    # ).copy()
    # print(len(rack_pcd_template))
    # rack_pcd_template = dcuf.remove_outlier(rack_pcd_template, downsampling_voxelsize=None, radius=0.1)
    # print(len(rack_pcd_template))
    # transformation[:3, 3] = transformation[:3, 3].copy() - np.array([0, 0, 0.055])
    # gm.gen_pointcloud(rm.homomat_transform_points(transformation, rack_pcd_template), ).attach_to(base)
    fs.dump_pickle(keypoints,
                   path=fs.workdir_vision / "template" / "rack_2")
    base.run()
