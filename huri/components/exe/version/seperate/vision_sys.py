__VERSION__ = "0.0.1_seperate"

import time
from typing import Literal

from huri.core.common_import import wd, ym, fs, np, gm, rm
from huri.core.constants import SENSOR_INFO
from huri.components.exe.utils import gen_camera_obs, capture_vision_info, init_real_rbt, is_restart_planning
from huri.definitions.rack_def import TubeRack
from huri.core.print_tool import text_pd, print_with_border
from huri.definitions.utils_structure import MotionElement, MotionBatch, MotionBatchPPP
# motion planning
from huri.components.planning.common_grasp_seq import CommonGraspSolver
import huri.components.planning.symmetric_pick_and_place_planner_c as ppp
# vision system
from huri.definitions.rack_def import ArrangeTubeSolSeq, Rack_Soft_Proto
from huri.components.vision.tube_detector import TestTubeDetector
# task planning
from huri.learning.env.arrangement_planning_rack.utils import isdone
from huri.components.task_planning.tube_puzzle_learning_solver import DQNSolver
from huri.vision.phoxi_capture import vision_pipeline, SensorMarkerHandler, vision_read_data
from huri.vision.pnt_utils import RACK_HARD_TEMPLATE, RACK_SOFT_TEMPLATE
import huri.components.vision.extract as extract
from huri.components.vision.extract import dcuf
from basis.trimesh import Trimesh, bounds
import cv2

GOAL_PATTERN = np.array([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]])

RBT_END_CONF = np.array([-0.0137881, -0.97703532, -1.50848807, 0.87929688, -1.99840199,
                         0.13788101, 1.51669112])

DEBUG_CACHE = []


def pos_filter(pcd: np.ndarray, min: float, max: float, filter_axis: Literal['x', 'y', 'z'] = "x"):
    axis_id = 0
    if filter_axis == "x":
        axis_id = 0
    elif filter_axis == "y":
        axis_id = 1
    elif filter_axis == "z":
        axis_id = 2
    else:
        raise Exception("Undefined filter axis")
    return pcd[(pcd[:, axis_id] > min) & (pcd[:, axis_id] < max)]


def task_planning(tube_rack: TubeRack, solver: DQNSolver, infeasible_info_dict: dict,
                  infeasible_local_pattern: dict) -> ArrangeTubeSolSeq:
    task_sols = ArrangeTubeSolSeq(rack=tube_rack,
                                  solution=solver.solve(tube_rack.rack_status,
                                                        GOAL_PATTERN,
                                                        infeasible_info_dict,
                                                        infeasible_local_pattern,
                                                        toggle_result=False),
                                  tf=tube_rack.get_homomat())
    return task_sols


def vision_system(detector: TestTubeDetector,
                  toggle_yolo=False,
                  toggle_save=False,
                  debug_filepath=None,
                  toggle_obb=True,
                  is_debug=False) -> (TubeRack, np.ndarray):
    toggle_debug = True if debug_filepath is not None else False
    # capture the data through vision sensor
    if is_debug:
        filename = "D:\chen\huri_shared\huri\data\\vision_exp\\20220802-182310.pkl"
        pcd, img, depth_img, _, _ = vision_read_data(filename)
    else:
        pcd, img, depth_img = capture_vision_info(ip_adr=SENSOR_INFO.IP_ADR_DEPTH_SENSOR_DEBUG,
                                                  debug_filename=debug_filepath,
                                                  toggle_save=toggle_save,
                                                  toggle_debug=toggle_debug, )  # TODO revise debug back to DEBUG
    affine_mat = detector.affine_mat
    pcd_r = rm.homomat_transform_points(affine_mat, points=pcd)
    gm.gen_pointcloud(pcd_r, ).attach_to(base)

    z_min = .05
    z_max = .07
    x_min = .13
    x_max = 1
    downsampling_voxelsize: float = 0.002
    # gm.gen_frame(pos = [.13,0,0]).attach_to(base)
    pcd_rack_group = pos_filter(pos_filter(pcd_r, min=z_min, max=z_max, filter_axis="z"), min=x_min, max=x_max,
                                filter_axis="x")
    pcd_trimesh = Trimesh(vertices=dcuf.remove_outlier(src_nparray=pcd_rack_group.copy(),
                                                       downsampling_voxelsize=downsampling_voxelsize,
                                                       radius=downsampling_voxelsize * 2))
    orient_inv, extent = bounds.oriented_bounds(mesh=pcd_trimesh)
    orient = np.linalg.inv(orient_inv)

    init_homo = np.asarray(orient).copy()
    z_sim = init_homo[:3, :3].T.dot(np.array([0, 0, 1]))
    z_ind = np.argmax(abs(z_sim))
    z_d = np.sign(z_sim[z_ind]) * init_homo[:3, z_ind]
    x_sim = init_homo[:3, :3].T.dot(np.array([1, 0, 0]))
    x_ind = np.argmax(abs(x_sim))
    x_d = np.sign(x_sim[x_ind]) * init_homo[:3, x_ind]
    y_d = np.cross(z_d, x_d)
    init_homo[:3, :3] = np.array([x_d, y_d, z_d]).T

    # gm.gen_pointcloud(pcd_rack_group, [[1, 0, 0, 1]]).attach_to(base)

    pcd_rack_group_org = rm.homomat_transform_points(np.linalg.inv(init_homo), pcd_rack_group)
    first_rack_pcd = pcd_rack_group[(pcd_rack_group_org[:, 1] > -0.096 * 2) & (pcd_rack_group_org[:, 1] < -0.096 * 1)]
    second_rack_pcd = pcd_rack_group[(pcd_rack_group_org[:, 1] > -0.096 * 1) & (pcd_rack_group_org[:, 1] < 0)]
    third_rack_pcd = pcd_rack_group[(pcd_rack_group_org[:, 1] > 0) & (pcd_rack_group_org[:, 1] < 0.096 * 1)]
    forth_rack_pcd = pcd_rack_group[(pcd_rack_group_org[:, 1] > 0.096 * 1) & (pcd_rack_group_org[:, 1] < 0.096 * 2)]

    first_rack_pcd = pos_filter(first_rack_pcd, min=.06, max=.07, filter_axis="z")
    second_rack_pcd = pos_filter(second_rack_pcd, min=.06, max=.07, filter_axis="z")
    third_rack_pcd = pos_filter(third_rack_pcd, min=.06, max=.07, filter_axis="z")
    forth_rack_pcd = pos_filter(forth_rack_pcd, min=.06, max=.07, filter_axis="z")

    if toggle_debug:
        gm.gen_pointcloud(first_rack_pcd, rgbas=[rm.random_rgba()]).attach_to(base)
        gm.gen_pointcloud(second_rack_pcd, rgbas=[rm.random_rgba()]).attach_to(base)
        gm.gen_pointcloud(third_rack_pcd, rgbas=[rm.random_rgba()]).attach_to(base)
        gm.gen_pointcloud(forth_rack_pcd, rgbas=[rm.random_rgba()]).attach_to(base)
    if toggle_obb:
        obb_gm = gm.GeometricModel(initor=pcd_trimesh.bounding_box_oriented)
        obb_gm.set_rgba([0, 0, 1, .3])
        obb_gm.attach_to(base)
        gm.gen_frame(pos=init_homo[:3, 3],
                     rotmat=init_homo[:3, :3]).attach_to(obb_gm)

    # yolo detect the scene
    yolo_img, yolo_results = detector._yolo_detect(texture_img=img[:, :, 0], toggle=toggle_yolo)
    if toggle_save:
        cv2.imwrite("yolo_tmp.jpg", yolo_img)

        # first rack
    rack_transform = extract.oriented_box_icp(pcd=first_rack_pcd,
                                              pcd_template=RACK_SOFT_TEMPLATE,
                                              downsampling_voxelsize=downsampling_voxelsize, )

    if rack_transform[:3, 0].T.dot(np.array([0, 1, 0])) > 0:
        rack_transform[:3, :3] = np.dot(rm.rotmat_from_axangle(rack_transform[:3,2], np.radians(180)),rack_transform[:3, :3])

    rack_height = Rack_Soft_Proto._geom.height
    rack_transform[:3, 3] = rack_transform[:3, 3] - rack_transform[:3, 2] * rack_height
    undetected_yolo_results, tube_rack_1, rack_tf_1 = detector.analyze_tubes_given_rack_tf_yolo(
        rack_proto=Rack_Soft_Proto,
        rack_tf=rack_transform,
        pcd=pcd,
        yolo_results=yolo_results,
        yolo_img=yolo_img,
        std_out=None,
        toggle_detect_tube_pos=True,
        save_detect=toggle_save)
    # second rack
    rack_transform = extract.oriented_box_icp(pcd=second_rack_pcd,
                                              pcd_template=RACK_SOFT_TEMPLATE,
                                              downsampling_voxelsize=downsampling_voxelsize, )
    if rack_transform[:3, 0].T.dot(np.array([0, 1, 0])) > 0:
        rack_transform[:3, :3] = np.dot(rm.rotmat_from_axangle(rack_transform[:3,2], np.radians(180)),rack_transform[:3, :3])

    rack_transform[:3, 3] = rack_transform[:3, 3] - rack_transform[:3, 2] * rack_height
    undetected_yolo_results, tube_rack_2, rack_tf_2 = detector.analyze_tubes_given_rack_tf_yolo(
        rack_proto=Rack_Soft_Proto,
        rack_tf=rack_transform,
        pcd=pcd,
        yolo_results=undetected_yolo_results,
        yolo_img=yolo_img,
        std_out=None,
        toggle_detect_tube_pos=True,
        save_detect=toggle_save)
    # third rack
    rack_transform = extract.oriented_box_icp(pcd=third_rack_pcd,
                                              pcd_template=RACK_SOFT_TEMPLATE,
                                              downsampling_voxelsize=downsampling_voxelsize, )

    if rack_transform[:3, 0].T.dot(np.array([0, 1, 0])) > 0:
        rack_transform[:3, :3] = np.dot(rm.rotmat_from_axangle(rack_transform[:3,2], np.radians(180)),rack_transform[:3, :3])

    rack_transform[:3, 3] = rack_transform[:3, 3] - rack_transform[:3, 2] * rack_height
    undetected_yolo_results, tube_rack_3, rack_tf_3 = detector.analyze_tubes_given_rack_tf_yolo(
        rack_proto=Rack_Soft_Proto,
        rack_tf=rack_transform,
        pcd=pcd,
        yolo_results=undetected_yolo_results,
        yolo_img=yolo_img,
        std_out=None,
        toggle_detect_tube_pos=True,
        save_detect=toggle_save)
    # forth rack
    rack_transform = extract.oriented_box_icp(pcd=forth_rack_pcd,
                                              pcd_template=RACK_SOFT_TEMPLATE,
                                              downsampling_voxelsize=downsampling_voxelsize, )

    if rack_transform[:3, 0].T.dot(np.array([0, 1, 0])) > 0:
        rack_transform[:3, :3] = np.dot(rm.rotmat_from_axangle(rack_transform[:3,2], np.radians(180)),rack_transform[:3, :3])

    rack_transform[:3, 3] = rack_transform[:3, 3] - rack_transform[:3, 2] * rack_height
    undetected_yolo_results, tube_rack_4, rack_tf_4 = detector.analyze_tubes_given_rack_tf_yolo(
        rack_proto=Rack_Soft_Proto,
        rack_tf=rack_transform,
        pcd=pcd,
        yolo_results=undetected_yolo_results,
        yolo_img=yolo_img,
        std_out=None,
        toggle_detect_tube_pos=True,
        save_detect=toggle_save)

    return (tube_rack_1, tube_rack_2, tube_rack_3, tube_rack_4), (rack_tf_1, rack_tf_2, rack_tf_3, rack_tf_4)


def main(open_jaw_width=.034,
         depth_sensor_debug_path=None,
         debug=False,
         retry_num=3,  # number of retry when grasping failed
         goal_place_offset_dis=.04,
         is_vision_feedback=True,
         toggle_save=False, ):
    # Simulation Environment
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    # vision system
    detector = TestTubeDetector(affine_mat_path=SENSOR_INFO.PNT_CLD_CALIBR_MAT_PATH, )
    (tube_rack_1, tube_rack_2, tube_rack_3, tube_rack_4), (rack_tf_1, rack_tf_2, rack_tf_3, rack_tf_4) = vision_system(
        detector, debug_filepath=depth_sensor_debug_path,
        toggle_save=toggle_save)
    tube_rack_1.gen_mesh_model(True).attach_to(base)
    tube_rack_2.gen_mesh_model(True).attach_to(base)
    tube_rack_3.gen_mesh_model(True).attach_to(base)
    tube_rack_4.gen_mesh_model(True).attach_to(base)
    # fs.dump_pickle(data=[tube_rack_1, tube_rack_2, tube_rack_3, tube_rack_4], path="rack.pkl", )
    base.run()


if __name__ == '__main__':
    main(open_jaw_width=.034,
         debug=False,
         depth_sensor_debug_path=None,
         retry_num=3,
         goal_place_offset_dis=.04,
         is_vision_feedback=True,
         toggle_save=False)
