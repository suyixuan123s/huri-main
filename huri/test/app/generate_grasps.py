import numpy as np
from typing import List
from huri.core.file_sys import dump_json, workdir
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.yumi_gripper.yumi_gripper as yumi_gripper
import trimesh as tm
from basis.robot_math import angle_between_vectors


def add_symmetric_attribute(grasp_info_list: List[List],
                            symmetry_axis: np.ndarray) -> List[List]:
    angles = np.zeros(len(grasp_info_list))
    pos = np.zeros((len(grasp_info_list), 3))
    for idx, graps_info in enumerate(grasp_info_list):
        angles[idx] = angle_between_vectors(graps_info[2][:, 2], symmetry_axis)
        pos[idx] = graps_info[1]
    group = []
    for uni_vector in np.unique(pos.round(decimals=7), axis=0):
        pos_same_indx = (np.abs(pos - uni_vector) < 1e-7).all(axis=1).nonzero()[0]
        angles_of_pos_indx = angles[pos_same_indx]
        uni_angles = np.unique(angles_of_pos_indx.round(decimals=7))
        for uni_angle in uni_angles:
            group_ele = pos_same_indx[(np.abs(angles_of_pos_indx - uni_angle) < 1e-7).nonzero()]
            if len(group_ele) > 1:
                group.append(group_ele)
    print(len(group))
    for idx, graps_info in enumerate(grasp_info_list):
        group_id = [i for i, g in enumerate(group) if idx in g]
        if len(group_id) > 0:
            graps_info.append(group_id[0])
        else:
            graps_info.append(-1)
        print(group_id)
    return grasp_info_list


def gen_grasps(mdl_path, grasp_name, show=False):
    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)

    # attach object to the world
    mdl_cm = cm.CollisionModel(str(mdl_path))
    mdl_cm.set_rgba([1, 0, 0, 1])
    mdl_cm.attach_to(base)

    # hnd_s
    gripper_s = yumi_gripper.YumiGripper()
    grasp_info_list = gpa.plan_grasps(gripper_s, mdl_cm, max_samples=7)

    print(f"     :: number of grasps:{len(grasp_info_list)}")

    # grasp_info_list = add_symmetric_attribute(grasp_info_list=grasp_info_list,
    #                                           symmetry_axis=tm.load(str(mdl_path)).symmetry_axis)
    dump_json(grasp_info_list, workdir / "data" / "grasps" / grasp_name)

    if show:
        for grasp_info in grasp_info_list:
            aw_width, gl_jaw_center, gl_jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
            gripper_s.fix_to(hnd_pos, hnd_rotmat)
            gripper_s.jaw_to(aw_width)
            gripper_s.gen_meshmodel(rgba=[0, 1, 0, .3]).attach_to(base)
        base.run()


if __name__ == "__main__":
    from huri.definitions.tube_def import TubeType

    # # blue cap
    # obj_path = TubeType.BLUE_CAP_TUBE.model_path
    # gen_grasps(mdl_path=obj_path, grasp_name="blue_cap_tube_grasps.json", show=True)
    # white cap
    obj_path = TubeType.WHITE_CAP_TUBE.model_path
    gen_grasps(mdl_path=obj_path, grasp_name="white_cap_tube_grasps.json", show=True)
    # # purple cap
    # obj_path = TubeType.PURPLE_CAP_TUBE.model_path
    # gen_grasps(mdl_path=obj_path, grasp_name="purple_cap_tube_grasps.json", show=True)
    # # white small cap
    # obj_path = TubeType.WHITE_CAP_SMALL_TUBE.model_path
    # gen_grasps(mdl_path=obj_path, grasp_name="white_cap_small_tube_grasps.json", show=True)
