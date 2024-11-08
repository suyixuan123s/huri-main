import numpy as np
import robot_sim.end_effectors.gripper.yumi_gripper.yumi_gripper as yg
import modeling.collision_model as cm
import visualization.panda.world as wd
from grasping.annotation.utils import define_grasp_with_rotation
from huri.definitions.tube_def import TubeType
from huri.core.file_sys import dump_json, workdir
import basis.robot_math as rm
from modeling.model_collection import ModelCollection

base = wd.World(cam_pos=[.5, .5, .3], lookat_pos=[0, 0, 0])
gripper_s = yg.YumiGripper(enable_cc=True)

tube = TubeType.WHITE_CAP_TUBE
# tube = TubeType.PURPLE_CAP_TUBE
# tube = TubeType.BLUE_CAP_TUBE
objpath = tube.model_path
objcm = cm.CollisionModel(objpath)
objcm.attach_to(base)
objcm.show_localframe()
grasp_info_list = []
angle_list = tube.grasps_angle_list
pos_list = tube.grasps_pos_list
for angle in angle_list:
    for pos in pos_list:
        grasp_info_list_tmp = define_grasp_with_rotation(gripper_s,
                                                         objcm,
                                                         gl_jaw_center_pos=pos,
                                                         gl_jaw_center_z=
                                                         np.dot(rm.rotmat_from_axangle(np.array([1, 0, 0]),
                                                                                       -angle),
                                                                rm.unit_vector(np.array([0, 1, 0]))),
                                                         gl_jaw_center_y=
                                                         np.dot(rm.rotmat_from_axangle(np.array([1, 0, 0]),
                                                                                       -angle),
                                                                np.array([0, 0, 1])),
                                                         jaw_width=tube.radius * 2 + 0.002,
                                                         rotation_interval=np.radians(15),
                                                         gl_rotation_ax=np.array([0, 0, 1]),
                                                         toggle_debug=False)
        grasp_info_list.extend(grasp_info_list_tmp)
print("Number of grasps is", len(grasp_info_list))
dump_json(grasp_info_list, tube.grasps_path)

plot_node = [None]
counter = [0]


def update(grasp_info_list, counter, gripper_s, task):
    if base.inputmgr.keymap["space"]:
        base.inputmgr.keymap["space"] = False
        if counter[0] >= len(grasp_info_list) - 1:
            counter[0] = 0
            return task.again
        else:
            counter[0] += 1
            if plot_node[0] is not None:
                plot_node[0].detach()
            plot_node[0] = ModelCollection()
            jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info_list[counter[0]]
            gic = gripper_s.copy()
            gic.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
            gic.gen_meshmodel().attach_to(plot_node[0])
            plot_node[0].attach_to(base)
    return task.again

taskMgr.doMethodLater(0.02, update, "update",
                      extraArgs=[grasp_info_list, counter, gripper_s],
                      appendTask=True)

base.run()
