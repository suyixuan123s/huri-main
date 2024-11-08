"""
This is an example to
1. solve inverse kinematics (IK)
2. check the self-collision of the simulation robot
(Run the 4_define_grasp.py First)
"""
import numpy as np

from huri.core.common_import import *

# create the virtual environment
base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

# generate the yumi robot
ym_rbt = ym.Yumi(enable_cc=True)
component_name = "rgt_arm"

# generate the tube to be grasped
obj_mdl = cm.CollisionModel(initor="../../models/tubebig.stl")
obj_mdl.set_pos(np.array([0.36, -.07, 0.1]))
obj_mdl.set_rgba(np.array([0, 0, 0, 1]))
obj_mdl.attach_to(base)

# load the grasp poses for the tube
grasps_list_info = fs.load_pickle("grasps.pkl")

for ind, (jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat) in enumerate(grasps_list_info):
    print(f"--------------------- grasp pose index: {ind} ---------------------------")
    obj_pose = obj_mdl.get_homomat()
    grasp_pose = rm.homomat_from_posrot(jaw_center_pos, jaw_center_rotmat)
    print(f"the homogenous matrix of the grasp pose is: {grasp_pose}")
    rbt_ee_pose = np.dot(obj_pose, grasp_pose)
    ik_sol = ym_rbt.ik(component_name, tgt_pos=rbt_ee_pose[:3, 3], tgt_rotmat=rbt_ee_pose[:3, :3])
    if ik_sol is not None:  # check IK-feasible
        ym_rbt.fk(component_name, ik_sol)
        if ym_rbt.is_collided():  # check if is self-collided
            print("The robot is self-collided")
            continue
        else:
            # illustrate the ik solution for grasping the tube
            ym_rbt.rgt_hnd.jaw_to(jaw_width)
            ym_rbt.gen_meshmodel().attach_to(base)
            base.run()
exit(-1)
