"""
Example to manually define the grasp poses
"""
from huri.core.common_import import *
import robot_sim.end_effectors.gripper.yumi_gripper.yumi_gripper as yg
from grasping.annotation.utils import define_grasp_with_rotation

# create the virtual environment
base = wd.World(cam_pos=[.5, .5, .3], lookat_pos=[0, 0, 0])
gripper_s = yg.YumiGripper(enable_cc=True)

# Generate collision model and attach to the virtual environment
objpath = "../../models/tubebig.stl"
objcm = cm.CollisionModel(objpath)
objcm.attach_to(base)
objcm.show_localframe()

# define the grasp poses
grasp_info_list = define_grasp_with_rotation(gripper_s,
                                             objcm,
                                             gl_jaw_center_pos=np.array([0, 0, .09]),
                                             gl_jaw_center_z=
                                             np.dot(rm.rotmat_from_axangle(np.array([1, 0, 0]),
                                                                           np.radians(-30)),
                                                    rm.unit_vector(np.array([0, 1, 0]))),
                                             gl_jaw_center_y=
                                             np.dot(rm.rotmat_from_axangle(np.array([1, 0, 0]),
                                                                           np.radians(-30)),
                                                    np.array([0, 0, 1])),
                                             jaw_width=.018,
                                             rotation_interval=np.radians(60),
                                             gl_rotation_ax=np.array([0, 0, 1]),
                                             toggle_debug=True)

print("Number of grasps is", len(grasp_info_list))

# save the grasp_info_list
fs.dump_pickle(grasp_info_list, "grasps.pkl")

# show all the grasp poses
for grasp_info in grasp_info_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    gic = gripper_s.copy()
    gic.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
    gic.gen_meshmodel().attach_to(base)

base.run()
