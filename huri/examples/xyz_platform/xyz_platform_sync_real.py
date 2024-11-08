"""
This is the example to sync the pose of the real to the sim
"""

# import the necessary library
from huri.core.common_import import *
from robot_sim.xyz_platform.xyz_platform import XYZPlatform
from robot_con.xyz_platform.xyz_platform import XYZPlatformController

# create the virtual environment
base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

# generate the yumi robot
ym_rbt = ym.Yumi(enable_cc=True)
# generate the model of the yumi robot and attach to the environment
ym_rbt_mdl = ym_rbt.gen_meshmodel()
ym_rbt_mdl.attach_to(base)

# generate the xyz platform
xyz_platform = XYZPlatform()
xyz_config = xyz_platform.get_jnt_values(component_name="all")
xyz_platform.gen_meshmodel().attach_to(base)


# # generate the mesh model
# # TODO install the zero position sensor to the movable platform
xyz_con = XYZPlatformController()
# !!! currently, zero position is manually set up. Pay attention when setting the zero position
xyz_con.calibrate()
xyz_platform.gen_meshmodel().attach_to(base)
xyz_con.set_x(xyz_config[0])
xyz_con.set_y(xyz_config[1])
xyz_con.set_z(xyz_config[2])

base.run()
