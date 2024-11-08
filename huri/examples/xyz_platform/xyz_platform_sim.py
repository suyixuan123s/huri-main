"""
This is an example to generate the movable platform
"""

# import the necessary library
from huri.core.common_import import *
from robot_sim.xyz_platform.xyz_platform import XYZPlatform

# create the virtual environment
base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

# generate the yumi robot
ym_rbt = ym.Yumi(enable_cc=True)
# generate the model of the yumi robot and attach to the environment
ym_rbt_mdl = ym_rbt.gen_meshmodel()
ym_rbt_mdl.attach_to(base)

# generate the xyz platform
xyz_platform = XYZPlatform()
xyz_platform.fk(component_name="all", motion_val=np.zeros(3))
# generate the mesh model
xyz_platform.gen_meshmodel().attach_to(base)

base.run()
