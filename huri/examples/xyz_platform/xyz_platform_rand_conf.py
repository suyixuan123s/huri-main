"""
This is an example to generate random configuration for the xyz platform
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
rand_conf = xyz_platform.rand_conf("all", region=np.array([
    [.05, .35],
    [.3, .45],
    [0, np.radians(350)],
]))
print(f"The x platform pos is {rand_conf[0]}\n"
      f"The y platform pos is {rand_conf[1]}\n"
      f"The z platform ang is {np.rad2deg(rand_conf[2])}")
xyz_platform.fk("all", rand_conf)
xyz_platform.gen_meshmodel().attach_to(base)

base.run()
