import numpy as np
from numpy import array
from huri.core.common_import import wd, fs, gm, rm
from utils import init_env_rbt, np
from robot_sim.robots.xarm_lite6_wrs.xarm_lite6_wrs2 import XArmLite6WRSGripper2

base = wd.World(cam_pos=(1, -0.6, 0.7), lookat_pos=[.2, 0, .1], w=1920,
                h=1080, )

rbt = XArmLite6WRSGripper2()
rbt.fk("arm", np.array([0.27132, - 0.12273, 0.78933, - 0.08392, 0.84483, 3.8086]))
_, p = rbt.get_gl_tcp("arm")
# print(p)
r1 = np.array([-179.3, 5.3, 160.4])



r2 = np.array([29.8, 172.3, -2.4])
rad_r2 = np.deg2rad(r2)
axis= rm.unit_vector(rad_r2)
magnitude = (rad_r2/axis)





rot = rm.rotmat_from_axangle(axis, magnitude)
print(axis, magnitude)
print(rot,rm.rotmat_from_euler(*np.deg2rad(r1)))


# gm.gen_frame(rotmat=rm.rotmat_from_euler(*np.deg2rad(r1))).attach_to(base)
#
# gm.gen_mycframe(rotmat=rm.rotmat_from_euler(*np.deg2rad(r2))).attach_to(base)
# base.run()
