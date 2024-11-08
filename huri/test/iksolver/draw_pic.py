import numpy as np

from huri.core.common_import import *
from ik_offline_utils import random_sample_range

base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
yumi_s = ym.Yumi(enable_cc=False)

x_range = [0.15, .65]
y_range = [-0.4, 0.4]
z_range = [0, 0.4]
x_angles = [np.radians(-180), np.radians(180)]
y_angles = [np.radians(-180), np.radians(180)]
z_angles = [np.radians(-180), np.radians(180)]

# pos = np.array([random_sample_range(*x_range), random_sample_range(*y_range), random_sample_range(*z_range)])
pos = np.array([0.23906668, -0.26129227, 0.18935586])
# rot = np.dot(
#     rm.rotmat_from_axangle([1, 0, 0], random_sample_range(*x_range)).dot(
#         rm.rotmat_from_axangle([0, 1, 0], random_sample_range(*y_angles))).dot(
#         rm.rotmat_from_axangle([0, 0, 1], random_sample_range(*z_angles))),
#     np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).T)
rot = np.array([[0.99958869, 0.02840775, -0.00393022],
                [-0.01382989, 0.59755175, 0.80171107],
                [0.02512332, -0.80132697, 0.59769885]])

gm.gen_frame(pos, rot, length=.15, thickness=.01).attach_to(base)
print(pos, rot)
base.rbt = yumi_s
base.rbt_draw_list = []
yumi_s.gen_meshmodel().attach_to(base)
base.run()
seedjnt = np.array([ 0.78991009, -1.93105111, -1.42597166 , 0.02642755, -0.18327923  ,2.09140094,
 -0.71054459])
l = yumi_s.ik("rgt_arm", pos, rot, local_minima="end", max_niter=100,
              seed_jnt_values= seedjnt)

l = len(base.rbt_draw_list)
sq = np.logspace(0, 1, num=l) / 10
# sq = np.flip(1 - sq) + .1
for id, aplpha in enumerate(sq):
    print(aplpha)
    for oobj in base.rbt_draw_list[id].cm_list:
        oobj._objpdnp.setAlphaScale(aplpha)

    base.rbt_draw_list[id].attach_to(base)

base.run()
# print("l is", l)
# yumi_s.fk("rgt_arm", l)
# yumi_s.gen_meshmodel().attach_to(base)
base.run()
