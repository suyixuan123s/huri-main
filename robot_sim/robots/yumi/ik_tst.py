import numpy as np

from huri.core.common_import import *
import time
import visualization.panda.world as wd
import modeling.geometric_model as gm
import basis
from robot_sim.robots.yumi.trac_ik.trac_ik import Trac_IK

base = wd.World(cam_pos=[3, 1, 1], lookat_pos=[0, 0, 0.5])
gm.gen_frame().attach_to(base)
yumi_instance = ym.Yumi(enable_cc=True)

# yumi_meshmodel = yumi_instance.gen_meshmodel(toggle_tcpcs=True)
# yumi_instance.fk("rgt_hnd", np.zeros(7))
# yumi_instance.fk("lft_hnd", np.zeros(7))
# yumi_meshmodel.attach_to(base)
# yumi_instance.show_cdprimit()
# base.run()
# ik test

component_name = 'rgt_arm'
tgt_pos = np.array([.3, -.4, .1])
tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], np.pi / 3)
gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
print(repr(yumi_instance.rgt_arm.homeconf))
import os
os.environ['NUMEXPR_MAX_THREADS'] = str(8)  # NumExpr max threads
tic = time.time()
for i in range(5000):
# jnt_values = yumi_instance.ik(component_name, tgt_pos, tgt_rotmat, toggle_debug=False, max_niter=100)
    yumi_instance.fk("rgt_arm",jnt_values=yumi_instance.rgt_arm.homeconf)
    yumi_instance.fk("rgt_arm", np.array([-0.34906585, -1.57079633, -2.0943951, 0.52359878, 0.,
                                       0.6981317, 0.]))
toc = time.time()
print(toc - tic)

exit(0)

iksolver_rgt = Trac_IK("yumi_body", "yumi_link_7_r",
                       "D:\\chen\\huri_shared\\robot_sim\\robots\yumi\\trac_ik\\yumi.urdf")
seed_jnt = yumi_instance.rgt_arm.homeconf
tic = time.time()

tgt_pos = tgt_pos - np.dot(tgt_rotmat, yumi_instance.rgt_hnd.jaw_center_pos)
jnt_values_trac_ik = iksolver_rgt.solve(tgt_pos, tgt_rotmat, seed_jnt_values=seed_jnt)
toc = time.time()
print(toc - tic)

yumi_instance.fk(component_name, jnt_values)
yumi_instance.gen_meshmodel().attach_to(base)
yumi_instance.fk(component_name, jnt_values_trac_ik)
yumi_instance.gen_meshmodel(rgba=[1, 0, 0, 1]).attach_to(base)
# ik_pos, ik_rot = yumi_instance.get_gl_tcp(component_name)
# print(f"pos value is: {ik_pos}, diff is {(ik_pos-tgt_pos)*1000} mm")

base.run()
