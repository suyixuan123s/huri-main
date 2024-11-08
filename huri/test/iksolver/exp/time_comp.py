import time

import numpy as np

from huri.core.common_import import *

PI = np.pi


def gen_mesh_model_at_jnts(rbt: ym.Yumi, jnts: np.ndarray, armname="rgt_arm"):
    if jnts is None:
        return None
    jnts_bk = rbt.get_jnt_values(armname)
    rbt.fk(armname, jnts)
    mesh_model = rbt.gen_meshmodel()
    rbt.fk(armname, jnts_bk)
    return mesh_model


# base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
yumi_s = ym.Yumi(enable_cc=True)
#


pos = np.array([.30, 0.13, .23])
rot = np.dot(
    rm.rotmat_from_axangle([1, 0, 0], np.radians(0)),
    np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).T
)

work_arm = "rgt_arm"
# yumi_s.gen_meshmodel().attach_to(base)
# base.run()
def test():
    jnts = yumi_s.ik(work_arm,tgt_pos=pos,tgt_rotmat=rot)
    # print(jnts)
import timeit
loop = 100
result = timeit.timeit('test()', globals=globals(),number=loop)
print(result / loop)
#     mesh_model = gen_mesh_model_at_jnts(yumi_s,jnts)
#     if mesh_model is not None:
#         mesh_model.attach_to(base)
#         yumi_s.fk(work_arm,jnts)
#         print(yumi_s.is_collided())


# base.run()
