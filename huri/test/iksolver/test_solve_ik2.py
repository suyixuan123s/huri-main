import time

import numpy as np

from huri.core.common_import import *
from huri.test.iksolver.ik_database import sample_space, sample_ik_at_pose

PI = np.pi


def gen_mesh_model_at_jnts(rbt: ym.Yumi, jnts: np.ndarray, armname="rgt_arm"):
    if jnts is None:
        return None
    jnts_bk = rbt.get_jnt_values(armname)
    rbt.fk(armname, jnts)
    mesh_model = rbt.gen_meshmodel()
    rbt.fk(armname, jnts_bk)
    return mesh_model


base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
yumi_s = ym.Yumi(enable_cc=True)
#


pos = np.array([.30, 0.13, .23])
rot = np.dot(
    rm.rotmat_from_axangle([1, 0, 0], np.radians(0)),
    np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).T
)

work_arm = "rgt_arm"
# for i in range(50):
#     jnts = yumi_s.ik(work_arm,tgt_pos=pos,tgt_rotmat=rot,seed_jnt_values=yumi_s.rand_conf(work_arm))
#     mesh_model = gen_mesh_model_at_jnts(yumi_s,jnts)
#     if mesh_model is not None:
#         mesh_model.attach_to(base)
#         yumi_s.fk(work_arm,jnts)
#         print(yumi_s.is_collided())

ang = rm.rotmat_to_euler(rot)

pos_ind_rd = np.random.rand(900, 6)
pos_ind_rd[133] = np.concatenate((pos, ang))

iks, iks_mbp = sample_ik_at_pose(yumi_s, pos, rot, num_sample=30)
print(iks)
print(iks_mbp)
for i in iks:
    mesh_model = gen_mesh_model_at_jnts(yumi_s, i)
    mesh_model.attach_to(base)

rbt = yumi_s
armname = "rgt_arm"
pos2 = np.array([.30 + 0.03, 0.13 + 0.02, .23 + +0.02])
rot2 = np.dot(
    rm.rotmat_from_axangle([1, 0, 0], np.radians(15)),
    np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).T
)

start = time.time()
ik_tmp = rbt.ik(armname, tgt_pos=pos, tgt_rotmat=rot, seed_jnt_values=rbt.rand_conf(armname), local_minima="end")
end = time.time()
print("IK solver:", end - start)
init_iks = iks[np.argmax(iks_mbp)]
start = time.time()
ang2 = rm.rotmat_to_euler(rot2)
id = np.concatenate((pos2, ang2))
print(np.argmin(np.sum((pos_ind_rd-id),axis=1)))
ik_tmp = rbt.ik(armname, tgt_pos=pos, tgt_rotmat=rot, seed_jnt_values=init_iks, local_minima="end")
end = time.time()
print("IK solver2:", end - start)

# mesh_model.attach_to(base)


base.run()
