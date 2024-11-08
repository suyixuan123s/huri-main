import time

import numpy as np

import huri.core.file_sys as fs
from huri.core.common_import import *
from huri.test.iksolver.ik_offline_utils import load_sample

dt, kd_t, kd_a = fs.load_pickle("result_iks")
print(dt)
# # pos_sample, ang_sample = fs.load_pickle("space_samples.pkl")
#
# pos_sample, ang_sample = sample_space(pos_space_range=[[0.12, .5], [-0.1, 0.1], [0, 0.3]],
#                                       pos_sample_dense=.25,
#                                       rot_sample_dense=PI)
#
# base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
# yumi_s = ym.Yumi(enable_cc=True)
#
#
# gm.gen_pointcloud(pos_sample).attach_to(base)
#
# info = r[66]
#
# v = info["vector"]
# iks = info["iks"]
# iks_mpa = info["iks_mpa"]
# id = np.argmin(iks_mpa)
# print(len(iks))
# iks = [iks[id]]
# for ik in iks:
#     gen_mesh_model_at_jnts(yumi_s, ik, armname="rgt_arm").attach_to(base)

# base.run()
# from sklearn.neighbors import KDTree
# pos_sample, ang_sample = load_sample(interval=(0,3))
# print(f"total number of samples is {len(pos_sample) * len(ang_sample)}") #38683008
#
#
# print("start")
# tree = KDTree(pos_sample)
# tree2 = KDTree(ang_sample)
# print("end")
# s= time.time()
# dist, ind = tree.query(np.array([[0.4,0.3,0.2]]), k=1)
# print(ind.item())
# dist2,ind2 = tree2.query(np.array([[.2,.3,.5]]), k=1)
# print(dist2, ind2)
# e = time.time()
# print(f"time-consuming is {e-s}")
