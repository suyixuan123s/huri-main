import numpy as np

from huri.core.common_import import *
from huri.test.iksolver.ik_offline_utils import save_workspacesamples, sample_workspace
from sklearn.neighbors import KDTree

base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
yumi_s = ym.Yumi(enable_cc=True)

yumi_s.gen_meshmodel().attach_to(base)

pos_sample, ang_sample = sample_workspace(pos_space_range=[[0.15, .65], [-0.4, 0.4], [0, 0.4]],
                                          pos_sample_dense=.015,
                                          rot_sample_dense=PI / 6)

print(f"total number of samples is {len(pos_sample) * len(ang_sample)}")
# Show
gm.gen_pointcloud(pos_sample).attach_to(base)
# Save
save_workspacesamples(pos_sample, ang_sample, slice_num=5)
# generate KDTree
pos_KDtree = KDTree(pos_sample)
ang_KDtree = KDTree(ang_sample)
fs.dump_pickle([pos_KDtree, ang_KDtree],"KD_tree_pos_ang.pkl")
base.run()
