from huri.core.common_import import *
import pandas as pd
from sklearn.neighbors import KDTree


def np_same_row(mat: np.ndarray, row: np.ndarray) -> np.ndarray:
    return np.where((mat == row).all(axis=1))[0]


def extract_feasible_workspace_indice(pm, x, y, z):
    return np.where((pm[:, 0] >= x[0]) & (pm[:, 0] <= x[1]) &
                    (pm[:, 1] >= y[0]) & (pm[:, 1] <= y[1]) &
                    (pm[:, 2] >= z[0]) & (pm[:, 2] <= z[1]))


def load_data(reachability_names=("reachability_db_0.pkl",
                                  "reachability_db_1.pkl",
                                  "reachability_db_2.pkl",
                                  "reachability_db_3.pkl"),
              jntspace_samples_names=("jntspace_samples_0.pkl",
                                      "jntspace_samples_1.pkl",
                                      "jntspace_samples_2.pkl",
                                      "jntspace_samples_3.pkl"),
              armname="rgt_arm"):
    assert len(reachability_names) == len(jntspace_samples_names)
    pm = None
    jnt_samples = None
    manipuability = None
    for fname_id in range(len(reachability_names)):
        database = fs.load_pickle(reachability_names[fname_id])
        database_j = fs.load_pickle(jntspace_samples_names[fname_id])
        pm_tmp = database[armname]["pose_mat"]
        mv_tmp = database[armname]["manipuability_vec"]
        jnt_samples_tmp = database_j
        if pm is None:
            pm = pm_tmp
        else:
            pm = np.vstack((pm, pm_tmp))
        if jnt_samples is None:
            jnt_samples = jnt_samples_tmp
        else:
            jnt_samples = np.vstack((jnt_samples, jnt_samples_tmp))
        if manipuability is None:
            manipuability = mv_tmp
        else:
            manipuability = np.vstack((manipuability, mv_tmp))
    return pm, jnt_samples, manipuability


def voxelize(pm, voxel_size=0.02):
    return np.round(np.array(pm / voxel_size)).astype(int)


def group_voxel(pm, voxel_size=0.02):
    # voxel size
    pm_q = voxelize(pm, voxel_size)

    uniq_pm_q, ind_inverse = np.unique(pm_q, axis=0, return_inverse=True)
    df = pd.DataFrame({'uniq_id2pm_id': ind_inverse})
    gg = df.groupby(by=df.uniq_id2pm_id)
    return uniq_pm_q, gg.groups


pm, jnt_samples, manipuability = load_data()

feasible_indices = extract_feasible_workspace_indice(pm[:, :3], [0.15, .65], [-0.4, 0.4], [0, 0.4])

pm_f = pm[feasible_indices]
jnt_samples_f = jnt_samples[feasible_indices]
manipuability = manipuability[feasible_indices]

voxel_size = 0.02
uniq_pm_f, group = group_voxel(pm_f, voxel_size)



print("Finished Generate")

# show animation
# base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
# gm.gen_pointcloud(pos_samples).attach_to(base)
# base.run()
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
