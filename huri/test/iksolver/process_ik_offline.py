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


def voxelize(pm, voxel_size_pos=0.02, voxel_size_rot=0.1):
    return np.hstack((np.round(np.array(pm[:, :3] / voxel_size_pos)).astype(int),
                      np.round(np.array(pm[:, 3:] / voxel_size_rot)).astype(int)))


def group_voxel(pm, voxel_size_pos=0.02, voxel_size_rot=0.1):
    # voxel size
    pm_q = voxelize(pm, voxel_size_pos, voxel_size_rot)

    uniq_pm_q, ind_inverse = np.unique(pm_q, axis=0, return_inverse=True)
    df = pd.DataFrame({'uniq_id2pm_id': ind_inverse})
    gg = df.groupby(by=df.uniq_id2pm_id)
    return uniq_pm_q, gg.groups


if __name__ == "__main__":
    # Hyperparameters
    voxel_size_pos = 0.01
    voxel_size_rot = 0.05
    # Workspace range
    x_range = [0.15, .65]
    y_range = [-0.4, 0.4]
    z_range = [0, 0.4]
    #
    armname = "rgt_arm"
    save_name = "ik_offline_db"

    pm, jnt_samples, manipuability = load_data()

    feasible_indices = extract_feasible_workspace_indice(pm[:, :3], x_range, y_range, z_range)

    pm_f = pm[feasible_indices]
    jnt_samples_f = jnt_samples[feasible_indices]
    manipuability_f = manipuability[feasible_indices]
    # [16  6 12  0 48 13]
    uniq_pm_f, group = group_voxel(pm_f, voxel_size_pos, voxel_size_rot)

    print(f"number of feasible: {pm_f.shape[0]}, number of unique: {uniq_pm_f.shape[0]}")

    uniq_pm_f_KDtree = KDTree(uniq_pm_f)

    print("KD tree generated finished")

    fs.dump_pickle({
        armname: {
            "KDTree": uniq_pm_f_KDtree,
            "voxel_size": [voxel_size_pos, voxel_size_rot],
            "group": group,
            "jnt_samples": np.c_[jnt_samples_f, np.zeros(jnt_samples_f.shape[0])],
            "manipuability": manipuability_f,
        }
    }, save_name)

    print("Finished Generate")
