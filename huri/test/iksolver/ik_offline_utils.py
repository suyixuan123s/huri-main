import time
import basis.robot_math as rm
import numpy as np
import huri.core.file_sys as fs
from collections import ChainMap

PI = np.pi


def random_sample_range(a, b):
    return (b - a) * np.random.random_sample() + a


def sample_workspace(pos_space_range=([-.5, .5], [-.5, .5], [-.5, .5]),
                     rot_space_range=([-PI, PI], [-PI, PI], [-PI, PI]),
                     pos_sample_dense=(.1, .1, .1),
                     rot_sample_dense=(PI / 36, PI / 36, PI / 36)):
    if isinstance(pos_sample_dense, float):
        pos_sample_dense = (pos_sample_dense, pos_sample_dense, pos_sample_dense)

    if isinstance(rot_sample_dense, float):
        rot_sample_dense = (rot_sample_dense, rot_sample_dense, rot_sample_dense)

    pos_space_range[0][1] += pos_sample_dense[0]
    pos_space_range[1][1] += pos_sample_dense[1]
    pos_space_range[2][1] += pos_sample_dense[2]

    x = np.arange(*pos_space_range[0], pos_sample_dense[0])
    y = np.arange(*pos_space_range[1], pos_sample_dense[1])
    z = np.arange(*pos_space_range[2], pos_sample_dense[2])

    pos_sampled = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    pitch = np.arange(*rot_space_range[0], rot_sample_dense[0])
    roll = np.arange(*rot_space_range[1], rot_sample_dense[1])
    yaw = np.arange(*rot_space_range[2], rot_sample_dense[2])

    ang_sampled = np.array(np.meshgrid(roll, yaw, pitch)).T.reshape(-1, 3)

    return pos_sampled, ang_sampled


def sample_jnt_space(jnt_space_range, ang_sample_dense=.1):
    ndof = len(jnt_space_range)
    if isinstance(ang_sample_dense, float):
        ang_sample_dense = [ang_sample_dense for _ in range(len(jnt_space_range))]
    elif isinstance(ang_sample_dense, tuple) or isinstance(ang_sample_dense, list) or isinstance(ang_sample_dense,
                                                                                                 np.ndarray):
        if len(jnt_space_range) != len(ang_sample_dense):
            raise Exception(f"fThe length of jnt space range({len(jnt_space_range)})"
                            f" should be same as rot sample dense({len(ang_sample_dense)})")
    else:
        raise Exception("Rot sample dense must be tuple list or np.ndarray")

    sample_jnt_angle_list = []
    for jnt_id, jnt in enumerate(jnt_space_range):
        tmp_sample_ang = np.arange(*jnt, ang_sample_dense[jnt_id])
        sample_jnt_angle_list.append(tmp_sample_ang)
    print(f"Expected jnt space size: {np.prod([len(i) for i in sample_jnt_angle_list])}")
    jnt_sample = np.array(np.meshgrid(*sample_jnt_angle_list)).T.reshape(-1, ndof)
    return jnt_sample


def sample_ik_at_pose(rbt, pos, rot, num_sample=30, armname="rgt_arm"):
    ndof = rbt.manipulator_dict[armname].ndof
    jnts_bk = rbt.get_jnt_values(armname)
    iks_at_pose = np.zeros((num_sample, ndof))
    iks_manipulability = np.zeros((num_sample, 1))
    cnt = 0
    for i in range(num_sample):
        ik_tmp = rbt.ik(armname, tgt_pos=pos, tgt_rotmat=rot, seed_jnt_values=rbt.rand_conf(armname),
                        local_minima="end")
        if ik_tmp is not None:
            rbt.fk(armname, ik_tmp)
            # if not rbt.is_collided():
            if cnt < 1 or not np.all(np.linalg.norm(iks_at_pose[:cnt] - ik_tmp, axis=1) < .2):
                iks_at_pose[cnt] = ik_tmp
                iks_manipulability[cnt] = rbt.manipulability(armname)
                cnt += 1
            else:
                print("exact")
                print(np.linalg.norm(iks_at_pose[:cnt] - ik_tmp, axis=1))
    rbt.fk(armname, jnts_bk)
    return iks_at_pose[: cnt], iks_manipulability[:cnt]


def sample_ik_from_pose_set(rbt, pos_set, rot_set, num_sample=30, armname="rgt_arm", pos_start_id=0):
    total_num_cal = len(pos_set) * len(rot_set)
    cnt = 0
    result = {}
    for pos_id, pos in enumerate(pos_set):
        for rot_id, rot in enumerate(rot_set):
            if rot.shape != (3, 3):
                _rot = rm.rotmat_from_euler(*rot)
            else:
                _rot = rot
            s = time.time()
            iks, iks_mpa = sample_ik_at_pose(rbt, pos, _rot, num_sample=num_sample, armname=armname)
            total_t = time.time() - s
            tmp_r = {}
            tmp_r[(pos_start_id + pos_id, rot_id)] = {
                "vector": np.concatenate((pos, rot)),
                "iks": iks,
                "iks_mpa": iks_mpa}
            cnt += 1
            result.update(tmp_r)
            print(f"Start index is {pos_start_id}, "
                  f"Current progress: {cnt}/{total_num_cal}, "
                  f"prediction remaining time is {total_t * (total_num_cal - cnt) / 3600} hours")
    return result


def create_ik_db_ws(rbt,
                    pos_sample,
                    ang_sample,
                    num_ik_sample=30,
                    armname="rgt_arm",
                    past_slice_len=0,
                    num_worker=1):
    data_table = {}
    if num_worker > 1:
        # multiprocess
        import multiprocessing as mlp
        interval = int(len(pos_sample) / num_worker)
        interval_list = np.arange(0, len(pos_sample), interval)
        with mlp.Pool(processes=num_worker) as pool:
            multiple_results = [pool.apply_async(sample_ik_from_pose_set, args=(
                rbt,
                pos_sample[i:i + interval],
                ang_sample,
                num_ik_sample,
                armname,
                i + past_slice_len)) for i in interval_list]
            r = [res.get() for res in multiple_results]
        result_dict = dict(ChainMap(*r))
    else:
        result_dict = sample_ik_from_pose_set(rbt,
                                              pos_set=pos_sample,
                                              rot_set=ang_sample,
                                              num_sample=num_ik_sample,
                                              armname=armname)

    data_table[armname] = result_dict
    fs.dump_pickle(data_table, f"result_iks_{past_slice_len}")
    print("Generated Finished")


def sample_pose_at_jnt(rbt, jnt_angs, arb_jnt_id=None, arb_jnt_value=.0, armname="rgt_arm"):
    """
    arb_jnt_id: arbitrary joint id. Some joint does not contribute to tcp position, only contribute to rotation
    arb_jnt_value: default value for the arbitrary joint. a range or a value. If input is a range,
                   it will randomly sample a value from the range as the default value.
    """
    assert isinstance(arb_jnt_id, np.ndarray) or isinstance(arb_jnt_value, float)
    assert arb_jnt_id is None or isinstance(arb_jnt_id, int)
    ndof = rbt.manipulator_dict[armname].ndof
    if ndof > len(jnt_angs) and arb_jnt_id is not None:
        default_v = random_sample_range(*arb_jnt_value) if isinstance(arb_jnt_value, np.ndarray) else arb_jnt_value
        jnt_angs_ = np.insert(jnt_angs, arb_jnt_id - 1, default_v)
    else:
        raise Exception(f"robot's dof({ndof}) is larger than the input jnt angle dimension({len(jnt_angs)}) ")
    rbt.fk(component_name=armname, jnt_values=jnt_angs_)
    pos, rot = rbt.get_gl_tcp(manipulator_name=armname)
    mpab = rbt.manipulability(armname)
    return pos, rot, mpab


def sample_poses_from_jnt_set(rbt,
                              jnt_ang_set,
                              arb_jnt_id=None,
                              arb_jnt_value=.0,
                              u_axis=None,
                              armname="rgt_arm"):
    assert u_axis is None or u_axis in "xyz"
    jnt_ang_set_len = len(jnt_ang_set)
    cnt = 1
    pose_matrix = np.zeros((jnt_ang_set_len, 6))
    manipulability_vector = np.zeros((jnt_ang_set_len, 1))
    for jnt_angs_id, jnt_angs in enumerate(jnt_ang_set):
        s = time.time()
        pos, rot, mpab = sample_pose_at_jnt(rbt, jnt_angs, arb_jnt_id, arb_jnt_value, armname)
        if u_axis is None:
            r_v = rm.rotmat_to_euler(rot)
        else:
            if u_axis == "x":
                r_v = rot[:, 0]
            elif u_axis == "y":
                r_v = rot[:, 1]
            else:
                r_v = rot[:, 2]
        print(r_v)
        pose_matrix[jnt_angs_id] = np.concatenate((pos, r_v))
        manipulability_vector[jnt_angs_id] = mpab
        t = time.time() - s
        print(f"Current progress: {cnt}/{jnt_ang_set_len}, "
              f"prediction remaining time is {t * (jnt_ang_set_len - cnt) / 60:.2f} mins")
        cnt += 1
    print("Generated Finished")
    return pose_matrix, manipulability_vector


def create_ik_db_js(rbt,
                    jnt_ang_set,
                    arb_jnt_id=None,
                    arb_jnt_value=.0,
                    u_axis=None,
                    armname="rgt_arm",
                    save_fname="reachability_db",
                    num_worker=1):
    data_table = {}
    if num_worker > 1:
        # multiprocess
        import multiprocessing as mlp
        interval = int(len(jnt_ang_set) / num_worker)
        interval_list = np.arange(0, len(jnt_ang_set), interval)
        with mlp.Pool(processes=num_worker) as pool:
            multiple_results = [pool.apply_async(sample_poses_from_jnt_set, args=(
                rbt,
                jnt_ang_set[i:i + interval],
                arb_jnt_id,
                arb_jnt_value,
                u_axis,
                armname)) for i in interval_list]
            r = [res.get() for res in multiple_results]
        pose_mat = np.zeros((len(jnt_ang_set), 6))
        manipulability_vec = np.zeros((len(jnt_ang_set), 1))
        cnt = 0
        for pm, mv in r:
            pm_row = pm.shape[0]
            pose_mat[cnt:cnt + pm_row] = pm
            manipulability_vec[cnt:cnt + pm_row] = mv
            cnt += pm_row
    else:
        pose_mat, manipulability_vec = sample_poses_from_jnt_set(rbt,
                                                                 jnt_ang_set,
                                                                 arb_jnt_id,
                                                                 arb_jnt_value,
                                                                 u_axis,
                                                                 armname)

    data_table[armname] = {
        "pose_mat": pose_mat,
        "manipuability_vec": manipulability_vec
    }
    fs.dump_pickle(data_table, f"{save_fname}.pkl")
    print("Successfully Saved")


def load_sample(fname="space_samples", interval=None):
    assert interval is None or isinstance(interval, tuple) or isinstance(interval, list)
    pos_sample = None
    if interval is not None:
        for i in range(interval[0], interval[1] + 1):
            pos_sample_tmp, ang_sample = fs.load_pickle(f"{fname}_{i}.pkl")
            if pos_sample is None:
                pos_sample = pos_sample_tmp
            else:
                pos_sample = np.concatenate((pos_sample, pos_sample_tmp), axis=0)
    else:
        pos_sample, ang_sample = fs.load_pickle(f"{fname}.pkl")

    return pos_sample, ang_sample


def save_workspacesamples(pos_sample, ang_sample, slice_num=1, fname="space_samples"):
    if slice_num > 1:
        interval = np.linspace(0, len(pos_sample), slice_num + 1).astype(int)
        cnt = 0
        for i in range(0, len(interval) - 1):
            start_ind = interval[i]
            end_ind = interval[i + 1]
            fs.dump_pickle([pos_sample[start_ind:end_ind], ang_sample], f"{fname}_{cnt}.pkl")
            cnt += 1
    else:
        fs.dump_pickle([pos_sample, ang_sample], f"{fname}.pkl")
    print("Successfully Savead")


def save_jntspacesamples(ang_samples, slice_num=1, fname="jntspace_samples"):
    if slice_num > 1:
        interval = np.linspace(0, len(ang_samples), slice_num + 1).astype(int)
        cnt = 0
        for i in range(0, len(interval) - 1):
            start_ind = interval[i]
            end_ind = interval[i + 1]
            fs.dump_pickle(ang_samples[start_ind:end_ind], f"{fname}_{cnt}.pkl")
            cnt += 1
    else:
        fs.dump_pickle(ang_samples, f"{fname}.pkl")
    print("Successfully Savead")


def gen_mesh_model_at_jnts(rbt, jnts: np.ndarray, armname="rgt_arm"):
    jnts_bk = rbt.get_jnt_values(armname)
    rbt.fk(armname, jnts)
    mesh_model = rbt.gen_meshmodel()
    rbt.fk(armname, jnts_bk)
    return mesh_model


if __name__ == "__main__":
    from huri.core.common_import import *

    yumi_s = ym.Yumi(enable_cc=True)

    pos_sample, ang_sample = sample_workspace(pos_sample_dense=.5, rot_sample_dense=PI / 3)
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    for i in pos_sample:
        gm.gen_sphere(pos=i).attach_to(base)
    for ang in ang_sample:
        gm.gen_frame(pos=i, rotmat=rm.rotmat_from_euler(*ang)).attach_to(base)

    print(yumi_s.ik("rgt_arm", tgt_pos=pos_sample[0], tgt_rotmat=rm.rotmat_from_euler(*ang_sample[0])))

    base.run()
