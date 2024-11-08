import numpy as np
import basis.robot_math as rm
from huri.test.iksolver.process_ik_offline import voxelize


def solve_ik(offline_data, rbt, tgt_pos, tgt_rot,
             seed_jnt_values=None,
             armname="rgt_arm"):
    # load offline data
    data, pos_kdt, ang_kdt = offline_data
    # get the index of the neareast point in the offline data
    pos_dist, pos_ind = pos_kdt.query(tgt_pos, k=1)
    ang_dist, ang_ind = ang_kdt.query(rm.rotmat_to_euler(tgt_rot), k=1)
    # get the nearest point's ik info
    info = data[armname][(pos_ind.item(), ang_ind.item())]
    iks = info["iks"]
    iks_mpa = info["iks_mpa"]
    # solve ik
    if len(iks) > 0:
        if seed_jnt_values is None:
            init_ik = iks[np.argmax(iks_mpa)]
        else:
            init_ik = iks[np.argmin(np.linalg.norm(iks - np.array(seed_jnt_values), axis=0))]
        r_ik = rbt.ik(armname,
                      tgt_pos=tgt_pos,
                      tgt_rotmat=tgt_rot,
                      seed_jnt_values=init_ik,
                      local_minima="end")
        return r_ik
    else:
        return None


global_time = [0]


def solve_ik2(offline_data: dict, rbt, tgt_pos, tgt_rot,
              seed_jnt_values=None,
              armname="rgt_arm"):
    offline_data_arm = offline_data.get(armname, None)
    if offline_data_arm is not None:
        # load offline data
        uniq_pm_f_KDtree = offline_data_arm["KDTree"]
        voxel_size_pos, voxel_size_rot  = offline_data_arm["voxel_size"]
        group = offline_data_arm["group"]
        jnt_samples = offline_data_arm["jnt_samples"]
        manipuability = offline_data_arm["manipuability"]
        # get the index of the neareast point in the offline data
        tgt_vec = voxelize(np.concatenate((tgt_pos, tgt_rot[:3, 2])).reshape(1,-1), voxel_size_pos, voxel_size_rot)
        # a = time.time()
        dist, vec_id = uniq_pm_f_KDtree.query(tgt_vec, k=1)
        if dist.item()>5:
            return None
        # print(tgt_vec)
        # print("distance", dist.item())
        # print("sovle_ik2 KDTREE time", time.time() - a)
        # print(group[vec_id.item()])
        # print("dist",dist)
        iks = jnt_samples[group[vec_id.item()]]
        iks_mpa = manipuability[group[vec_id.item()]]

        if len(iks) > 0:
            if seed_jnt_values is None:
                # print(iks_mpa)
                init_ik = iks[np.argmax(iks_mpa)]
            else:
                init_ik = iks[np.argmin(np.linalg.norm(iks - np.array(seed_jnt_values), axis=1))]
            rbt.fk(armname, init_ik)
            _, rot = rbt.get_gl_tcp(armname)
            init_ik[-1] = -rm.angle_between_vectors(rot[:3,0],tgt_rot[:3,0])
            # if len(iks) >= 3:
            #     for iik_id, iik in enumerate(iks):
            #         if iik_id == np.argmax(iks_mpa):
            #             continue
            #         rbt.fk(armname, iik)
            #         _, rot = rbt.get_gl_tcp(armname)
            #         iik[-1] = -rm.angle_between_vectors(rot[:3, 0], tgt_rot[:3, 0])
            #         rbt.fk(armname, iik)
            #         mesh = rbt.gen_meshmodel()
            #         mesh2 = rbt.gen_meshmodel2(rgba=[0,0,1,.2])
            #         # for oobj in mesh2.cm_list:
            #         #     oobj._objpdnp.setAlphaScale(.9)
            #         mesh.attach_to(base)
            #         mesh2.attach_to(base)
            #         gm.gen_frame(tgt_pos, tgt_rot, length=.15, thickness=.01).attach_to(base)
            #     rbt.fk(armname, init_ik)
            #     mesh = rbt.gen_meshmodel()
            #     mesh2 = rbt.gen_meshmodel2(rgba=[1, 0, 0, .8])
            #     # for oobj in mesh2.cm_list:
            #     #     oobj._objpdnp.setAlphaScale(.9)
            #     mesh.attach_to(base)
            #     mesh2.attach_to(base)
            #     gm.gen_frame(tgt_pos, tgt_rot, length=.15, thickness=.01).attach_to(base)
            #     base.run()
            # yumi_s.fk(armname, init_ik)
            # yumi_s.gen_meshmodel(rgba=[0, 0, 1, 1]).attach_to(base)
            a = time.time()
            r_ik = rbt.ik(armname,
                          tgt_pos=tgt_pos,
                          tgt_rotmat=tgt_rot,
                          seed_jnt_values=init_ik,
                          local_minima="end",
                          max_niter=50)
            global_time[0] = global_time[0] + time.time() - a
            # print("r_ik", r_ik)
            # yumi_s.fk(armname, r_ik)
            # yumi_s.gen_meshmodel().attach_to(base)
            # base.run()
            # print("sovle_ik2 IK time", time.time() - a)
            return r_ik
        else:
            return None
    else:
        return None


if __name__ == "__main__":
    from huri.core.common_import import *
    import time
    from ik_offline_utils import random_sample_range

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    yumi_s = ym.Yumi(enable_cc=False)
    a = time.time()
    offline_db = fs.load_pickle("ik_offline_db")
    print("load data time", time.time() - a)
    print("T")
    pos2 = np.array([.30 + 0.03, 0.13, .23])
    rot2 = np.dot(
        rm.rotmat_from_axangle([1, 0, 0], np.radians(15)),
        np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).T
    )

    # Workspace range
    x_range = [0.15, .65]
    y_range = [-0.4, 0.4]
    z_range = [0, 0.4]
    x_angles = [np.radians(-180), np.radians(180)]
    y_angles = [np.radians(-180), np.radians(180)]
    z_angles = [np.radians(-180), np.radians(180)]

    ik2_t = 0
    ik2_total = 0
    vanilla_t = 0
    vanilla_total = 0
    solved_ik2_num = 0
    solved_vik_num = 0
    for i in range(200):
        print("===================================")
        pos2 = np.array([random_sample_range(*x_range), random_sample_range(*y_range), random_sample_range(*z_range)])

        rot2 = np.dot(
            rm.rotmat_from_axangle([1, 0, 0], random_sample_range(*x_range)).dot(
                rm.rotmat_from_axangle([0, 1, 0], random_sample_range(*y_angles))).dot(
                rm.rotmat_from_axangle([0, 0, 1], random_sample_range(*z_angles))),
            np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).T)
        a = time.time()
        result_ik2 = solve_ik2(offline_db, yumi_s, pos2, rot2)
        ik2_t = time.time() - a

        a = time.time()
        result_vik = yumi_s.ik("rgt_arm", pos2, rot2, local_minima="end", max_niter=100)
        vanilla_t = time.time() - a

        if result_ik2 is not None:
            solved_ik2_num += 1
        if result_vik is not None:
            solved_vik_num += 1
        # if result_vik is not None and result_ik2 is not None:
        #     yumi_s.fk("rgt_arm",result_ik2)
        #     yumi_s.gen_meshmodel(rgba=[0,0,1,.7]).attach_to(base)
        #     print("Manipulability IK2", yumi_s.manipulability("rgt_arm"))
        #     yumi_s.fk("rgt_arm",result_vik)
        #     yumi_s.gen_meshmodel().attach_to(base)
        #     print("Manipulability Vanilla IK", yumi_s.manipulability("rgt_arm"))

            # base.run()


        ik2_total += ik2_t
        vanilla_total += vanilla_t

        print("sovle_ik2 time", ik2_t)
        print("vanilla time", vanilla_t)
        print("solve ik2 saved time", vanilla_t-ik2_t)
        print("===================================")
        # base.run()
    print(f"Solving IK2 -> IK time {global_time} | Solving IK2: {ik2_total}")
    print(f"Vanilla IK total time {vanilla_total}")
    print(f"IK2 solved num: {solved_ik2_num}, Vanilla IK solved num:{solved_vik_num}")
