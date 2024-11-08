""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230910osaka

"""
from tqdm import tqdm
from constants import GRASP_ROT, Z_ROT_DEG, np
from utils import yumi_solve_ik, YumiController, examine_rbt_con_fk, generate_slot_centers, Mm, \
    yumi_gen_motion_slot_centers
from huri.core.common_import import ym, np, fs, wd, gm, rm

if __name__ == '__main__':

    GRASP_ROT_R = np.dot(rm.rotmat_from_axangle(np.array([0, 0, 1]),
                                                np.radians(Z_ROT_DEG)), GRASP_ROT)

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    rbt_sim = ym.Yumi()
    rbt_con = YumiController()
    examine_rbt_con_fk(rbt_con)
    component_name = 'rgt_arm'

    # pos = np.array([0.4, -0.1, 0.18])
    # ik_last = np.array([0.63809237, -0.4569272, -0.79639374, 0.80913464, -1.63397725,
    #                     0.89151418, -0.69202305])
    # ik = yumi_solve_ik(rbt_sim, rbt_con,
    #                    pos=pos,
    #                    rot=GRASP_ROT_R,
    #                    component_name=component_name,
    #                    seed_jnt_values=ik_last,
    #                    toggle_collision_detection=True,
    #                    toggle_visual=True)
    # if ik is None:
    #     ik = yumi_solve_ik(rbt_sim, rbt_con,
    #                        pos=pos,
    #                        rot=np.dot(np.array([[-1, 0, 0],
    #                                             [0, -1, 0],
    #                                             [0, 0, 1]]), GRASP_ROT),
    #                        component_name=component_name,
    #                        seed_jnt_values=ik_last,
    #                        toggle_collision_detection=True,
    #                        toggle_visual=False)
    # if ik is None:
    #     print(ik)

    ik = np.array([0.66235245, -0.43511058, -1.16692714, 0.50474922, -1.5393804,
                   0.55693456, -1.40778257])

    goal_pos = np.array([0.18378317, -0.81890849, -0.6435029, 0.48904126, -1.12975162,
                         0.89936816, -1.99857653])

    # (array([ 0.43591002, -0.19292001,  0.12067001], dtype=float32)
    p, r = rbt_con.fk(component_name, ik)
    jnt_value = rbt_sim.ik(component_name, p, r, np.array([0.99291781, -1.52541777, -1.52925749, 0.3122394, -0.33946654,
                             1.15313904, 0.7162831267948966]))
    _, _, conf, ext_axis = rbt_con.fk(component_name, jnt_value, return_conf=True)
    ik2 = rbt_con.ik(component_name, p, r, conf, ext_axis)

    # rbt_sim.fk(component_name, ik)
    # rbt_sim.gen_meshmodel(rgba=[0, 0, 1, 1]).attach_to(base)
    # rbt_sim.fk(component_name, ik2)
    # rbt_sim.gen_meshmodel().attach_to(base)
    # base.run()

    slot_centers = generate_slot_centers(rack_shape=(5, 10), center_dist_x=Mm(22), center_dist_y=Mm(20), offset_x=0,
                                         offset_y=0, offset_z=-.0, )

    iks, iks_approach, iks_departure = yumi_gen_motion_slot_centers(rbt_sim=rbt_sim,
                                                                    rbt_con=rbt_con,
                                                                    slot_centers=slot_centers,
                                                                    component_name=component_name,
                                                                    init_joint_val=ik2,
                                                                    toggle_visual=True,
                                                                    approach_dist=Mm(70),
                                                                    departure_dist=Mm(130), )

    fs.dump_pickle([iks, iks_approach, iks_departure], "iks.pkl", reminder=False)
    base.run()
    exit(0)
    row, column = iks.shape[:2]
    rbt_con.close_gripper(component_name=component_name)
    v = -1
    v2 = 500
    for i in range(row):
        for j in range(column):
            if iks_approach[i, j] is not None and len(iks_approach[i, j]) > 0:
                rbt_con.set_gripper_width(component_name, 0.018)
                rbt_con.move_jnts(component_name=component_name, jnt_vals=iks_approach[i, j][0], speed_n=v)
                rbt_con.move_jntspace_path(component_name, iks_approach[i, j], speed_n=v2)
                # grasp tube
                rbt_con.open_gripper(component_name=component_name)
                rbt_con.move_jntspace_path(component_name, iks_departure[i, j], speed_n=v)
                rbt_con.move_jnts(component_name=component_name, jnt_vals=goal_pos, speed_n=v)
                rbt_con.set_gripper_width(component_name, 0.018)
