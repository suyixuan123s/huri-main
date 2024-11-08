""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230911osaka

"""

if __name__ == '__main__':
    from tqdm import tqdm
    from constants import GRASP_ROT, Z_ROT_DEG, np
    from utils import yumi_solve_ik, YumiController, examine_rbt_con_fk
    from huri.core.common_import import ym, np, fs, wd, gm, rm, cm
    from huri.math.units import Mm

    if __name__ == '__main__':
        GRASP_ROT_R = np.dot(rm.rotmat_from_axangle(np.array([0, 0, 1]),
                                                    np.radians(Z_ROT_DEG)), GRASP_ROT)

        base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
        rbt_sim = ym.Yumi()
        rbt_sim.fk('rgt_arm', np.array([0.39426988, -0.48153634, -0.86044732, 0.57019907, -1.39975406,
                                        0.60789818, -1.50761541]))
        rbt_sim.gen_meshmodel().attach_to(base)

        space_x_range = np.arange(start=0.43591002, stop=.6, step=.03)
        space_y_range = np.arange(start=-0.19292001, stop=.13, step=.03)
        space_z_range = np.arange(start=0.12067001, stop=0.16, step=.03)

        space = np.array(np.meshgrid(space_x_range, space_y_range, space_z_range)).T.reshape(-1, 3)

        for pos in space:
            gm.gen_sphere(pos=pos, rgba=[0, 1, 0, 1], radius=.005).attach_to(base)
        collision_box = cm.CollisionModel(gm.gen_box([Mm(120), Mm(200), Mm(100)], rgba=[1, 0, 0, .1]))
        collision_box.set_pos(np.array([Mm(435.91002) + Mm(120) / 2, Mm(-192.92001) + Mm(200) / 2, Mm(120.67001)]))
        collision_box.attach_to(base)
        collision_box2 = cm.CollisionModel(gm.gen_box([Mm(200), Mm(350), Mm(100)], rgba=[1, 0, 0, .1]))
        collision_box2.set_pos(
            np.array([Mm(435.91002) + Mm(200) / 2, Mm(-192.92001) + Mm(350) / 2, Mm(120.67001) ]))
        collision_box2.attach_to(base)
        gm.gen_frame([Mm(435.91002), Mm(-192.92001), Mm(120.67001)]).attach_to(base)
        collision_box2.set_rgba([0, 1, 0, .4])
        print(rbt_sim.is_collided(obstacle_list=[collision_box2]))
        base.run()
