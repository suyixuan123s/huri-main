""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230910osaka

"""
from tqdm import tqdm
from constants import GRASP_ROT, Z_ROT_DEG, np
from utils import yumi_solve_ik, YumiController, examine_rbt_con_fk
from huri.core.common_import import ym, np, fs, wd, gm, rm, cm

if __name__ == '__main__':

    GRASP_ROT_R = np.dot(rm.rotmat_from_axangle(np.array([0, 0, 1]),
                                                np.radians(Z_ROT_DEG)), GRASP_ROT)

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    rbt_sim = ym.Yumi()
    rbt_con = YumiController()
    examine_rbt_con_fk(rbt_con)
    component_name = 'rgt_arm'

    space_x_range = np.arange(start=.3, stop=.6, step=.03)
    space_y_range = np.arange(start=-.1, stop=.13, step=.03)
    space_z_range = np.arange(start=0.15, stop=.25, step=.03)
    # make combinations of x, y, z
    space = np.array(np.meshgrid(space_x_range, space_y_range, space_z_range)).T.reshape(-1, 3)
    ik_last = None
    # Calculate the IK solution for each point in the space
    for pos in tqdm(space):
        ik = yumi_solve_ik(rbt_sim, rbt_con,
                           pos=pos,
                           rot=GRASP_ROT_R,
                           component_name=component_name,
                           seed_jnt_values=ik_last,
                           toggle_collision_detection=True,
                           toggle_visual=True)
        if ik is None:
            ik = yumi_solve_ik(rbt_sim, rbt_con,
                               pos=pos,
                               rot=np.dot(np.array([[-1, 0, 0],
                                                    [0, -1, 0],
                                                    [0, 0, 1]]), GRASP_ROT),
                               component_name=component_name,
                               seed_jnt_values=ik_last,
                               toggle_collision_detection=True,
                               toggle_visual=True)
        if ik is None:
            color = [1, 0, 0, 1]
        else:
            color = [0, 1, 0, 1]
        if ik is not None:
            # rbt_con.move_jnts(component_name=component_name, jnt_vals=ik, speed_n=20)
            pass
        ik_last = ik
        gm.gen_sphere(pos=pos, rgba=color, radius=.005).attach_to(base)
    base.run()
