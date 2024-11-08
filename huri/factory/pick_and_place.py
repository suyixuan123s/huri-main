""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230911osaka

"""

if __name__ == '__main__':
    from huri.core.common_import import ym, np, fs, wd, gm, rm, cm

    [iks, iks_approach, iks_departure] = fs.load_pickle("iks.pkl")

    pick_approach = iks_approach
    pick_depatrure = iks_departure

    # iks_departure is an array with 4 axis, make axis 3 reverse order
    place_approach = iks_departure.copy()
    for i in range(place_approach.shape[0]):
        for j in range(place_approach.shape[1]):
            if place_approach[i, j] is not None:
                place_approach[i, j] = place_approach[i, j][::-1]
    place_depatrure = iks_departure

    rack_state = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, ],
                           [2, 2, 1, 0, 0, 0, 0, 0, 0, 0, ],
                           [2, 1, 2, 0, 0, 0, 0, 0, 0, 0, ],
                           [1, 2, 1, 0, 0, 0, 0, 0, 0, 0, ],
                           [1, 1, 2, 0, 0, 0, 0, 0, 0, 0, ]])

    goal_state_dict = {}
    goal_state_dict[1] = {'pattern': np.array([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, ],
                                               [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, ],
                                               [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, ],
                                               [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, ],
                                               [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, ]]),
                          'place_approach_motion': place_approach,
                          'place_departure_motion': place_depatrure,
                          'occ_cnt': 0, }
    goal_state_indices = np.argwhere(goal_state_dict[1]['pattern'] > 0)
    goal_state_dict[1]['goal_state_indices'] = goal_state_indices[np.argsort(goal_state_indices[:, 1])][::-1]

    goal_state_dict[2] = {'pattern': np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, ],
                                               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, ],
                                               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, ],
                                               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, ],
                                               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, ]]),
                          'place_approach_motion': place_approach,
                          'place_departure_motion': place_depatrure,
                          'occ_cnt': 0, }
    goal_state_indices = np.argwhere(goal_state_dict[2]['pattern'] > 0)
    goal_state_dict[2]['goal_state_indices'] = goal_state_indices[np.argsort(goal_state_indices[:, 1])][::-1]

    goal_state_dict[3] = {'pattern': np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, ],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, ],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, ],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, ],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, ]]),
                          'place_approach_motion': place_approach,
                          'place_departure_motion': place_depatrure,
                          'occ_cnt': 0, }
    goal_state_indices = np.argwhere(goal_state_dict[3]['pattern'] > 0)
    goal_state_dict[3]['goal_state_indices'] = goal_state_indices[np.argsort(goal_state_indices[:, 1])][::-1]

    # Get the indices of the entries in `rack_state` with positive value, represented them in (i,j) format and resort them in ascending order of column index
    rack_state_indices = np.argwhere(rack_state > 0)
    rack_state_indices = rack_state_indices[np.argsort(rack_state_indices[:, 1])]

    tube_name = 'tube1'

    # Get the indices of the entries in `goal_state_dict['tube1']['pattern']` with positive value, represented them in (i,j) format and resort them in descending order of column index

    from utils import yumi_solve_ik, examine_rbt_con_fk
    from huri.components.yumi_control.yumi_con import YumiController, move_rrt
    from huri.math.units import Mm

    collision_box = cm.CollisionModel(gm.gen_box([Mm(240), Mm(350), Mm(100)], rgba=[1, 0, 0, .1]))
    collision_box.set_pos(np.array([Mm(435.91002) + Mm(200) / 2, Mm(-192.92001) + Mm(300) / 2, Mm(120.67001) + Mm(50)]))
    collision_box2 = cm.CollisionModel(gm.gen_box([Mm(200), Mm(350), Mm(100)], rgba=[1, 0, 0, .1]))
    collision_box2.set_pos(
        np.array([Mm(435.91002) + Mm(200) / 2, Mm(-192.92001) + Mm(350) / 2, Mm(120.67001) - Mm(10)]))
    component_name = 'rgt_arm'
    rbt_sim = ym.Yumi()
    rbt_con = YumiController()
    examine_rbt_con_fk(rbt_con)
    component_name = 'rgt_arm'
    v = -1
    v2 = 500
    # v = 100
    # v2 = 50

    toggle_move_p = True

    for (i, j) in rack_state_indices:
        tube_type = rack_state[i, j]
        if goal_state_dict[tube_type]['occ_cnt'] >= len(goal_state_dict[tube_type]['goal_state_indices']):
            break
        goal_index = goal_state_dict[tube_type]['goal_state_indices'][goal_state_dict[tube_type]['occ_cnt']]
        goal_state_dict[tube_type]['occ_cnt'] += 1

        place_approach_motion = goal_state_dict[tube_type]['place_approach_motion'][tuple(goal_index)]
        place_depatrure_motion = goal_state_dict[tube_type]['place_departure_motion'][tuple(goal_index)]

        if iks_approach[i, j] is not None and len(
                iks_approach[i, j]) > 0 and place_approach_motion is not None and place_approach_motion is not None:
            rbt_con.set_gripper_width(component_name, 0.018)
            # rbt_con.move_jnts(component_name=component_name, jnt_vals=iks_approach[i, j][0], speed_n=v)
            if (i,j) == (0,0):
                rrt_path = move_rrt(rbt_sim, rbt_con, iks_approach[i, j][0], component_name, speed_n=v,
                                    obstacle_list=[collision_box2, ])
            if not toggle_move_p:
                rrt_path = move_rrt(rbt_sim, rbt_con, iks_approach[i, j][0], component_name, speed_n=v,
                                    obstacle_list=[collision_box2, ])
            else:
                p, r, c, e = rbt_con.fk(component_name, iks_approach[i, j][0], return_conf=True)
                rbt_con.move_p(component_name, p, r, c, e, linear=True, speed_n=v)

            rbt_con.move_jntspace_path(component_name, iks_approach[i, j], speed_n=v2)
            # grasp tube
            rbt_con.open_gripper(component_name=component_name)
            rbt_con.move_jntspace_path(component_name, iks_departure[i, j], speed_n=v)

            # move to another place
            if not toggle_move_p:
                rrt_path = move_rrt(rbt_sim, rbt_con, place_approach_motion[0], component_name, speed_n=v,
                                    obstacle_list=[collision_box, ])
            else:
                p, r, c, e = rbt_con.fk(component_name, place_approach_motion[0], return_conf=True)
                rbt_con.move_p(component_name, p, r, c, e, linear=True, speed_n=v)
            # rbt_con.move_jnts(component_name=component_name, jnt_vals=place_approach_motion[0], speed_n=v)
            rbt_con.move_jntspace_path(component_name, place_approach_motion, speed_n=v2)
            rbt_con.set_gripper_width(component_name, 0.018)
            rbt_con.move_jntspace_path(component_name, place_depatrure_motion, speed_n=v)
        # input("Press Enter to continue...")
