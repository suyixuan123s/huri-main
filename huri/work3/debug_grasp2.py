if __name__ == "__main__":
    import numpy as np
    from numpy import array
    from huri.core.common_import import wd, fs
    from huri.definitions.tube_def import TubeType
    from huri.definitions.rack_def import Rack_Hard_Proto
    from robot_sim.end_effectors.gripper.lite6_wrs_gripper import Lite6WRSGripper
    from utils import init_env_rbt, np
    import huri.components.planning.symmetric_pick_and_place_planner_simple as ppp

    base, rbthp = init_env_rbt()
    #

    rack_state = array([[1, 0, 0, 0, 0, 2, 2, 0, 3, 0],
                        [0, 0, 0, 0, 0, 2, 2, 0, 0, 0],
                        [0, 0, 0, 0, 0, 2, 0, 0, 3, 1],
                        [1, 1, 0, 0, 0, 0, 0, 1, 3, 1],
                        [1, 0, 0, 0, 2, 2, 0, 3, 1, 1]])
    rack_state[rack_state == 5] = 1
    print(repr(rack_state))
    rack_tf = np.array([[0.99682, -0.079741, -0.0010215, 0.35342],
                        [0.079738, 0.99642, 0.027996, -0.055426],
                        [-0.0012146, -0.027989, 0.99961, 0.0087556],
                        [0, 0, 0, 1]])

    tube_rack = Rack_Hard_Proto.copy()
    tube_rack.set_homomat(rack_tf)
    slot_ids = np.vstack(np.where(rack_state > 0)).T
    for slot_id in slot_ids:
        tube_rack.insert_tube(slot_id=slot_id, tube=TubeType.gen_tube_by_tubetype(rack_state[tuple(slot_id)]))

    from_slot_id = (4, 9)
    to_slot_id = (4, 1)
    # is_grasp_start = True
    # if is_grasp_start:
    rack_model = tube_rack.gen_mesh_model(gen_tube=True)
    rack_model.attach_to(base)

    moved_tube_type, moved_tube_homo_gl = tube_rack.remove_slot(slot_id=from_slot_id)

    # if not is_grasp_start:
    tube_rack.insert_tube(slot_id=to_slot_id, tube=moved_tube_type)
    rack_model = tube_rack.gen_mesh_model(gen_tube=True)
    rack_model.attach_to(base)
    moved_tube_type, moved_tube_homo_gl_goal = tube_rack.remove_slot(slot_id=to_slot_id)

    grpr = Lite6WRSGripper(enable_cc=True)
    grasp_path = fs.Path(TubeType.gen_tube_by_tubetype(moved_tube_type).grasps_path)
    grasp_path = grasp_path.parent.joinpath("lite6", grasp_path.name)
    grasp_info_list = fs.load_json(path=grasp_path)
    cnt = 0
    # moved_tube_homo_gl[:3, 3] = moved_tube_homo_gl[:3, 3] + grpr.jaw_center_pos
    init_pos = moved_tube_homo_gl[:3, 3]
    init_rotmat = moved_tube_homo_gl[:3, :3]

    hnd_name = "arm"
    obs_testubes_cmlist = tube_rack.gen_collision_model(gen_tube=True)[1].cm_list

    moved_tube = TubeType.gen_tube_by_tubetype(moved_tube_type)

    rbthp.sim_rbt.ik = lambda hand_name, pos, rot, *args, **kwargs: rbthp.ik(pos, rot)

    common_grasp_iter = ppp.CommonGraspIteratorGenerator(robot_s=rbthp.sim_rbt)
    grasp_iter = common_grasp_iter.gen_iterator(
        grasp_info_list=grasp_info_list,
        hand_name=hnd_name,
        obj_init_homomat=moved_tube_homo_gl.copy(),
        sym_obj_tgt_homomats_angles=moved_tube.get_symmetry_homomat_angle(3, moved_tube_homo_gl_goal),
        pick_approach_pos_offset=-tube_rack.get_homomat()[:3, 2] * .12,
        place_approach_pos_offset=-tube_rack.get_homomat()[:3, 2] * (.12),
        obstacle_list=obs_testubes_cmlist,
        toggle_debug=True)
    rr = []
    for gid, ppp in enumerate(grasp_iter):
        print(ppp[0])
        if ppp[0]:
            jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info_list[gid]
            gic = grpr.copy()
            gic.grip_at_with_jcpose(np.array(jaw_center_pos), np.array(jaw_center_rotmat), np.array(jaw_width))
            gic.gen_meshmodel(rgba=[0, 1, 0, .3]).attach_to(base)
            print("GOOD")
            print(ppp)
    grasp_iter.print_error_info()
    del rbthp.rbt_con
    from huri.components.debug.inspect import save_rbt_info, show_animation

    show_animation(info_type="ik_error")
    base.run()

    exit(0)

    for (jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat) in grasp_info_list:
        cnt += 1
        # if cnt % 50 != 0:
        #     continue
        gic = grpr.copy()
        init_jaw_center_pos = init_pos + init_rotmat.dot(jaw_center_pos)
        # robot ee rot
        init_jaw_center_rotmat = init_rotmat.dot(jaw_center_rotmat)
        gic.grip_at_with_jcpose(gl_jaw_center_pos=init_jaw_center_pos,
                                gl_jaw_center_rotmat=init_jaw_center_rotmat,
                                jaw_width=.034)
        is_hnd_collision = gic.is_collided(tube_rack.gen_collision_model(gen_tube=True)[1].cm_list)
        if is_hnd_collision:
            # gic.gen_meshmodel(rgba=[1, 0, 0, .3]).attach_to(base)
            pass
        else:
            init_jaw_center_approach_pos = init_jaw_center_pos - init_jaw_center_rotmat[:3, 2] * 0.15
            # approach_jnt_values = rbthp.sim_rbt.ik("arm",
            #                                        init_jaw_center_approach_pos,
            #                                        init_jaw_center_rotmat,
            #                                        # seed_jnt_values=jnt_values.copy(),
            #                                        local_minima="end")
            approach_jnt_values = rbthp.ik(init_jaw_center_approach_pos,
                                           init_jaw_center_rotmat, )

            if approach_jnt_values is not None:
                rbthp.fk(approach_jnt_values)
                rbthp.gen_mesh_model().attach_to(base)
                # jnt_values = rbthp.sim_rbt.ik("arm",
                #                               init_jaw_center_pos,
                #                               init_jaw_center_rotmat,
                #                               seed_jnt_values=approach_jnt_values.copy(),
                #                               local_minima="end")
                jnt_values = rbthp.ik(init_jaw_center_pos,
                                      init_jaw_center_rotmat, )

                if jnt_values is not None:
                    gic.gen_meshmodel(rgba=[0, 1, 0, .3]).attach_to(base)
                else:
                    gic.gen_meshmodel(rgba=[1, 1, 0, .3]).attach_to(base)
                    pass
            else:
                gic.gen_meshmodel(rgba=[1, 1, 0, .3]).attach_to(base)
                pass

    # base.boost.add_task(zoombase, args=[base, np.array([1, 0, 1])], timestep=0.2)
    # base.boost.add_task(lambda task: base.boost.screen_shot("grasps"), timestep=0.4)
    # base.boost.add_task(lambda task: exit(), timestep=0.6)
    base.run()
