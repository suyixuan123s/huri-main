import time

if __name__ == "__main__":
    import numpy as np
    from numpy import array
    from huri.core.common_import import wd, fs, rm
    from huri.definitions.tube_def import TubeType
    from huri.definitions.rack_def import Rack_Hard_Proto
    from robot_sim.end_effectors.gripper.lite6_wrs_gripper import Lite6WRSGripper
    from utils import init_env_rbt, np

    base, rbthp = init_env_rbt()
    #

    rack_state = array([[1, 1, 0, 0, 0, 0, 0, 0, 1, 3],
       [0, 0, 1, 1, 3, 3, 0, 0, 0, 0],
       [1, 1, 1, 2, 2, 3, 0, 0, 0, 0],
       [1, 0, 0, 2, 2, 2, 0, 0, 0, 0],
       [1, 2, 1, 2, 0, 0, 0, 0, 0, 0]])
    rack_state[rack_state == 5] = 1
    print(repr(rack_state))
    rack_tf = np.array([[0.99682, -0.079741, -0.0010215, 0.35342],
                        [0.079738, 0.99642, 0.027996, -0.055426],
                        [-0.0012146, -0.027989, 0.99961, 0.0087556],
                        [0, 0, 0, 1]])
    rack_tf = np.dot(rm.homomat_from_posrot(rot=rm.rotmat_from_axangle([1, 1, 1], np.radians(20))),
                     rack_tf)

    tube_rack = Rack_Hard_Proto.copy()
    tube_rack.set_homomat(rack_tf)
    slot_ids = np.vstack(np.where(rack_state > 0)).T
    for slot_id in slot_ids:
        tube_rack.insert_tube(slot_id=slot_id, tube=TubeType.gen_tube_by_tubetype(rack_state[tuple(slot_id)]))

    from_slot_id = (1,5)
    to_slot_id = (3, 0)
    is_grasp_start = True
    if is_grasp_start:
        rack_model = tube_rack.gen_mesh_model(gen_tube=True)
        rack_model.attach_to(base)

    grpr = Lite6WRSGripper(enable_cc=True)
    grasp_path = fs.Path(TubeType.gen_tube_by_tubetype(rack_state[tuple(from_slot_id)]).grasps_path)
    grasp_path = grasp_path.parent.joinpath("lite6", grasp_path.name)
    grasp_info_list = fs.load_json(path=grasp_path)
    a = time.time()
    grasp_info_list = tube_rack.get_grasps_local_state(from_slot_id, grasp_info_list)
    b = time.time()
    print(b - a)

    # grasp direction
    rack_tmp = tube_rack
    init_slot_ind = from_slot_id
    # ------
    init_local_geometry = np.pad(rack_tmp.rack_status, 1)[init_slot_ind[0]:init_slot_ind[0] + 3,
                          init_slot_ind[1]:init_slot_ind[1] + 3].copy()
    init_local_geometry[init_local_geometry > 0] = 1
    u, d, l, r = init_local_geometry[0, 1], init_local_geometry[2, 1], init_local_geometry[1, 0], \
                 init_local_geometry[1, 2]
    ul, ur, dl, dr = init_local_geometry[0, 0], init_local_geometry[0, 2], init_local_geometry[2, 0], \
                     init_local_geometry[2, 2]

    u_dir = rack_tf[:3, 0]
    d_dir = -rack_tf[:3, 0]
    l_dir = rack_tf[:3, 1]
    r_dir = -rack_tf[:3, 1]
    ul_dir = rack_tf[:3, 0]
    ur_dir = rack_tf[:3, 0]
    dl_dir = -rack_tf[:3, 0]
    dr_dir = -rack_tf[:3, 0]

    z_dir = rack_tf[:3, 2]
    lean_factor = .3
    dir = u * u_dir + d * d_dir + l * l_dir + r * r_dir + ul * ul_dir + ur * ur_dir + dl * dl_dir + dr * dr_dir
    if np.linalg.norm(dir) > 1e-5:
        grasp_dir = rm.unit_vector(rm.unit_vector(dir) * lean_factor + z_dir)
    else:
        grasp_dir = z_dir
    import modeling.geometric_model as gm

    # ------

    moved_tube, moved_tube_homo_gl = tube_rack.remove_slot(slot_id=from_slot_id)

    if not is_grasp_start:
        tube_rack.insert_tube(slot_id=to_slot_id, tube=moved_tube)
        rack_model = tube_rack.gen_mesh_model(gen_tube=True)
        rack_model.attach_to(base)
        moved_tube, moved_tube_homo_gl = tube_rack.remove_slot(slot_id=to_slot_id)

    cnt = 0
    init_pos = moved_tube_homo_gl[:3, 3]
    init_rotmat = moved_tube_homo_gl[:3, :3]


    print("Number of grasps is", len(grasp_dir))

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
        gm.gen_arrow(spos=init_jaw_center_pos, epos=init_jaw_center_pos - grasp_dir * .05).attach_to(base)
        # gic.gen_meshmodel().attach_to(base)
        # continue

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
