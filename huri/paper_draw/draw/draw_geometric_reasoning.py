if __name__ == "__main__":
    import numpy as np
    from huri.core.common_import import wd, fs, ym
    from huri.core.base_boost import zoombase, boost_base
    from huri.definitions.tube_def import TubeType
    from huri.definitions.rack_def import Rack_Hard_Proto
    from robot_sim.end_effectors.gripper.yumi_gripper.yumi_gripper import YumiGripper

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])

    robot = ym.Yumi()
    #
    rack_arrangement = np.array([[2, 1, 1, 2, 2, 2, 0, 0, 3, 0],
       [1, 0, 0, 2, 0, 0, 0, 0, 0, 0],
       [1, 1, 0, 0, 3, 1, 0, 0, 3, 0],
       [1, 0, 0, 2, 3, 0, 0, 0, 0, 0],
       [1, 0, 0, 2, 2, 0, 0, 0, 3, 3]])
    rack = Rack_Hard_Proto.copy()
    rack.set_homomat(np.array([[0.99428, -0.10667, 0.0054028, 0.38444],
                               [0.10672, 0.99422, -0.011958, -0.22955],
                               [-0.0040961, 0.012466, 0.99991, -0.00034298],
                               [0, 0, 0, 1]]))

    slot_ids = np.vstack(np.where(rack_arrangement > 0)).T
    for slot_id in slot_ids:
        rack.insert_tube(slot_id=slot_id,
                         tube=TubeType.gen_tube_by_tubetype(rack_arrangement[slot_id[0], slot_id[1]]))

    from_slot_id = (0, 0)
    to_slot_id = (0, 9)
    is_grasp_start = True
    if is_grasp_start:
        rack_model = rack.gen_mesh_model(gen_tube=True)
        rack_model.attach_to(base)

    moved_tube, moved_tube_homo_gl = rack.remove_slot(slot_id=from_slot_id)

    if not is_grasp_start:
        rack.insert_tube(slot_id=to_slot_id, tube=moved_tube)
        rack_model = rack.gen_mesh_model(gen_tube=True)
        rack_model.attach_to(base)
        moved_tube, moved_tube_homo_gl = rack.remove_slot(slot_id=to_slot_id)

    grpr = YumiGripper(enable_cc=True)
    grasp_info_list = fs.load_json(path=TubeType.gen_tube_by_tubetype(moved_tube).grasps_path)
    cnt = 0
    init_pos = moved_tube_homo_gl[:3, 3]
    init_rotmat = moved_tube_homo_gl[:3, :3]
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
        is_hnd_collision = gic.is_collided(rack.gen_collision_model(gen_tube=True)[1].cm_list)
        if is_hnd_collision:
            # gic.gen_meshmodel(rgba=[1, 0, 0, .3]).attach_to(base)
            pass
        else:
            init_jaw_center_approach_pos = init_jaw_center_pos + np.array([0, 0, .15])
            approach_jnt_values = robot.ik("rgt_arm",
                                           init_jaw_center_approach_pos,
                                           init_jaw_center_rotmat,
                                           # seed_jnt_values=jnt_values.copy(),
                                           local_minima="end")

            if approach_jnt_values is not None:
                jnt_values = robot.ik("rgt_arm",
                                      init_jaw_center_pos,
                                      init_jaw_center_rotmat,
                                      seed_jnt_values=approach_jnt_values.copy(),
                                      local_minima="end")
                if jnt_values is not None:
                    gic.gen_meshmodel(rgba=[0, 1, 0, .3]).attach_to(base)
                else:
                    gic.gen_meshmodel(rgba=[1, 1, 0, .3]).attach_to(base)
                    pass
            else:
                gic.gen_meshmodel(rgba=[1, 1, 0, .3]).attach_to(base)
                pass

    base = boost_base(base)
    # base.boost.add_task(zoombase, args=[base, np.array([1, 0, 1])], timestep=0.2)
    # base.boost.add_task(lambda task: base.boost.screen_shot("grasps"), timestep=0.4)
    # base.boost.add_task(lambda task: exit(), timestep=0.6)
    base.run()
