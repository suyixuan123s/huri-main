if __name__ == "__main__":
    import numpy as np
    from huri.core.common_import import wd, cm, fs, ym
    from huri.core.base_boost import zoombase, boost_base
    from huri.definitions.tube_def import TubeType
    from huri.definitions.rack_def import Rack_Hard_Proto
    from robot_sim.end_effectors.gripper.yumi_gripper.yumi_gripper import YumiGripper

    base = wd.World(cam_pos=[2, 0, .4], lookat_pos=[.3, 0, .3])

    robot = ym.Yumi()
    rgt_arm_homeconf = np.radians(np.array([-20, -90, -120, 30, .0, 40, 0]))
    robot_arm = ym.ya.IRB14050(pos=np.array([0.05355, -0.0725, 0.41492]),
                               rotmat=np.array([[-0.57125912, 0.10708855, 0.81375369],
                                                [-0.61975354, -0.70629867, -0.3421224],
                                                [0.53811575, -0.69976727, 0.46984806]]),
                               homeconf=rgt_arm_homeconf, enable_cc=False)
    #
    rack_arrangement = np.array([[0, 0, 0, 2, 0, 0, 2, 0, 0, 2],
                                 [1, 0, 3, 0, 3, 2, 0, 3, 0, 0],
                                 [0, 0, 1, 2, 0, 0, 0, 1, 0, 0],
                                 [0, 1, 0, 0, 3, 1, 2, 1, 0, 1],
                                 [0, 0, 3, 0, 0, 0, 0, 0, 2, 0]])
    rack = Rack_Hard_Proto.copy()
    rack.set_homomat(np.array([[0.9998, -0.018932, 0.006814, 0.4082],
                               [0.019002, 0.99977, -0.010387, -0.23048],
                               [-0.0066158, 0.010514, 0.99992, 0.00024461],
                               [0, 0, 0, 1]]))

    slot_ids = np.vstack(np.where(rack_arrangement > 0)).T
    for slot_id in slot_ids:
        rack.insert_tube(slot_id=slot_id,
                         tube=TubeType.gen_tube_by_tubetype(rack_arrangement[slot_id[0], slot_id[1]]))

    from_slot_id = (2, 7)
    to_slot_id = (3, 0)
    is_grasp_start = False
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
        # if cnt % 2 != 0:
        #     continue
        gic = grpr.copy()
        init_jaw_center_pos = init_pos + init_rotmat.dot(jaw_center_pos)
        # robot ee rot
        init_jaw_center_rotmat = init_rotmat.dot(jaw_center_rotmat)
        gic.grip_at_with_jcpose(gl_jaw_center_pos=init_jaw_center_pos,
                                gl_jaw_center_rotmat=init_jaw_center_rotmat,
                                jaw_width=jaw_width + .015)
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
                    gic.gen_meshmodel(rgba=[0, 1, 0, .3]).attach_to(base)
                    robot_arm.fk(jnt_values)
                    robot_arm.gen_meshmodel(rgba=[0, 1, 0, .7], toggle_tcpcs=False).attach_to(base)
                else:
                    # gic.gen_meshmodel(rgba=[1, 1, 0, .3]).attach_to(base)
                    pass
            else:
                # gic.gen_meshmodel(rgba=[1, 1, 0, .3]).attach_to(base)
                pass


    robot.gen_meshmodel().attach_to(base)

    base = boost_base(base)
    # base.boost.add_task(zoombase, args=[base, np.array([1, 0, 1])], timestep=0.2)
    base.boost.add_task(lambda task: base.boost.screen_shot("grasps"), timestep=0.4)
    base.boost.add_task(lambda task: exit(), timestep=0.6)

    base.run()

    tube = TubeType.gen_tube_by_name("purple cap")
    tube_cm = tube.gen_mesh_model()
    grasp_info_list = fs.load_json(path=tube.grasps_path)
    # tube_cm.attach_to(base)
    # tube_cm = cm.CollisionModel("20220610_blue_tube.stl")

    grpr = YumiGripper(enable_cc=True)
    cnt = 0

    # blue_tube_cm = cm.CollisionModel("20220610_blue_tube.stl", )
    purple_tube_cm = cm.CollisionModel("20220610_purple_tube.stl")
    # white_tube_cm = cm.CollisionModel("20220610_white_tube.stl")
    #

    # blue_tube_cm.set_pos(np.array([-.05, 0, 0]))
    # white_tube_cm.set_pos(np.array([.05, 0, 0]))
    #
    # blue_tube_cm.set_rgba(np.array([65 / 255, 105 / 255, 225 / 255, 1]))
    purple_tube_cm.set_rgba(np.array([186 / 255, 85 / 255, 211 / 255, 1]))
    # # white_tube_cm.set_rgba(np.array([0, 0, 0, 1]))
    #
    # blue_tube_cm.attach_to(base)
    purple_tube_cm.attach_to(base)
    # white_tube_cm.attach_to(base)

    base.run()
