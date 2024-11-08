if __name__ == "__main__":
    import numpy as np
    from huri.core.common_import import wd, cm, fs, ym, gm
    from huri.core.base_boost import zoombase, boost_base
    from huri.definitions.tube_def import TubeType
    from huri.definitions.rack_def import Rack_Hard_Proto
    from robot_sim.end_effectors.gripper.yumi_gripper.yumi_gripper import YumiGripper

    # base = wd.World(cam_pos=[1.85, -1., .4], lookat_pos=[.3, 0, .3])
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
    rack_tf = np.array([[0.9998, -0.018932, 0.006814, 0.4082],
                        [0.019002, 0.99977, -0.010387, -0.23048],
                        [-0.0066158, 0.010514, 0.99992, 0.00024461],
                        [0, 0, 0, 1]])
    rack.set_homomat(rack_tf)

    slot_ids = np.vstack(np.where(rack_arrangement > 0)).T
    for slot_id in slot_ids:
        rack.insert_tube(slot_id=slot_id,
                         tube=TubeType.gen_tube_by_tubetype(rack_arrangement[slot_id[0], slot_id[1]]))

    moving_sequence = [[(2, 7), (3, 0)], [(3, 9), (0, 0)], [(4, 8), (2, 5)],
                       [(1, 4), (3, 9)], [(4, 2), (0, 7)], [(3, 7), (4, 0)],
                       [(1, 2), (4, 8)], [(3, 4), (3, 7)], [(3, 5), (2, 0)],
                       [(0, 9), (4, 6)]]
    moving_sequence = moving_sequence[:1]
    for i, (from_slot_id, to_slot_id) in enumerate(moving_sequence):
        moved_tube, moved_tube_homo_gl_start = rack.remove_slot(slot_id=from_slot_id)

        rack.insert_tube(slot_id=to_slot_id, tube=moved_tube)
        if i == len(moving_sequence) - 1:
            rack.set_homomat(np.eye(4))
            rack_model = rack.gen_mesh_model(gen_tube=True)
            rack_model.attach_to(base)
            print(repr(rack.rack_status))
            moved_tube, moved_tube_homo_gl_goal = rack.remove_slot(slot_id=to_slot_id)


    # base = boost_base(base)
    # base.boost.add_task(zoombase, args=[base, np.array([0, 0, 1])], timestep=0.2)
    # base.boost.add_task(lambda task: base.boost.screen_shot("grasps"), timestep=0.4)
    # base.boost.add_task(lambda task: exit(), timestep=0.6)
    #
    # base.run()
    # exit()

    grpr = YumiGripper(enable_cc=True, )
    grpr.fix_to(robot_arm.jnts[-1]['gl_posq'], rotmat=robot_arm.jnts[-1]['gl_rotmatq'])
    grasp_info_list = fs.load_json(path=TubeType.gen_tube_by_tubetype(moved_tube).grasps_path)
    # cnt = 0
    # init_pos = moved_tube_homo_gl[:3, 3]
    # init_rotmat = moved_tube_homo_gl[:3, :3]
    # for (jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat) in grasp_info_list:
    #     cnt += 1
    #     # if cnt % 2 != 0:
    #     #     continue
    #     gic = grpr.copy()
    #     init_jaw_center_pos = init_pos + init_rotmat.dot(jaw_center_pos)
    #     # robot ee rot
    #     init_jaw_center_rotmat = init_rotmat.dot(jaw_center_rotmat)
    #     gic.grip_at_with_jcpose(gl_jaw_center_pos=init_jaw_center_pos,
    #                             gl_jaw_center_rotmat=init_jaw_center_rotmat,
    #                             jaw_width=jaw_width + .015)
    #     is_hnd_collision = gic.is_collided(rack.gen_collision_model(gen_tube=True)[1].cm_list)
    #     if is_hnd_collision:
    #         # gic.gen_meshmodel(rgba=[1, 0, 0, .3]).attach_to(base)
    #         pass
    #     else:
    #         init_jaw_center_approach_pos = init_jaw_center_pos + np.array([0, 0, .15])
    #         approach_jnt_values = robot.ik("rgt_arm",
    #                                        init_jaw_center_approach_pos,
    #                                        init_jaw_center_rotmat,
    #                                        # seed_jnt_values=jnt_values.copy(),
    #                                        local_minima="end")
    #
    #
    #         if approach_jnt_values is not None:
    #             jnt_values = robot.ik("rgt_arm",
    #                                   init_jaw_center_pos,
    #                                   init_jaw_center_rotmat,
    #                                   seed_jnt_values=approach_jnt_values.copy(),
    #                                   local_minima="end")
    #             if jnt_values is not None:
    #                 gic.gen_meshmodel(rgba=[0, 1, 0, .3]).attach_to(base)
    #                 gic.gen_meshmodel(rgba=[0, 1, 0, .3]).attach_to(base)
    #                 robot_arm.fk(jnt_values)
    #                 robot_arm.gen_meshmodel(rgba=[0, 1, 0, .7], toggle_tcpcs=False).attach_to(base)
    #             else:
    #                 # gic.gen_meshmodel(rgba=[1, 1, 0, .3]).attach_to(base)
    #                 pass
    #         else:
    #             # gic.gen_meshmodel(rgba=[1, 1, 0, .3]).attach_to(base)
    #             pass

    # robot.gen_meshmodel().attach_to(base)
    import huri.components.planning.symmetric_pick_and_place_planner_c as ppp
    from huri.components.planning.common_grasp_seq import CommonGraspSolver

    hnd_name = "rgt_hnd"
    moved_tube = TubeType.gen_tube_by_tubetype(moved_tube)
    moved_tube_cm = moved_tube.gen_collision_model()
    homomat_list = [moved_tube_homo_gl_start, moved_tube_homo_gl_goal]
    start_conf = robot_arm.homeconf
    obs_testubes_cmlist = rack.gen_collision_model(gen_tube=True)[1].cm_list

    offline = fs.Path(f"ddd_{i}.pkl")
    if offline.exists():
        (start2pickapproach_motion_seg, pickapproach_motion_seg, pickdepart_motion_seg,
         pickdepart2placeapproach_motion_seg, placeapproach_motion_seg, placedepart_motion_seg) = fs.load_pickle(
            offline)
    else:
        common_grasp_solver = CommonGraspSolver(num_workers=10)
        pp_planner = ppp.PickPlacePlanner(robot_s=robot, common_grasp_solver=common_grasp_solver)
        start2pickapproach_motion_seg, pickapproach_motion_seg, pickdepart_motion_seg, \
        pickdepart2placeapproach_motion_seg, placeapproach_motion_seg, placedepart_motion_seg, \
        placedepart2goal_motion_seg, common_grasp_iter_info, is_init_feasible = \
            pp_planner.gen_pick_and_place_motion(hnd_name=hnd_name,
                                                 tube=moved_tube,
                                                 tube_cm=moved_tube_cm,
                                                 grasp_info_list=grasp_info_list,
                                                 goal_homomat_list=homomat_list,
                                                 start_conf=start_conf,
                                                 end_conf=None,
                                                 depart_direction_list=[rack_tf[:3, 2]] * len(
                                                     homomat_list),
                                                 approach_direction_list=[-rack_tf[:3, 2]] * len(
                                                     homomat_list),
                                                 depart_distance_list=[.15, .15],
                                                 approach_distance_list=[.15, .15],
                                                 common_grasp_iter=None,
                                                 approach_jawwidth=None,
                                                 depart_jawwidth=None,
                                                 ad_granularity=.005,
                                                 use_rrt=False,
                                                 obstacle_list=obs_testubes_cmlist,
                                                 grasp_obstacle_list=[],
                                                 use_incremental=False, )
        fs.dump_pickle([start2pickapproach_motion_seg, pickapproach_motion_seg, pickdepart_motion_seg,
                        pickdepart2placeapproach_motion_seg, placeapproach_motion_seg,
                        placedepart_motion_seg, ], offline)

    dv = 5
    ccc = 0
    print(start2pickapproach_motion_seg)
    for i in [pickapproach_motion_seg, pickdepart_motion_seg,
              pickdepart2placeapproach_motion_seg, placeapproach_motion_seg, ]:
        ccc += 1
        motions = i.conf_ls
        jaws = i.jaww_ls
        obj_motions = i.objpose_ls
        cnt = 0

        l = len(motions)
        sq = np.logspace((ccc - 1) / dv, (ccc) / dv, num=l) / 10
        for ii, jnts in enumerate(motions):
            cnt += 1
            if cnt % 5 != 0:
                continue
            robot_arm.fk(jnts)
            gic = grpr.copy()
            gic.fix_to(robot_arm.jnts[-1]['gl_posq'], rotmat=robot_arm.jnts[-1]['gl_rotmatq'])
            gic.jaw_to(jaws[ii])

            rbt_arm_cm = robot_arm.gen_meshmodel(toggle_tcpcs=False).cm_list
            for ooo in rbt_arm_cm + gic.gen_meshmodel().cm_list:
                ooo_color = ooo.get_rgba()
                ooo.set_rgba([ooo_color[0], ooo_color[1], ooo_color[2], sq[ii]])
                ooo.attach_to(base)

            sp_p = gic.pos + np.dot(gic.rotmat, gic.jaw_center_pos - np.array([0, 0, .0]))
            gm.gen_sphere(sp_p, radius=.005).attach_to(base)
            objcm = moved_tube_cm.copy()
            objcm.set_homomat(obj_motions[ii])
            obj_rgba = objcm.get_rgba()
            objcm.set_rgba([obj_rgba[0], obj_rgba[1], obj_rgba[2], .3])
            objcm.attach_to(base)
            # robot_arm
            # grpr.
    robot.gen_meshmodel().attach_to(base)
    # base = boost_base(base)
    # # base.boost.add_task(zoombase, args=[base, np.array([1, 0, 1])], timestep=0.2)
    # base.boost.add_task(lambda task: base.boost.screen_shot("grasps"), timestep=0.4)
    # base.boost.add_task(lambda task: exit(), timestep=0.6)
    #
    # base.run()
