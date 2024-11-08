if __name__ == "__main__":
    import numpy as np
    from huri.core.common_import import wd, cm, fs
    from huri.core.base_boost import zoombase, boost_base
    from huri.definitions.tube_def import TubeType
    from robot_sim.end_effectors.gripper.yumi_gripper.yumi_gripper import YumiGripper

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])

    tube = TubeType.gen_tube_by_name("purple cap")
    tube_cm = tube.gen_mesh_model()
    grasp_info_list = fs.load_json(path=tube.grasps_path)
    # tube_cm.attach_to(base)
    # tube_cm = cm.CollisionModel("20220610_blue_tube.stl")

    grpr = YumiGripper(enable_cc=True)
    cnt = 0
    for (jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat) in grasp_info_list:
        cnt += 1
        if cnt % 2 != 0:
            continue
        gic = grpr.copy()
        gic.grip_at_with_jcpose(np.array(jaw_center_pos), np.array(jaw_center_rotmat), np.array(jaw_width))
        gic.gen_meshmodel(rgba=[66 / 255, 66 / 255, 66 / 255, .1]).attach_to(base)

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

    base = boost_base(base)
    base.boost.add_task(zoombase, args=[base, np.array([0, 1, 0])], timestep=0.2)
    base.boost.add_task(lambda task: base.boost.screen_shot("blue_tube"), timestep=0.4)

    base.run()
