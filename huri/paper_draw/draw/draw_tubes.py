if __name__ == "__main__":
    import numpy as np
    from huri.core.common_import import wd, cm
    from huri.core.base_boost import zoombase, boost_base
    from huri.definitions.tube_def import TubeType
    from robot_sim.end_effectors.gripper.yumi_gripper.yumi_gripper import YumiGripper

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])

    blue_tube_cm = TubeType.gen_tube_by_name("blue cap").gen_collision_model()
    purple_tube_cm = TubeType.gen_tube_by_name("purple cap").gen_collision_model()
    white_tube_cm = TubeType.gen_tube_by_name("white cap").gen_collision_model()

    blue_tube_cm.set_pos(np.array([-.05, 0, 0]))
    white_tube_cm.set_pos(np.array([.05, 0, 0]))

    # blue_tube_cm.set_rgba(np.array([65 / 255, 105 / 255, 225 / 255, 1]))
    # purple_tube_cm.set_rgba(np.array([186 / 255, 85 / 255, 211 / 255, 1]))
    # white_tube_cm.set_rgba(np.array([0, 0, 0, 1]))
    blue_tube_cm.show_cdprimit()
    blue_tube_cm.attach_to(base)
    purple_tube_cm.attach_to(base)
    purple_tube_cm.show_cdprimit()
    white_tube_cm.attach_to(base)
    white_tube_cm.show_cdprimit()

    base = boost_base(base)
    base.boost.add_task(zoombase, args=[base, np.array([0, 1.2, 0])], timestep=0.2)
    base.boost.add_task(lambda task: base.boost.screen_shot("tubes"), timestep=0.4)

    base.run()
