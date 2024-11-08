if __name__ == "__main__":
    import numpy as np
    import visualization.panda.world as wd
    from huri.core.base_boost import zoombase, boost_base
    from robot_sim.end_effectors.gripper.yumi_gripper.yumi_gripper import YumiGripper

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
    grpr = YumiGripper(enable_cc=True)
    grpr.jaw_to(.03)

    base = boost_base(base)
    grpr.gen_meshmodel().attach_to(base)

    base.boost.add_task(zoombase, args=[base, np.array([0, 1, 0])], timestep=0.2)
    base.boost.add_task(lambda task: base.boost.screen_shot("robot_hand"), timestep=0.4)

    base.run()
