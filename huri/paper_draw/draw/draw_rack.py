if __name__ == "__main__":
    import numpy as np
    from huri.core.common_import import wd, cm
    from huri.core.base_boost import zoombase, boost_base
    from huri.definitions.rack_def import Rack_Hard_Proto
    from robot_sim.end_effectors.gripper.yumi_gripper.yumi_gripper import YumiGripper

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])

    rack = Rack_Hard_Proto.gen_mesh_model()
    rack.attach_to(base)

    base = boost_base(base)
    base.boost.add_task(zoombase, args=[base, np.array([1, 1, 1])], timestep=0.2)
    base.boost.add_task(lambda task: base.boost.screen_shot("rack"), timestep=0.4)

    base.run()
