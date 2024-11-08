import numpy as np

from huri.core.common_import import wd, cm, fs, ym
from huri.core.base_boost import zoombase, boost_base
from utils import phoxi
from huri.definitions.tube_def import TubeType

base = wd.World(cam_pos=[3.4, 0, .6], lookat_pos=[.3, 0, .5])
robot = ym.Yumi()
robot.fk("rgt_arm", np.array([0.99291781, -1.52541777, -1.52925749, 0.3122394, -0.33946654,
                              1.15313904, -0.8545132]))


phoxi.set_pos(np.array([0.27, 0, 1.05]))
phoxi.set_rpy(np.radians(90), 0, np.radians(90))
phoxi.attach_to(base)

pos, rot = robot.get_gl_tcp("rgt_arm")

purple_tube = TubeType.gen_tube_by_name("purple cap")
purple_tube_cm = purple_tube.gen_collision_model()
purple_tube_cm.set_pos(pos - rot[:3,2]*0.03 + rot[:3,1]*0.05)
# purple_tube_cm.set_rotmat(rot)
purple_tube_cm.attach_to(base)

robot.jaw_to("rgt_hnd", 0.019)
robot.gen_meshmodel().attach_to(base)

# phoxi.objpdnp.place()
base = boost_base(base)
# base.boost.add_task(zoombase, args=[base, np.array([1, 0, 1])], timestep=0.2)
base.boost.add_task(lambda task: base.boost.screen_shot("robot_movement"), timestep=0.4)
base.boost.add_task(lambda task: exit(), timestep=0.6)

base.run()
