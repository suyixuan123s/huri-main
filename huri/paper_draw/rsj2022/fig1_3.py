import numpy as np

from huri.core.common_import import wd, cm, fs, ym, rm, gm
from huri.core.base_boost import zoombase, boost_base
from utils import phoxi
from huri.definitions.tube_def import TubeType
from huri.definitions.rack_def import Rack_Hard_Proto

base = wd.World(cam_pos=[3.4, 0, .6], lookat_pos=[.3, 0, .5])
robot = ym.Yumi()
robot.fk("rgt_arm", np.array([1.15052104, -1.52541777, -1.28246793, 0.39828414, -1.61809475,
                              1.04248516, -0.03543018]))

phoxi.set_pos(np.array([0.27, 0, 1.05]))
phoxi.set_rpy(np.radians(90), 0, np.radians(90))
phoxi.attach_to(base)

pos, rot = robot.get_gl_tcp("rgt_arm")
print(pos)
rack = Rack_Hard_Proto.copy()
rack_arrangement = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [3, 0, 3, 0, 3, 0, 3, 0, 3, 0],
                             [0, 3, 0, 3, 0, 3, 0, 3, 0, 3],
                             [3, 0, 3, 0, 3, 0, 3, 0, 3, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
rack_pos = [0.35331554, -0.1433284 + .095, 0.00024461]
rack.set_homomat(rm.homomat_from_posrot(np.array(rack_pos)))
slot_ids = np.vstack(np.where(rack_arrangement > 0)).T
for slot_id in slot_ids:
    rack.insert_tube(slot_id=slot_id,
                     tube=TubeType.gen_tube_by_tubetype(1))
rack_cm = rack.gen_mesh_model(gen_tube=True)
rack_cm.attach_to(base)

for obj in [robot.gen_meshmodel().cm_list[0]] + robot.gen_meshmodel().cm_list[2:8]:
    obj.attach_to(base)

# human_hnd = gm.GeometricModel("big-hand2.stl")
# human_hnd.set_pos(np.array([.3, -.37, .08]))
# human_hnd.set_rpy(np.radians(0), np.radians(90), np.radians(90))
# human_hnd.set_rgba([250 / 255, 231 / 255, 218 / 255, 1])

human_hnd = gm.GeometricModel("arm.stl")
human_hnd.set_pos(np.array([.37, -0.15, .08]))
# human_hnd.set_rgba([250 / 255, 231 / 255, 218 / 255, 1])
human_hnd.set_rgba([1, 1, 0, 1])
human_hnd.attach_to(base)
# human_hnd.objpdnp.place()

# human_hnd.attach_to(base)
# human_hnd.set_rgba([250 / 255, 231 / 255, 0, 1])
# e = np.array([.3 - 0.01875 , -.36, .08-0.01375])
# a = gm.gen_stick(e, epos=e-np.array([0,.3,0]), thickness=.015,)
# a.attach_to(base)
# robot.gen_stickmodel().attach_to(base)
# human_hnd.objpdnp.place()
base = boost_base(base)
# base.boost.add_task(zoombase, args=[base, np.array([1, 0, 1])], timestep=0.2)
base.boost.add_task(lambda task: base.boost.screen_shot("robot_movement"), timestep=0.4)
base.boost.add_task(lambda task: exit(), timestep=0.6)

base.run()
