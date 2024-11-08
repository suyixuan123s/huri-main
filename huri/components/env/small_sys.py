import numpy as np

from robot_sim.manipulators.xarm_lite6.xarm_lite6 import XArmLite6
from huri.core.common_import import wd, rm
from huri.definitions.rack_def import Rack_Hard_Proto, TubeType
from huri.learning.env.arrangement_planning_rack.utils import get_random_states

GOAL_PATTERN = np.array([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]])

base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2], w=2000, h=2000)
rbt = XArmLite6(enable_cc=True)

mesh_mdl = rbt.gen_meshmodel()
mesh_mdl.attach_to(base)

rack_hard = Rack_Hard_Proto.copy()
random_state = get_random_states(rack_size=np.array([5, 10]),
                                 goalpattern=GOAL_PATTERN, min_obj_num=20)
slot_ids = np.vstack(np.where(random_state > 0)).T
for slot_id in slot_ids:
    rack_hard.insert_tube(slot_id=slot_id, tube=TubeType.gen_tube_by_tubetype(random_state[tuple(slot_id)]))
rack_pos = np.array([.25, 0, 0])
rack_rot = np.eye(3)
rack_homomat = rm.homomat_from_posrot(rack_pos, rack_rot)
rack_hard.set_homomat(rack_homomat)
rack_hard.gen_mesh_model(gen_tube=True).attach_to(base)

base.run()
