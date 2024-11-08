from typing import List, Union

import numpy as np

from huri.core.common_import import wd, cm
from robot_sim.robots.xarm_lite6_wrs import XArmLite6WRSGripper
from robot_con.xarm_lite6 import XArmLite6X
from robot_sim.robots.robot_interface import RobotInterface
from huri.core.base_boost import boost_base
from modeling.model_collection import ModelCollection

from robot_sim.end_effectors.gripper.lite6_wrs_gripper import Lite6WRSGripper


class RobotHelper():
    def __init__(self, sim_rbt: RobotInterface, rbt_con: 'RobotController'):
        self.sim_rbt = sim_rbt
        self.rbt_con = rbt_con

    def fk(self, *args, **kwargs):
        raise NotImplementedError

    def ik(self, *args, **kwargs):
        raise NotImplementedError


class XArmLite6Helper(RobotHelper):
    def __init__(self, sim_rbt: XArmLite6WRSGripper, rbt_con: XArmLite6X):
        super(XArmLite6Helper, self).__init__(sim_rbt, rbt_con)
        self._toggle_online = False if self.rbt_con is None else True
        self.gripper = Lite6WRSGripper(enable_cc=True)
        self.sync()

    def fk(self, arm_jnts: np.ndarray):
        self.sim_rbt.fk("arm", arm_jnts)

    def ik(self, tgt_pos: np.ndarray, tgt_rot: np.ndarray) -> np.ndarray:
        return self.rbt_con.ik(tgt_pos, tgt_rot)

    def sync(self):
        if self._toggle_online:
            self.fk(self.rbt_con.get_jnt_values())

    def get_jnt_values(self) -> np.ndarray:
        if self._toggle_online:
            self.sync()
        return self.sim_rbt.get_jnt_values("arm")

    def gen_mesh_model(self, alpha=None) -> ModelCollection:
        mdl_mc = self.sim_rbt.gen_meshmodel()
        if alpha is not None:
            for mdl_cm in mdl_mc.cm_list:
                _c = mdl_cm.get_rgba()
                mdl_cm.set_rgba([_c[0], _c[1], _c[2], alpha])
        return mdl_mc

    def check_grasp_collision_at_pose(self, pos, rot, obs_list=None, toggle=False):
        if obs_list is None:
            obs_list = []
        hnd = self.gripper.copy()
        hnd.grip_at_with_jcpose(pos, rot, .034)
        if toggle:
            hnd.show_cdprimit()
        return hnd.is_collided(obs_list)

    def hnd_model_at(self, pos, rot):
        hnd = self.gripper.copy()
        hnd.grip_at_with_jcpose(pos, rot, .034)
        return hnd

    def exec_pick(self):
        pass

    def exec_place(self):
        pass

    def exec_pick_place(self):
        pass


def init_env_rbt(toggle_online=True) -> (wd.World, XArmLite6Helper):
    sim_rbt = XArmLite6WRSGripper()
    if toggle_online:
        rbt_con = XArmLite6X(has_gripper=True)
    else:
        rbt_con = None
    rbthp = XArmLite6Helper(sim_rbt=sim_rbt, rbt_con=rbt_con)
    return rbthp


class MotionElement:
    store_robot_gm = None
    store_obj_gm = None
    store_obs_list_gm = None

    def __init__(self,
                 init_homo: np.ndarray,
                 init_approach_homo: np.ndarray,
                 init_jawwidth: float,
                 tgt_homo_list: np.ndarray,
                 tgt_approach_homo_list: np.ndarray,
                 tgt_tcp_sym_angles: list,
                 tgt_jawwidth_list: list,
                 grasp_id: int,
                 grasp_info_list: list,
                 move_tube,
                 rack_tf: np.ndarray,
                 init_slot_id,
                 goal_slot_id,
                 hnd_name: str,
                 obs_list: Union[List[cm.CollisionModel], ModelCollection], ):

        self.init_homo = init_homo
        self.init_approach_homo = init_approach_homo
        self.init_jawwidth = init_jawwidth
        self.tgt_homo_list = tgt_homo_list
        self.tgt_approach_homo_list = tgt_approach_homo_list
        self.tgt_tcp_sym_angles = tgt_tcp_sym_angles
        self.tgt_jawwidth_list = tgt_jawwidth_list

        self.grasp_id = grasp_id
        self.tube = move_tube
        self.rack_tf = rack_tf
        self.init_slot_id = init_slot_id
        self.goal_slot_id = goal_slot_id
        self.grasp_info_list = grasp_info_list

        self.hnd_name = hnd_name
        if obs_list is not None:
            if not isinstance(obs_list, ModelCollection):
                obs_list_mc = ModelCollection("obs model list")
                [obs.attach_to(obs_list_mc) for obs in obs_list]
            else:
                obs_list_mc = obs_list
            self.obs_list = obs_list_mc
        else:
            self.obs_list = None
        self._counter = 0

    @property
    def pick_approach(self):
        return self.init_approach_homo

    @property
    def pick(self):
        return self.init_homo

    @property
    def pick_jawwidth(self):
        return self.init_jawwidth

    @property
    def place_approach(self):
        return self.tgt_approach_homo_list[0]

    @property
    def place(self):
        return self.tgt_homo_list[0]

    @property
    def place_jawwidth(self):
        return self.tgt_jawwidth_list[0]


class MotionBatch:
    def __init__(self):
        self.elements = []
        self.element_names = []
        self._counter = 0
        self.motionelement_iter = None

    def append(self, ele: MotionElement, ele_name: str = "none"):
        self.elements.append(ele)
        self.element_names.append(ele_name)

    @property
    def current(self):
        if self.motionelement_iter is None:
            self.__next__()
        return self.motionelement_iter

    def add_(self, other):
        self.elements.extend(other.elements[other._counter_rgt:])

    def __len__(self):
        return len(self.elements)

    def __iter__(self):
        self._counter = 0
        return self

    def __getitem__(self, item):
        return self.elements[item]

    def __next__(self):
        if self._counter >= len(self.elements):
            self._counter = self._counter - 1
            raise StopIteration
        self.motionelement_iter = iter(self.elements[self._counter])
        self.motion_name = self.element_names[self._counter]
        self._counter += 1
        return self.motion_name, self.motionelement_iter

    def get_last_rbt_homomat(self, component_name):
        if len(self.elements) == 0:
            return None
        last_rbt_homomat = None
        for _ in range(0, len(self.elements))[::-1]:
            if self.elements[_].hnd_name == component_name:
                last_rbt_homomat = self.elements[_].place_approach
                break
        return last_rbt_homomat
