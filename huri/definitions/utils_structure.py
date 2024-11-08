from typing import List, Union

from modeling.model_collection import ModelCollection

from huri.core.common_import import np, cm


class MotionElement:
    store_robot_gm = None
    store_obj_gm = None
    store_obs_list_gm = None
    counter_incremental = 1

    def __init__(self,
                 obj_cm: cm.CollisionModel,
                 objpose_list: List[np.ndarray],
                 conf_list: List[np.ndarray],
                 jawwidth_list: List[np.ndarray],
                 hnd_name: str,
                 obs_list: Union[List[cm.CollisionModel], ModelCollection], ):
        self.obj_cm = obj_cm
        self.objpose_list = objpose_list
        self.conf_list = conf_list
        self.jawwidth_list = jawwidth_list
        self.hnd_name = hnd_name
        if not isinstance(obs_list, ModelCollection):
            obs_list_mc = ModelCollection("obs model list")
            [obs.attach_to(obs_list_mc) for obs in obs_list]
        else:
            obs_list_mc = obs_list
        self.obs_list = obs_list_mc
        self._counter = 0
        if self.obj_cm is not None:
            self.obj_cm.set_homomat(np.eye(4))

    @classmethod
    def reset_robot_gm(cls, model: Union[ModelCollection, cm.CollisionModel]):
        if cls.store_robot_gm is not None:
            cls.store_robot_gm.remove()
        cls.store_robot_gm = model

    @classmethod
    def reset_obj_gm(cls, model: cm.CollisionModel):
        if cls.store_obj_gm is not None:
            cls.store_obj_gm.remove()
        cls.store_obj_gm = model

    @classmethod
    def reset_obs_list_gm(cls, model: ModelCollection):
        if cls.store_obs_list_gm is not None:
            cls.store_obs_list_gm.remove()
        cls.store_obs_list_gm = model

    def is_end(self):
        return self._counter >= len(self)

    def __iter__(self):
        self._counter = 0
        return self

    def __len__(self):
        if self.objpose_list is not None:
            return len(self.objpose_list)
        else:
            return 0

    def __next__(self):
        if self._counter >= len(self):
            self._counter = len(self) - 1
            raise StopIteration
        result = self.obj_cm, self.objpose_list[self._counter], self.conf_list[self._counter], self.jawwidth_list[
            self._counter], self.hnd_name, self.obs_list
        self._counter += self.counter_incremental
        return result

    def __getitem__(self, val):
        return MotionElement(obj_cm=self.obj_cm,
                             objpose_list=self.objpose_list[val],
                             conf_list=self.conf_list[val],
                             jawwidth_list=self.jawwidth_list[val],
                             hnd_name=self.hnd_name,
                             obs_list=self.obs_list, )


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

    def get_last_motion_rbt_jnt(self, component_name):
        if len(self.elements) == 0:
            return None
        last_rbt_jnt = None
        for _ in range(0, len(self.elements))[::-1]:
            if self.elements[_].hnd_name == component_name:
                if self.elements[_].conf_list is None:
                    continue
                last_rbt_jnt = self.elements[_].conf_list[-1]
                break
        return last_rbt_jnt
        # for id in
        # motion_ele = self.elements[-1]
        # motion_ele.jawwidth_list[]


class MotionBatchPPP(MotionBatch):
    def __init__(self,
                 to_pick_motion: MotionElement = None,
                 pick_approach_motion: MotionElement = None,
                 pick_depart_motion: MotionElement = None,
                 to_place_motion: MotionElement = None,
                 place_approach_motion: MotionElement = None,
                 place_depart_motion: MotionElement = None,
                 to_end_motion: MotionElement = None, ):
        super(MotionBatchPPP, self).__init__()
        self.elements = np.array([
            to_pick_motion,
            pick_approach_motion,
            pick_depart_motion,
            to_place_motion,
            place_approach_motion,
            place_depart_motion,
            to_end_motion,
        ], dtype=object)
        self.element_names = [
            "to_pick_motion",
            "pick_approach_motion",
            "pick_depart_motion",
            "to_place_motion",
            "place_approach_motion",
            "place_depart_motion",
            "to_end_motion", ]

    @property
    def to_pick(self):
        return self.elements[0]

    @property
    def pick_approach(self):
        return self.elements[1]

    @property
    def pick_depart(self):
        return self.elements[2]

    @property
    def to_place(self):
        return self.elements[3]

    @property
    def place_approach(self):
        return self.elements[4]

    @property
    def place_depart(self):
        return self.elements[5]

    @property
    def to_end(self):
        return self.elements[6]

    def append(self, ele: MotionElement, ele_name: str = "none"):
        raise Exception("MotionBatch PPP does not support append function")

    def add_(self, other):
        raise Exception("MotionBatch PPP does not support add_ function")
