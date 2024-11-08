from typing import List

import numpy as np

from huri.core.common_import import cm, gm, rm
from panda3d.core import NodePath
import modeling.model_collection as mc


class Phoxi(object):
    def __init__(self, show_cone = True):
        scanner_p1 = cm.CollisionModel(initor="./model/scanner_part_1.stl")
        scanner_p2 = cm.CollisionModel(initor="./model/scanner_part_2.stl")
        cone = cm.CollisionModel(initor="./model/cone.stl")
        origin = gm.gen_sphere(radius=.0001)
        scanner_p1.objpdnp.reparentTo(origin.objpdnp)
        scanner_p2.objpdnp.reparentTo(origin.objpdnp)
        if show_cone:
            cone.objpdnp.reparentTo(origin.objpdnp)
        scanner_p1.set_rgba(np.array([14 / 255, 17 / 255, 17 / 255, 1]))
        scanner_p2.set_rgba(np.array([129 / 255, 136 / 255, 140 / 255, 1]))
        cone.set_rgba(np.array([0, 0, 0, .1]))
        self.base_mdl = cm.CollisionModel(origin.objpdnp)
        self.origin = np.array([-.17272, -.01741, -.0027])

    def attach_to(self, *args, **kwargs):
        self.base_mdl.attach_to(*args, **kwargs)

    def set_pos(self, *args, **kwargs):
        self.base_mdl.set_pos(*args, **kwargs)

    def set_rpy(self, *args, **kwargs):
        self.base_mdl.set_rpy(*args, **kwargs)

    def get_origin(self):
        return rm.homomat_transform_points(self.base_mdl.get_homomat(), self.origin)


phoxi = Phoxi()
phoxi_nocone = Phoxi(show_cone=True)