import copy
import math
import modeling.geometric_model as gm
import numpy as np
import modeling.dynamics.bullet.bdbody as bdb
from visualization.panda.world import ShowBase
import basis.data_adapter as da
from panda3d.bullet import BulletDebugNode, BulletWorld


class BDModel(object):
    """
    load an object as a bullet dynamics model
    author: weiwei
    date: 20190627
    """

    def __init__(self,
                 objinit,
                 mass=None,
                 restitution=0,
                 allowdeactivation=False,
                 allowccd=True,
                 friction=.2,
                 dynamic=True,
                 physics_scale: None or float = 1e3,
                 type="convex",
                 name="bdm"):
        """
        :param objinit: GeometricModel (CollisionModel also work)
        :param mass:
        :param restitution:
        :param allowdeactivation:
        :param allowccd:
        :param friction:
        :param dynamic:
        :param type: "convex", "triangle", "box", "sphere"
        :param name:
        """
        if physics_scale is None:
            self.physics_scale = base.physics_scale
        else:
            self.physics_scale = physics_scale

        if isinstance(objinit, BDModel):
            self._gm = copy.deepcopy(objinit.gm)
            self._bdb = objinit.bdb.copy()
        elif isinstance(objinit, gm.GeometricModel):
            if mass is None:
                mass = 0
            self._gm = objinit
            self._bdb = bdb.BDBody(self._gm, type, mass, restitution, allow_deactivation=allowdeactivation,
                                   allow_ccd=allowccd, friction=friction, dynamic=dynamic, name=name)
        else:
            if mass is None:
                mass = 0
            self._gm = gm.GeometricModel(objinit)
            self._bdb = bdb.BDBody(self._gm, type, mass, restitution, allow_deactivation=allowdeactivation,
                                   allow_ccd=allowccd, friction=friction, dynamic=dynamic, name=name,
                                   physics_scale=self.physics_scale)

    @property
    def gm(self):
        # read-only property
        return self._gm

    @property
    def bdb(self):
        # read-only property
        return self._bdb

    @property
    def is_kinemetic(self):
        return self._bdb.is_kinematic()

    def set_rgba(self, rgba):
        self._gm.set_rgba(rgba)

    def clear_rgba(self):
        self._gm.clear_rgba()

    def get_rgba(self):
        return self._gm.get_rgba()

    def set_pos(self, npvec3):
        homomat_bdb = self._bdb.get_homomat()
        homomat_bdb[:3, 3] = npvec3
        self._bdb.set_homomat(homomat_bdb)
        self._gm.set_homomat(homomat_bdb)

    def get_pos(self):
        return self._bdb.get_pos()

    def set_homomat(self, npmat4):
        self._bdb.set_homomat(npmat4)
        self._gm.set_homomat(npmat4)

    def get_homomat(self):
        return self._bdb.get_homomat()

    def set_mass(self, mass):
        """
        Set the mass of the object
        :param mass:
        :return:
        """
        self._bdb.set_mass(mass)

    def set_restitution(self, restitution: float):
        """
        Set the coefficient of restitution
        :param cor:
        :return:
        """
        assert 0 <= restitution <= 1, "The coefficient of restitution must be between 0 and 1"
        self._bdb.set_restitution(restitution)

    def set_friction(self, friction: float):
        """
        Set the friction coefficient
        :param friction:
        :return:
        """
        assert 0 <= friction, "The friction coefficient must be non-negative"
        self._bdb.set_friction(friction)

    def set_speed(self, linear_speed: np.ndarray = None, angular_speed: np.ndarray = None):
        """
        Set the speed of the object
        :param speed:
        :return:
        """
        if linear_speed is not None:
            self.set_linear_vel(linear_speed)
        if angular_speed is not None:
            self.set_angular_vel(angular_speed)

    def set_linear_vel(self, vel: np.ndarray):
        """
        Set the speed of the object
        :param vel:
        :return:
        """
        self._bdb.set_linear_velocity(da.npv3_to_pdv3(vel) * self.physics_scale)

    def set_angular_vel(self, vel: np.ndarray):
        """
        Set the angular speed of the object
        :param vel:
        :return:
        """
        self._bdb.set_angular_velocity(da.npv3_to_pdv3(vel) * self.physics_scale)

    def apply_central_impulse(self, force: np.ndarray):
        """
        Apply a central force impulse to the object
        :param force:
        :return:
        """
        self._bdb.apply_central_impulse(da.npv3_to_pdv3(force) * self.physics_scale)

    def set_kinematic(self, kinematic: bool):
        """
        Set the object to be kinematic
        :param kinematic:
        :return:
        """
        self._bdb.set_kinematic(kinematic)

    def attach_to(self, obj):
        """
        obj must be base.render
        :param obj:
        :return:
        author: weiwei
        date: 20190627
        """
        if isinstance(obj, ShowBase):
            # for rendering to base.render
            self._gm.set_homomat(self.bdb.get_homomat())  # get updated with dynamics
            self._gm.attach_to(obj)
        else:
            raise ValueError("Must be ShowBase!")

    def remove(self):
        self._gm.remove()

    def detach(self):
        self._gm.detach()

    def attach_to_bw(self, bw: BulletWorld):
        assert isinstance(bw, BulletWorld), "The input must be a BulletWorld object!"
        bw.attach(self._bdb)

    def detach_from_bw(self, bw: BulletWorld):
        assert isinstance(bw, BulletWorld), "The input must be a BulletWorld object!"
        bw.remove(self._bdb)

    def start_physics(self):
        base.physicsworld.attach(self._bdb)

    def end_physics(self):
        base.physicsworld.remove(self._bdb)

    def sync_to_physical(self):
        self._gm.set_homomat(self.bdb.get_homomat())

    def show_loc_frame(self):
        self._gm.showlocalframe()

    def unshow_loc_frame(self):
        self._gm.unshowlocalframe()

    def copy(self):
        return BDModel(self)


if __name__ == "__main__":
    import os
    import numpy as np
    import basis
    import basis.robot_math as rm
    import visualization.panda.world as wd
    import random

    # base = wd.World(cam_pos=[1000, 300, 1000], lookat_pos=[0, 0, 0], toggle_debug=True)
    base = wd.World(cam_pos=[.3, .3, 1], lookat_pos=[0, 0, 0], toggle_debug=False)
    base.setFrameRateMeter(True)
    objpath = os.path.join(basis.__path__[0], "objects", "bunnysim.stl")
    # objpath = os.path.join(basis.__path__[0], "objects", "block.stl")
    bunnycm = BDModel(objpath, mass=1, type="box")

    objpath2 = os.path.join(basis.__path__[0], "objects", "bowlblock.stl")
    bunnycm2 = BDModel(objpath2, mass=0, type="triangles", dynamic=False)
    bunnycm2.set_rgba(np.array([0, 0.7, 0.7, 1.0]))
    bunnycm2.set_pos(np.array([0, 0, 0]))
    bunnycm2.start_physics()
    base.attach_internal_update_obj(bunnycm2)


    def update(bunnycm, task):
        if base.inputmgr.keymap['space'] is True:
            for i in range(1):
                bunnycm1 = bunnycm.copy()
                bunnycm1.set_mass(.1)
                rndcolor = np.random.rand(4)
                rndcolor[-1] = 1
                bunnycm1.set_rgba(rndcolor)
                rotmat = rm.rotmat_from_euler(0, 0, math.pi / 12)
                z = math.floor(i / 100)
                y = math.floor((i - z * 100) / 10)
                x = i - z * 100 - y * 10
                print(x, y, z, "\n")
                bunnycm1.set_homomat(
                    rm.homomat_from_posrot(np.array([x * 0.015 - 0.07, y * 0.015 - 0.07, 0.35 + z * 0.015]), rotmat))
                base.attach_internal_update_obj(bunnycm1)
                bunnycm1.start_physics()
        base.inputmgr.keymap['space'] = False
        return task.cont


    gm.gen_frame().attach_to(base)
    taskMgr.add(update, "addobject", extraArgs=[bunnycm], appendTask=True)

    base.run()
