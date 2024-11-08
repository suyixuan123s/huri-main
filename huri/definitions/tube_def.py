import copy
from typing import Literal

from scipy.optimize import minimize

import basis.robot_math as rm

from huri.core.common_import import np, fs, gm, cm
from huri.core.utils import color_hex2oct
from huri.math.units import Mm
from huri.math.optimize import Optimization
from huri.math.math import gen_line, distance_to_line, distance_to_line2


class TubeTypeNotDefined(Exception):
    def __init__(self, tubetype):
        super().__init__(f"Undefined tube type: {tubetype}")


class TestTube:
    tube_collision_sz = .002

    def __init__(self,
                 type: int,
                 name: str,
                 model_path: str,
                 grasps_path: str = None,
                 radius: float = 0.005,
                 height: float = 0.0073,
                 color: tuple = (0, 0, 0, 1),
                 grasps_pos_list: list = None,
                 grasp_angle_list: list = None,
                 ):
        if grasps_pos_list is None:
            grasps_pos_list = []

        if grasp_angle_list is None:
            grasp_angle_list = []

        self.type = type
        self.name = name
        self.model_path = str(model_path)
        self.grasps_path = str(grasps_path)
        self.radius = radius
        self.height = height
        self.color = color
        self._tf = np.eye(4)
        self.symmetry_axis = cm.CollisionModel(initor=self.model_path).objtrm.symmetry_axis
        self.grasps_pos_list = grasps_pos_list
        self.grasps_angle_list = grasp_angle_list

    @property
    def diameter(self):
        return self.radius * 2

    @diameter.setter
    def diameter(self, diameter):
        self.radius = diameter / 2

    def set_homomat(self, homomat: np.ndarray):
        assert isinstance(homomat, np.ndarray) and homomat.shape == (4, 4)
        self._tf = homomat

    def get_homomat(self) -> np.ndarray:
        return self._tf

    def get_symmetry_homomat(self, discrete_factor=6, rot_mat=None) -> np.ndarray:
        if rot_mat is None:
            rot_mat = self._tf
        symmetry_axis_gl = np.dot(rot_mat[:3, :3], self.symmetry_axis)
        angles = np.radians(np.arange(0, 360, 360 / discrete_factor))
        return np.asarray([rm.homomat_from_posrot(rot_mat[:3, 3],
                                                  np.dot(rm.rotmat_from_axangle(symmetry_axis_gl, _angle),
                                                         rot_mat[:3, :3])) for
                           _angle in angles])

    def get_symmetry_homomat_at_angle(self, angle: float, rot_mat=None):
        if rot_mat is None:
            rot_mat = self._tf
        if angle == 0:
            return rot_mat
        symmetry_axis_gl = np.dot(rot_mat[:3, :3], self.symmetry_axis)
        return rm.homomat_from_posrot(rot_mat[:3, 3],
                                      np.dot(rm.rotmat_from_axangle(symmetry_axis_gl, angle),
                                             rot_mat[:3, :3]))

    def get_symmetry_homomat_angle(self, discrete_factor=6, rot_mat=None) -> (np.ndarray, np.ndarray):
        if rot_mat is None:
            rot_mat = self._tf
        symmetry_axis_gl = np.dot(rot_mat[:3, :3], self.symmetry_axis)
        angles = np.radians(np.arange(0, 360, 360 / discrete_factor))
        return np.asarray([rm.homomat_from_posrot(rot_mat[:3, 3],
                                                  np.dot(rm.rotmat_from_axangle(symmetry_axis_gl, _angle),
                                                         rot_mat[:3, :3])) for
                           _angle in angles]), angles

    def gen_mesh_model(self) -> gm.GeometricModel:
        mesh_mdl = gm.GeometricModel(initor=self.model_path)
        mesh_mdl.set_rgba(self.color)
        mesh_mdl.set_homomat(self._tf)
        return mesh_mdl

    def gen_collision_model(self) -> cm.CollisionModel:
        primitive_fn = self._cd_primitive_fn_factory()
        if primitive_fn is not None:
            cdprimit_type = "user_defined"
        else:
            cdprimit_type = "cylinder"

        cm_mdl = cm.CollisionModel(initor=self.model_path,
                                   cdprimit_type=cdprimit_type,
                                   userdefined_cdprimitive_fn=primitive_fn,
                                   expand_radius=self.tube_collision_sz)
        cm_mdl.set_rgba(self.color)
        cm_mdl.set_homomat(self._tf)
        return cm_mdl

    def _cd_primitive_fn_factory(self):
        return None

    def __repr__(self):
        return f"Tube type: {self.type}"

    def copy(self):
        return copy.deepcopy(self)


class _TestTube(cm.CollisionModel):
    def __init__(self,
                 type: int,
                 model_path,
                 grasps_path,
                 radius=0.001,
                 height=0.0073,
                 color=(0, 0, 0, 1),
                 cdprimit_type='cylinder',
                 cdmesh_type='triangles',
                 expand_radius=None,
                 name="test tube",
                 userdefined_cdprimitive_fn=None,
                 btransparency=True,
                 btwosided=False):
        super().__init__(str(model_path),
                         cdprimit_type=cdprimit_type,
                         cdmesh_type=cdmesh_type,
                         expand_radius=expand_radius,
                         name=name,
                         userdefined_cdprimitive_fn=userdefined_cdprimitive_fn,
                         btransparency=btransparency,
                         btwosided=btwosided)
        self.type = type
        self.model_path = str(model_path)
        self.grasps_path = str(grasps_path)
        self.radius = radius
        self.height = height
        self.set_rgba(np.array(color))

    @property
    def diameter(self):
        return self.radius * 2

    @diameter.setter
    def diameter(self, diameter):
        self.radius = diameter / 2

    def __repr__(self):
        return f"Tube type: {self.type}"


from panda3d.core import CollisionNode, CollisionBox, Point3


def purple_cdnp(name, radius):
    """
    collison model for purple test tube
    :return:
    """
    collision_node = CollisionNode(name)
    collision_primitive_c0 = CollisionBox(Point3(0, 0, .096 / 2),
                                          x=.0065 + radius, y=.0065 + radius, z=.096 / 2 + radius)
    collision_node.addSolid(collision_primitive_c0)
    collision_primitive_c1 = CollisionBox(Point3(0, 0, .098),
                                          x=.0087 + radius, y=.0087 + radius, z=.002 + radius)
    collision_node.addSolid(collision_primitive_c1)
    return collision_node


# TODO use a configuration file to save the test tube information
class TubeType:
    # EMPTY = 0
    # # blue cap
    # BLUE_CAP_TYPE = 5
    # BLUE_CAP_NAME = "blue"
    # # purple cap
    # PURPLE_CAP_TYPE = 3  # for yumi 1
    # PURPLE_CAP_NAME = "purple"
    # # white cap
    # WHITE_CAP_TYPE = 1
    # WHITE_CAP_NAME = "white"
    # # white cap small
    # WHITE_CAP_SMALL_TYPE = 4
    # WHITE_CAP_SMALL_NAME = "white_small"
    # # purple ring
    # PURPLE_RING_CAP_TYPE = 2  # for yumi 5
    # PURPLE_RING_CAP_NAME = "purple_ring"
    EMPTY = 0
    # blue cap
    TUBE_TYPE_1 = 1
    TUBE_1_NAME = "blue"
    # purple cap
    TUBE_TYPE_2 = 2  # for yumi 1
    TUBE_2_NAME = "purple"
    # white cap
    TUBE_TYPE_3 = 3
    TUBE_3_NAME = "white"
    # white cap small
    TUBE_TYPE_4 = 4
    TUBE_4_NAME = "white_small"
    # purple ring
    TUBE_TYPE_5 = 5  # for yumi 5
    TUBE_5_NAME = "purple_ring"

    TUBE_FOR_DRAW_TYPE = 99
    TUBE_FOR_DRAW_NAME = "draw"

    ################# UNCOMMENT FOR REAL WORLD EXP ################

    TUBE_1 = TestTube(name=TUBE_1_NAME,
                      type=TUBE_TYPE_1,
                      model_path=fs.workdir_model / "tube3.stl",
                      grasps_path=fs.workdir / "data" / "grasps" / "tube1.json",
                      grasp_angle_list=[np.radians(87), np.radians(90)],
                      grasps_pos_list=[np.array([0, 0, Mm(107.5) - Mm(11)]),
                                       # np.array([Mm(2), Mm(2), Mm(107.5) - Mm(12)]),
                                       # np.array([Mm(2), -Mm(2), Mm(107.5) - Mm(12)]),
                                       # np.array([-Mm(2), Mm(2), Mm(107.5) - Mm(12)]),
                                       # np.array([Mm(2), -Mm(2), Mm(107.5) - Mm(12)])
                                       ],
                      radius=Mm(17.90) / 2, height=Mm(107.5),
                      # color=(0 / 255, 0 / 255, 0 / 255, 1),
                      color=(0.86, 0.08, 0.24, 1),
                      )
    TUBE_2 = TestTube(name=TUBE_2_NAME,
                      type=TUBE_TYPE_2,
                      model_path=fs.workdir_model / "tube1.stl",
                      grasps_path=fs.workdir / "data" / "grasps" / "tube2.json",
                      grasp_angle_list=[np.radians(87), np.radians(90)],
                      grasps_pos_list=[np.array([0, 0, Mm(107.5) - Mm(12)])],
                      radius=Mm(17.90) / 2, height=Mm(107.5),
                      color=(243 / 255, 225 / 255, 107 / 255, 1))
    TUBE_3 = TestTube(name=TUBE_3_NAME,
                      type=TUBE_TYPE_3,
                      model_path=fs.workdir_model / "tube3.stl",
                      grasps_path=fs.workdir / "data" / "grasps" / "tube3.json",
                      grasp_angle_list=[np.radians(87), np.radians(90)],
                      grasps_pos_list=[np.array([0, 0, Mm(101) - Mm(12)]),
                                       # np.array([Mm(2), Mm(2), Mm(101) - Mm(12)]),
                                       # np.array([Mm(2), -Mm(2), Mm(101) - Mm(12)]),
                                       # np.array([-Mm(2), Mm(2), Mm(101) - Mm(12)]),
                                       # np.array([Mm(2), -Mm(2), Mm(101) - Mm(12)]),
                                       ],
                      radius=Mm(17.5) / 2, height=Mm(101.4),
                      color=(0.31, 0.78, 0.47, 1))
    TUBE_4 = TestTube(name=TUBE_4_NAME,
                      type=TUBE_TYPE_4,
                      model_path=fs.workdir_model / "tube3.stl",
                      grasps_path=fs.workdir / "data" / "grasps" / "tube4.json",
                      grasp_angle_list=[np.radians(87), np.radians(90)],
                      grasps_pos_list=[np.array([0, 0, Mm(107) - Mm(12)]),
                                       ],
                      radius=Mm(17) / 2, height=Mm(107),
                      color=(0.53, 0.6, 0.990, 1),
                      )
    TUBE_5 = TestTube(name=TUBE_5_NAME,
                      type=TUBE_TYPE_5,
                      model_path=fs.workdir_model / "tube3.stl",
                      grasps_path=fs.workdir / "data" / "grasps" / "tube5.json",
                      grasp_angle_list=[np.radians(87), np.radians(90)],
                      grasps_pos_list=[np.array([0, 0, Mm(107) - Mm(12)]),
                                       ],
                      radius=Mm(18.1) / 2, height=Mm(107),
                      color=(100 / 255, 100 / 255, 100 / 255, 1))

    TUBE_DRAW = TestTube(name=TUBE_FOR_DRAW_NAME,
                         type=TUBE_FOR_DRAW_TYPE,
                         model_path=fs.workdir_model / "20220610_white_tube.stl",
                         grasps_path=fs.workdir / "data" / "grasps" / "draw_tube_grasps.json",
                         radius=Mm(16) / 2, height=Mm(85),
                         grasp_angle_list=[np.radians(85), np.radians(80)],
                         grasps_pos_list=[np.array([0, 0, Mm(107.5) - Mm(15)])],
                         color=(1, 1, 1, 1))
    ################## UNCOMMENT FOR DEMO ##################
    Tubes = {
        6: TestTube(name='PURPLE',
                    type=6,
                    model_path=fs.workdir_model / "20220610_purple_ring_tube.stl",
                    grasps_path=fs.workdir / "data" / "grasps" / "purple_ring_cap_tube_grasps.json",
                    radius=Mm(18) / 2,
                    height=Mm(107.5),
                    color=(0.53, 0.6, 0.990, 1),
                    grasp_angle_list=[np.radians(85)],
                    grasps_pos_list=[np.array([0, 0, Mm(107.5) - Mm(15)])], ),
        7: TestTube(name='WHITE',
                    type=7,
                    model_path=fs.workdir_model / "white_cap.stl",
                    grasps_path=fs.workdir / "data" / "grasps" / "blue_cap_tube_grasps.json",
                    grasp_angle_list=[np.radians(80), np.radians(90)],
                    grasps_pos_list=[np.array([0, 0, Mm(106) - Mm(15)])],
                    radius=Mm(18.3) / 2, height=Mm(106),
                    color=(0.86, 0.08, 0.24, 1)),
        8: TestTube(name=TUBE_3_NAME,
                    type=8,
                    model_path=fs.workdir_model / "tube3.stl",
                    grasps_path=fs.workdir / "data" / "grasps" / "tube3.json",
                    grasp_angle_list=[np.radians(87), np.radians(90)],
                    grasps_pos_list=[np.array([0, 0, Mm(101) - Mm(8)]),
                                     # np.array([Mm(2), Mm(2), Mm(101) - Mm(12)]),
                                     # np.array([Mm(2), -Mm(2), Mm(101) - Mm(12)]),
                                     # np.array([-Mm(2), Mm(2), Mm(101) - Mm(12)]),
                                     # np.array([Mm(2), -Mm(2), Mm(101) - Mm(12)]),
                                     ],
                    radius=Mm(17.5) / 2, height=Mm(101),
                    color=(0.31, 0.78, 0.47, 1)),
        99: TUBE_DRAW,
        100: TestTube(name='draw_obstacle',
                      type=100,
                      model_path=fs.workdir_model / "20220610_white_tube.stl",
                      grasps_path=fs.workdir / "data" / "grasps" / "draw_tube_grasps.json",
                      radius=Mm(16) / 2, height=Mm(85),
                      grasp_angle_list=[np.radians(85), np.radians(80)],
                      grasps_pos_list=[np.array([0, 0, Mm(107.5) - Mm(15)])],
                      color=(.2, .2, .2, 1)),
    }

    # BLUE_CAP_TUBE = TestTube(name=BLUE_CAP_NAME,
    #                          type=BLUE_CAP_TYPE,
    #                          model_path=fs.workdir_model / "white_cap.stl",
    #                          grasps_path=fs.workdir / "data" / "grasps" / "blue_cap_tube_grasps.json",
    #                          grasp_angle_list=[np.radians(80), np.radians(90)],
    #                          grasps_pos_list=[np.array([0, 0, Mm(106) - Mm(15)])],
    #                          radius=Mm(18.3) / 2, height=Mm(106),
    #                          color=(100 / 255, 100 / 255, 100 / 255, 1))
    #
    # PURPLE_CAP_TUBE = TestTube(name='PURPLE',
    #                            type=6,
    #                            model_path=fs.workdir_model / "20220610_purple_ring_tube.stl",
    #                            grasps_path=fs.workdir / "data" / "grasps" / "purple_ring_cap_tube_grasps.json",
    #                            radius=Mm(18) / 2,
    #                            height=Mm(107.5),
    #                            color=(186 / 255, 85 / 255, 211 / 255, 1),
    #                            grasp_angle_list=[np.radians(85)],
    #                            grasps_pos_list=[np.array([0, 0, Mm(107.5) - Mm(15)])], )
    #
    # WHITE_CAP_TUBE = TestTube(name=WHITE_CAP_NAME,
    #                           type=WHITE_CAP_TYPE,
    #                           # model_path=fs.workdir_model / "tubewhite.stl",
    #                           model_path=fs.workdir_model / "20220610_white_tube.stl",
    #                           grasps_path=fs.workdir / "data" / "grasps" / "white_cap_tube_grasps.json",
    #                           radius=Mm(16) / 2,
    #                           height=Mm(112),
    #                           # color=color_hex2oct("#faf0e6", alpha=.7),
    #                           color=(1, 1, 1, 1),
    #                           # grasp_angle_list=[np.radians(60), np.radians(75), np.radians(90)],
    #                           # grasp_angle_list=[np.radians(82.5), np.radians(90)],
    #                           grasp_angle_list=[np.radians(85)],
    #                           # grasps_pos_list=[np.array([0, 0, Mm(112) - Mm(12)])],
    #                           grasps_pos_list=[np.array([0, 0, Mm(112) - Mm(5)])], )

    ################## UNCOMMENT FOR DEMO ##################

    # BLUE_CAP_TUBE = TestTube(name=BLUE_CAP_NAME,
    #                          type=BLUE_CAP_TYPE,
    #                          # model_path=fs.workdir_model / "tubeblue.stl",
    #                          model_path=fs.workdir_model / "20220610_blue_tube.stl",
    #                          grasps_path=fs.workdir / "data" / "grasps" / "blue_cap_tube_grasps.json",
    #                          # grasp_angle_list=[np.radians(82.5), np.radians(90)],
    #                          grasp_angle_list=[np.radians(90)],
    #                          # grasps_pos_list=[np.array([0, 0, Mm(76) - Mm(6)])],
    #                          grasps_pos_list=[np.array([0, 0, Mm(74)])],
    #                          radius=Mm(15) / 2, height=Mm(76),
    #                          # color=color_hex2oct("#005b96", alpha=.7),
    #                          color=(65 / 255, 105 / 255, 225 / 255, 1))
    #
    # # X
    # # BLUE_CAP_TUBE = TestTube(name=BLUE_CAP_NAME,
    # #                          type=BLUE_CAP_TYPE,
    # #                          model_path=fs.workdir_model / "white_cap.stl",
    # #                          grasps_path=fs.workdir / "data" / "grasps" / "blue_cap_tube_grasps.json",
    # #                          grasp_angle_list=[np.radians(80), np.radians(90)],
    # #                          grasps_pos_list=[np.array([0, 0, Mm(106) - Mm(15)])],
    # #                          radius=Mm(18.3) / 2, height=Mm(106),
    # #                          color=(100 / 255, 100 / 255, 100 / 255, 1))
    #
    # PURPLE_CAP_TUBE = TestTube(name=PURPLE_CAP_NAME,
    #                            type=PURPLE_CAP_TYPE,
    #                            # model_path=fs.workdir_model / "tubepurple.stl",
    #                            model_path=fs.workdir_model / "20220610_purple_tube.stl",
    #                            grasps_path=fs.workdir / "data" / "grasps" / "purple_cap_tube_grasps.json",
    #                            radius=Mm(17) / 2, height=Mm(101),
    #                            # color=color_hex2oct("#8874a3", alpha=.7),
    #                            color=(186 / 255, 85 / 255, 211 / 255, 1),
    #                            # grasp_angle_list=[np.radians(82.5), np.radians(90)],
    #                            grasp_angle_list=[np.radians(85)],
    #                            # grasp_angle_list=[np.radians(75), np.radians(82.5), np.radians(90)],
    #                            grasps_pos_list=[np.array([0, 0, Mm(101) - Mm(9)])],
    #                            # grasps_pos_list=[np.array([0, 0, Mm(101) - Mm(13)])],
    #                            )
    #
    # # PURPLE_CAP_TUBE = TestTube(name=PURPLE_CAP_NAME,
    # #                            type=PURPLE_CAP_TYPE,
    # #                            # model_path=fs.workdir_model / "tubepurple.stl",
    # #                            model_path=fs.workdir_model / "transparent_cap.stl",
    # #                            grasps_path=fs.workdir / "data" / "grasps" / "purple_cap_tube_grasps.json",
    # #                            radius=Mm(17) / 2, height=Mm(110),
    # #                            # color=color_hex2oct("#8874a3", alpha=.7),
    # #                            color=(186 / 255, 85 / 255, 211 / 255, 1),
    # #                            # grasp_angle_list=[np.radians(82.5), np.radians(90)],
    # #                            grasp_angle_list=[np.radians(80), np.radians(90)],
    # #                            # grasp_angle_list=[np.radians(75), np.radians(82.5), np.radians(90)],
    # #                            grasps_pos_list=[np.array([0, 0, Mm(101) - Mm(5)])],
    # #                            # grasps_pos_list=[np.array([0, 0, Mm(101) - Mm(13)])],
    # #                            )
    #
    # # PURPLE_CAP_TUBE._cd_primitive_fn_factory = lambda: purple_cdnp
    #
    # WHITE_CAP_SMALL_TUBE = TestTube(name=WHITE_CAP_SMALL_NAME,
    #                                 type=WHITE_CAP_SMALL_TYPE,
    #                                 model_path=fs.workdir_model / "tubewhitesmall.stl",
    #                                 grasps_path=fs.workdir / "data" / "grasps" / "white_cap_small_tube_grasps.json",
    #                                 radius=Mm(12) / 2, height=Mm(84),
    #                                 color=color_hex2oct("#faf0e6", alpha=.7))
    #
    # # PURPLE_RING_CAP_TUBE = TestTube(name=PURPLE_RING_CAP_NAME,
    # #                                 type=PURPLE_RING_CAP_TYPE,
    # #                                 model_path=fs.workdir_model / "20220610_purple_ring_tube_small.stl",
    # #                                 grasps_path=fs.workdir / "data" / "grasps" / "purple_ring_cap_tube_grasps.json",
    # #                                 radius=Mm(18) / 2,
    # #                                 height=Mm(86),
    # #                                 color=(222 / 255, 222 / 255, 222 / 255, 1),
    # #                                 grasp_angle_list=[np.radians(85)],
    # #                                 grasps_pos_list=[np.array([0, 0, Mm(86) - Mm(8)])], )

    @classmethod
    def get_tubetype_by_name(cls, tube_name):
        """Parse the tube type from string"""
        if "blue" in tube_name:
            return cls.TUBE_TYPE_1
        elif "purple ring" in tube_name:
            return cls.TUBE_TYPE_5
        elif "purple" in tube_name:
            return cls.TUBE_TYPE_2
        elif "white cap small" in tube_name:
            return cls.TUBE_TYPE_4
        elif "white" in tube_name:
            return cls.TUBE_TYPE_3

        else:
            raise TubeTypeNotDefined(tube_name)

    @classmethod
    def gen_tube_by_name(cls, tube_name: Literal["blue cap", "purple cap", "white cap small", "white cap"]):
        if cls.TUBE_1_NAME in tube_name:
            return cls.TUBE_1.copy()
        elif cls.TUBE_2_NAME in tube_name:
            return cls.TUBE_2.copy()
        elif cls.TUBE_3_NAME in tube_name:
            return cls.TUBE_3.copy()
        elif cls.TUBE_4_NAME in tube_name:
            return cls.TUBE_4.copy()
        elif cls.TUBE_5_NAME in tube_name:
            return cls.TUBE_5.copy()
        elif cls.TUBE_FOR_DRAW_NAME in tube_name:
            return cls.TUBE_DRAW
        else:
            raise TubeTypeNotDefined(tube_name)

    @classmethod
    def gen_tube_by_tubetype(cls, tube_type):
        tube_type = int(tube_type)
        if tube_type == cls.TUBE_TYPE_1:
            return cls.TUBE_1.copy()
        elif tube_type == cls.TUBE_TYPE_2:
            return cls.TUBE_2.copy()
        elif tube_type == cls.TUBE_TYPE_3:
            return cls.TUBE_3.copy()
        elif tube_type == cls.TUBE_TYPE_4:
            return cls.TUBE_4.copy()
        elif tube_type == cls.TUBE_TYPE_5:
            return cls.TUBE_5.copy()
        elif tube_type in cls.Tubes:
            return cls.Tubes[tube_type].copy()
        else:
            raise TubeTypeNotDefined(tube_type)

    @classmethod
    def tube_type_color(cls, tube_type):
        if isinstance(tube_type, str):
            tube_type = cls.get_tubetype_by_name(tube_type)
        # name of color can be found at
        # https://matplotlib.org/stable/gallery/color/named_colors.html
        if tube_type == cls.TUBE_TYPE_1:
            tube_color = "orangered"
        elif tube_type == cls.TUBE_TYPE_2:
            tube_color = "mediumorchid"
        elif tube_type == cls.TUBE_TYPE_3:
            tube_color = "royalblue"
        elif tube_type == cls.TUBE_TYPE_4:
            tube_color = "lightgreen"
        elif tube_type == cls.TUBE_TYPE_5:
            tube_color = "violet"
        return tube_color

    @classmethod
    def get_modelpath_by_tubetype(cls, tube_type):
        if tube_type == cls.TUBE_TYPE_1:
            model_path = str(fs.workdir_model / "tubena.stl")
        elif tube_type == cls.TUBE_TYPE_2:
            model_path = str(fs.workdir_model / "")
        elif tube_type == cls.TUBE_TYPE_3:
            model_path = str(fs.workdir_model / "tubebig_capped.stl")
        elif tube_type == cls.TUBE_TYPE_4:
            model_path = str(fs.workdir_model / "")
        return model_path


def gen_bluecap_tube() -> _TestTube:
    return _TestTube(type=TubeType.TUBE_TYPE_1,
                     model_path=fs.workdir_model / "tubeblue.stl",
                     grasps_path=fs.workdir / "data" / "grasps" / "blue_cap_tube_grasps.json",
                     radius=Mm(15) / 2, height=Mm(76),
                     color=color_hex2oct("#005b96", alpha=.7))


def gen_purplecap_tube() -> _TestTube:
    return _TestTube(type=TubeType.TUBE_TYPE_2,
                     model_path=fs.workdir_model / "tubepurple.stl",
                     grasps_path=fs.workdir / "data" / "grasps" / "purple_cap_tube_grasps.json",
                     radius=Mm(17) / 2, height=Mm(100),
                     color=color_hex2oct("#8874a3", alpha=.7))


def gen_whitecap_tube() -> _TestTube:
    return _TestTube(type=TubeType.TUBE_TYPE_3,
                     model_path=fs.workdir_model / "tubewhite.stl",
                     grasps_path=fs.workdir / "data" / "grasps" / "white_cap_tube_grasps.json",
                     radius=Mm(16) / 2, height=Mm(112),
                     color=color_hex2oct("#faf0e6", alpha=.7))


def gen_whitecap_small_tube() -> _TestTube:
    return _TestTube(type=TubeType.TUBE_TYPE_4,
                     model_path=fs.workdir_model / "tubewhitesmall.stl",
                     grasps_path=fs.workdir / "data" / "grasps" / "white_cap_small_tube_grasps.json",
                     radius=Mm(12) / 2, height=Mm(82),
                     color=color_hex2oct("#faf0e6", alpha=.7))


class TestTubePoseAnalyzer(Optimization):

    def __init__(self, pcd, test_tube_radius, tubepos_in_rack):
        """

        Parameters
        ----------
        pcd: points of the test tube
        """
        super(TestTubePoseAnalyzer, self).__init__()
        self.pcd = pcd
        self.test_tube_radius = test_tube_radius
        self.alpha_v = self.variableid
        self.beta_v = self.variableid
        self.tubepos_in_rack = tubepos_in_rack
        self.radius_variant_v = self.variableid
        coeff = 1
        self.radius_variant_bounds = [- test_tube_radius / coeff, test_tube_radius / coeff]
        # self.x0 = [0, 0, test_tube_radius]
        self.x0 = [0, 0]
        _max_tilt_agl = np.deg2rad(5)
        # self.bounds = [[-_max_tilt_agl, _max_tilt_agl], [-_max_tilt_agl, _max_tilt_agl], self.radius_variant_bounds]
        self.bounds = [[-_max_tilt_agl, _max_tilt_agl], [-_max_tilt_agl, _max_tilt_agl], ]

    def gen_loss_func(self, n=1):
        radius = self.test_tube_radius
        points = self.pcd
        a = self.tubepos_in_rack
        alpha_id = self.alpha_v
        beta_id = self.beta_v
        # variance_id = self.radius_variant_v

        def loss_func(x):
            distances = distance_to_line(gen_line(x[alpha_id], x[beta_id])[0], points, a)
            # return np.sum((distances - radius + x[variance_id]) ** 2) * n
            # return (np.sum(distances[(distances - radius)>0])) * n
            return (np.sum((distances - radius) > 0)) * n

        return loss_func

    def get_solution(self):
        loss_func = self.gen_loss_func(n=1e3)
        solution = minimize(loss_func, self.x0, method="Powell", bounds=self.bounds,
                            options={'ftol': 1e-4, "maxiter": 100, "disp": False})
        if solution.status == 0:
            rot = gen_line(solution.x[0], solution.x[1])[1]
            distances = distance_to_line2(gen_line(solution.x[self.alpha_v], solution.x[self.beta_v])[0], self.pcd,
                                          self.tubepos_in_rack)
            # print(distances)
            loss = np.sum((distances - self.test_tube_radius) ** 2) * 1e5 / len(self.pcd)
            # print(len(self.pcd))
            # print(loss)
            return rot, loss
        else:
            raise Exception("Optimization Failed")


class TestTubePoseAnalyzer2(Optimization):

    def __init__(self, pcd, test_tube_radius, tubepos_in_rack):
        """

        Parameters
        ----------
        pcd: points of the test tube
        """
        super(TestTubePoseAnalyzer2, self).__init__()
        self.pcd = pcd
        self.test_tube_radius = test_tube_radius
        self.alpha_v = self.variableid
        self.beta_v = self.variableid
        self.tubepos_in_rack = tubepos_in_rack
        self.radius_variant_v = self.variableid
        coeff = 1
        self.radius_variant_bounds = [- test_tube_radius / coeff, test_tube_radius / coeff]
        self.x0 = [0, 0, test_tube_radius]
        self.bounds = [[-np.pi / 12, np.pi / 12], [-np.pi / 12, np.pi / 12], self.radius_variant_bounds]

    def gen_loss_func(self, n=1):
        radius = self.test_tube_radius
        points = self.pcd
        a = self.tubepos_in_rack
        alpha_id = self.alpha_v
        beta_id = self.beta_v
        variance_id = self.radius_variant_v

        def loss_func(x):
            distances = distance_to_line(gen_line(x[alpha_id], x[beta_id])[0], points, a)
            return np.sum((distances - radius + x[variance_id]) ** 2) * n

        return loss_func

    def get_solution(self):
        loss_func = self.gen_loss_func(n=1e5)
        solution = minimize(loss_func, self.x0, method="Powell", bounds=self.bounds,
                            options={'ftol': 1e-11, "maxiter": 1000, "disp": False})
        if solution.status == 0:
            rot = gen_line(solution.x[0], solution.x[1])[1]
            distances = distance_to_line2(gen_line(solution.x[self.alpha_v], solution.x[self.beta_v])[0], self.pcd,
                                          self.tubepos_in_rack)
            # print(distances)
            loss = np.sum((distances - self.test_tube_radius) ** 2) * 1e5 / len(self.pcd)
            # print(len(self.pcd))
            return rot, loss
        else:
            raise Exception("Optimization Failed")


if __name__ == "__main__":
    print(TubeType.BLUE_CAP_TUBE.height)
