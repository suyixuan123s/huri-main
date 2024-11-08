import copy
import logging
from typing import Union
from itertools import chain
from collections import namedtuple

from shapely.geometry.polygon import Polygon, Point

from basis.trimesh import Trimesh
from modeling import model_collection as mc

from huri.core.common_import import gm, cm, fs, rm, np
import huri.components.task_planning.tube_puzzle_solver as tps
from huri.definitions.tube_def import TubeType, TestTubePoseAnalyzer, TestTube
from huri.math.units import Mm
from huri.definitions.rack_geom import rectangle_polygon, rack_hard_geom, rack_soft_geom, RackGeom, SlotGeom
from huri.components.exe.version.grasp_filter import get_3x3_local_state
from huri.learning.env.rack_v3.utils import get_satisfied_constraints

# TODO encapusulate the program in examples/taskplanning to taskplanning

TaskSol = namedtuple("TaskSol",
                     ["tube",
                      "obs_rack_cm",
                      "obs_tube_cm_collection",
                      "init_homomat",
                      "goal_homomat",
                      "init_slot_id",
                      "goal_slot_id",
                      "grasp_direction",
                      "insert_direction",
                      "goal_rack_status"])

rack_def_logger = logging.getLogger(__name__)


class TubeTypeNotDefined(Exception):
    def __init__(self, tubetype):
        super().__init__(f"Undefined tube type: {tubetype}")


class InsertTubeIndexError(Exception):
    def __init__(self, index_shape):
        super().__init__(f"Error insert index shape: {index_shape}. The dimension of index must be 1x2 ")

class NoSolutionError(Exception):
    def __init__(self, rack_status, goal_pattern):
        super().__init__(f"No solution for the rack status {rack_status} and the goal pattern {goal_pattern}")


mask_ucbc = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
ucbc_dir = np.array([1, 0, 0])
mask_crcl = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])
crcl_dir = np.array([0, 1, 0])
mask_ul = np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]])
ul_dir = rm.unit_vector(np.array([-1, 1, 0]))
mask_ur = np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]])
ur_dir = rm.unit_vector(np.array([-1, -1, 0]))
mask_bl = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]])
bl_dir = rm.unit_vector(np.array([1, 1, 0]))
mask_br = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]])
br_dir = rm.unit_vector(np.array([1, 1, 0]))


class TubeRack:
    def __init__(self,
                 model_path: str,
                 dim=np.array([0, 0]),
                 slot_dim=np.array([0, 0]),
                 slot_center_x_set=np.array([0, 0, 0]),
                 slot_center_y_set=np.array([0, 0, 0]),
                 bottom_height=Mm(3.5),
                 color=(0, .3, 1, 1),
                 geom: RackGeom = None,
                 pcd_template: np.ndarray = None
                 ):
        # the model's path for the rack
        self.model_path = str(model_path)
        # the 2D dimension of the rack (not include the height)
        self.dim = dim
        # the 2D dimension of the slot on the rack
        self.slot_dim = slot_dim
        # the x coordinate of the slot's center
        self.slot_center_x_set = slot_center_x_set
        # the y coordinate of the slot's center
        self.slot_center_y_set = slot_center_y_set
        # the status of the rack
        self.rack_status = np.zeros(self.shape, dtype=int)
        # the confidence of each rack
        self.rack_status_confidence = np.zeros(self.shape, dtype=float)
        # the centers of the slot
        self.slot_centers = np.zeros((*self.shape, 2))
        # polygon representation for the slot
        self.slot_polygons = np.ndarray(shape=self.shape, dtype=object)
        # the height between the bottom of the slot to the bottom of the rack
        self.bottom_height = bottom_height
        # the color of the rack in the simulation
        self.color = color
        for _x_ind, _x in enumerate(self.slot_center_x_set):
            for _y_ind, _y in enumerate(self.slot_center_y_set):
                self.slot_centers[_x_ind, _y_ind] = np.array([_x, _y])
                self.slot_polygons[_x_ind, _y_ind] = rectangle_polygon(np.array([_x, _y]), self.slot_dim)

        self.goal_pattern = np.array([[3, 0, 0, 0, 0, 0, 0, 0, 0, 3],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [3, 0, 0, 0, 0, 0, 0, 0, 0, 3]])
        self.tubes_pose = np.zeros((*self.shape, 4, 4))
        self._rack_tf = np.eye(4)
        self._geom = geom
        self._pcd_template = pcd_template

    @property
    def shape(self):
        return len(self.slot_center_x_set), len(self.slot_center_y_set)

    def insert_tube(self, slot_id: np.ndarray, tube: Union[int, TestTube], tube_rel_pose=None, confidence=None):
        if not isinstance(slot_id, np.ndarray):
            slot_id = np.array(slot_id)
        if slot_id.shape == (1, 2):
            slot_id = slot_id.reshape(2, )
        if slot_id.shape != (2,):
            raise InsertTubeIndexError(slot_id.shape)
        assert 0 <= slot_id[0] < self.shape[0] and 0 <= slot_id[1] < self.shape[1]
        # the slot should be empty
        if confidence is None:
            assert self.rack_status[slot_id[0], slot_id[1]] == 0
        else:
            # print(confidence)
            assert confidence > 0

        tube_type = 0
        if isinstance(tube, int) or isinstance(tube, np.int_) or isinstance(tube, np.int32) or isinstance(tube,
                                                                                                          np.int64):
            tube_type = tube
        elif isinstance(tube, TestTube):
            tube_type = tube.type

        if confidence is not None:
            if self.rack_status_confidence[slot_id[0], slot_id[1]] > confidence:
                # print(f"Cannot insert the test tube due to confidence error, "
                #       f"input confidence {confidence} < the old confidence {self.rack_status_confidence[slot_id[0], slot_id[1]]}")
                return
        self.rack_status_confidence[slot_id[0], slot_id[1]] = confidence
        self.rack_status[slot_id[0], slot_id[1]] = tube_type

        if tube_type != 0:
            self.tubes_pose[slot_id[0], slot_id[1]] = tube_rel_pose if tube_rel_pose is not None \
                else rm.homomat_from_posrot(self.get_slot_bottom_center_by_id(slot_id))

    def get_slot_homomat(self, slot_id: np.ndarray):
        assert 0 <= slot_id[0] < self.shape[0] and 0 <= slot_id[1] < self.shape[1]
        if self.rack_status[slot_id[0], slot_id[1]] > 0:
            return self.convert_tube_homomat_gl(self.tubes_pose[slot_id[0], slot_id[1]])
        else:
            return self.convert_tube_homomat_gl(rm.homomat_from_posrot(self.get_slot_bottom_center_by_id(slot_id)))

    def remove_slot(self, slot_id: np.ndarray):
        if not isinstance(slot_id, np.ndarray):
            slot_id = np.array(slot_id)
        if slot_id.shape == (1, 2):
            slot_id = slot_id.reshape(2, )
        if slot_id.shape != (2,):
            raise InsertTubeIndexError(slot_id.shape)
        assert 0 <= slot_id[0] < self.shape[0] and 0 <= slot_id[1] < self.shape[1]
        if self.rack_status[slot_id[0], slot_id[1]] == 0:
            # TODO add alert when remove the empty slot
            rack_def_logger.warning(
                f"Remove the empty slot! slot index is {slot_id}, matrix is {repr(self.rack_status)} ")
            return None, None
        else:
            tube_type = self.rack_status[slot_id[0], slot_id[1]]
            tube_homomat_gl = self.convert_tube_homomat_gl(self.tubes_pose[slot_id[0], slot_id[1]])
            self.rack_status[slot_id[0], slot_id[1]] = 0
            self.tubes_pose[slot_id[0], slot_id[1]] = np.zeros((4, 4))
            return tube_type, tube_homomat_gl

    def insert_tube_with_pose(self, slot_id, test_tube, test_tube_pos):
        if not isinstance(slot_id, np.ndarray):
            slot_id = np.array(slot_id)
        if slot_id.shape == (1, 2):
            slot_id = slot_id.reshape(2, )
        if slot_id.shape != (2,):
            raise InsertTubeIndexError(slot_id.shape)
        self.rack_status[slot_id[0], slot_id[1]] = test_tube.type
        self.tubes_pose[slot_id[0], slot_id[1]] = test_tube_pos

    def get_local_state(self, center_id):
        return np.pad(self.rack_status, 1)[center_id[0] + 1,
               center_id[1]:center_id[1] + 3].copy()
        # return np.pad(self.rack_status, 1)[center_id[0]:center_id[0] + 3,
        #        center_id[1]:center_id[1] + 3].copy()

    def get_3x3_local_state(self, center_id):
        # return np.pad(self.rack_status, 1)[center_id[0] + 1,
        #        center_id[1]:center_id[1] + 3].copy()
        return np.pad(self.rack_status, 1)[center_id[0]:center_id[0] + 3,
               center_id[1]:center_id[1] + 3].copy()

    def get_local_tube_pose(self, center_id):
        return np.pad(self.tubes_pose, 1)[center_id[0] + 1,
               center_id[1]:center_id[1] + 3, 1:-1, 1:-1].copy()

    def get_local_state_data(self, center_id):
        state = self.get_local_state(center_id)
        tube_pose = np.pad(self.tubes_pose, 1)[center_id[0] + 1,
                    center_id[1]:center_id[1] + 3, 1:-1, 1:-1].copy()
        return (center_id, state, tube_pose)

    def get_grasps_local_state(self, center_id, grasp_info_list):
        local_state = self.get_3x3_local_state(center_id)
        tube_homomat = self.get_slot_homomat(center_id)
        print(f"center {center_id}, tube pose is ", repr(self.tubes_pose[center_id[0], center_id[1]]))
        ucub = np.all(np.logical_xor(local_state, mask_ucbc)[mask_ucbc > 0])
        crcl = np.all(np.logical_xor(local_state, mask_crcl)[mask_crcl > 0])
        ul = np.all(np.logical_xor(local_state, mask_ul)[mask_ul > 0])
        ur = np.all(np.logical_xor(local_state, mask_ur)[mask_ur > 0])
        bl = np.all(np.logical_xor(local_state, mask_bl)[mask_bl > 0])
        br = np.all(np.logical_xor(local_state, mask_br)[mask_br > 0])
        if ucub and crcl and ul and ur and bl and br:
            return grasp_info_list
        grasp_info_list = np.array(grasp_info_list)
        grasp_rot_y_list = tube_homomat[:3, :3].dot(np.array([_[2] for _ in grasp_info_list])[:, :, 0].T).T
        grasp_id_list = np.array([])
        for cnt, c in enumerate((ucub, crcl, ul, ur, bl, br)):
            if c:
                # print(cnt)
                # print(local_state)
                bound_diag = .9
                if cnt == 0:
                    dir = tube_homomat[:3, 0]
                    bounds = [-.9, .9]
                elif cnt == 1:
                    dir = tube_homomat[:3, 1]
                    bounds = [-.9, .9]
                elif cnt == 2:
                    dir = rm.unit_vector(-tube_homomat[:3, 0] + tube_homomat[:3, 1])
                    bounds = [-bound_diag, bound_diag]
                elif cnt == 3:
                    dir = rm.unit_vector(tube_homomat[:3, 0] + tube_homomat[:3, 1])
                    bounds = [-bound_diag, bound_diag]
                elif cnt == 4:
                    dir = rm.unit_vector(tube_homomat[:3, 0] + tube_homomat[:3, 1])
                    bounds = [-bound_diag, bound_diag]
                elif cnt == 5:
                    dir = rm.unit_vector(-tube_homomat[:3, 0] + tube_homomat[:3, 1])
                    bounds = [-bound_diag, bound_diag]
                # gm.gen_arrow(spos=tube_homomat[:3, 3], epos=tube_homomat[:3, 3] + dir * .1,
                #              thickness=.006, rgba=[0, 1, 0, 1]).attach_to(base)
                dir_val = np.dot(grasp_rot_y_list, dir)
                grasp_ids = np.where((dir_val > bounds[1]) | (dir_val < bounds[0]))[0]
                grasp_id_list = np.hstack((grasp_id_list, grasp_ids))
        # for i in grasp_rot_y_list[grasp_id_list.flatten().astype(int)]:
        #     gm.gen_arrow(spos=tube_homomat[:3, 3], epos=tube_homomat[:3, 3] + i * .1).attach_to(base)
        print("Feasible grasping id list = :", grasp_id_list.flatten().astype(int))
        return grasp_info_list[grasp_id_list.flatten().astype(int)]

    def update_local_state_data(self, center_id, local_state, local_tube_pose):
        state = np.pad(self.rack_status, 1)
        tube_pose = np.pad(self.tubes_pose, 1)
        state[center_id[0]:center_id[0] + 3, center_id[1]:center_id[1] + 3] = local_state
        tube_pose[center_id[0]:center_id[0] + 3, center_id[1]:center_id[1] + 3, 1:-1, 1:-1] = local_tube_pose
        self.state = state[1:-1, 1:-1]
        self.tubes_pose = tube_pose[1:-1, 1:-1, 1:-1, 1:-1]

    def solve(self, goal_pattern=None, weight_matrix=None):
        print("SOLVE?")
        # FOR DEMO
        if goal_pattern is None:
            goal_pattern = self.goal_pattern.copy()
        else:
            self.goal_pattern = goal_pattern.copy()
        if not self.rack_status.any():
            print("No Tube")
            return []
        # make sure the goal pattern is larger than other patterns
        tubetypes_in_rack = np.unique(self.rack_status)
        for i in tubetypes_in_rack:
            if i == 0:
                continue
            if len(self.rack_status[self.rack_status == i]) > len(self.goal_pattern[self.goal_pattern == i]):
                print(
                    f"The number of test tube:{len(self.rack_status[self.rack_status == i])}, the number of pattern {len(self.goal_pattern[self.goal_pattern == i])}. Too many test tubes! The goal pattern has no enough space to contain all of them")
                return []
            else:
                print(
                    f"The number of test tube:{len(self.rack_status[self.rack_status == i])}, the number of pattern {len(self.goal_pattern[self.goal_pattern == i])}.")

        id_pattern = np.array([[0, 0, 1, 0, 0, 3, 0, 3, 0, 0],
                               [0, 0, 0, 3, 3, 0, 0, 0, 0, 0],
                               [0, 0, 3, 0, 0, 1, 0, 1, 0, 0],
                               [0, 0, 0, 3, 0, 0, 3, 0, 1, 0],
                               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
        if np.array_equal(self.rack_status, id_pattern):
            tmp = tps.a_sol[-1]
            return tps.a_sol
            solver = tps.TubePuzzle(tmp)
            solver.goalpattern = goal_pattern
            path = solver.atarSearch(weightarray=weight_matrix)
            solution = tps.a_sol[:-1] + [v.grid for v in path]
        else:
            solver = tps.TubePuzzle(self.rack_status)
            solver.goalpattern = goal_pattern
            path = solver.atarSearch(weightarray=weight_matrix)
            solution = [v.grid for v in path]
        return np.asarray(solution)

    def reset(self):
        self.rack_status = np.zeros(self.shape, type=int)
        self.rack_status_confidence = np.zeros(self.shape, type=float)
        self.tubes_pose = np.zeros((*self.shape, 4, 4))

    def set_all_rack_by_label_holepos(self, labels, hole_pos):
        """
        This function will reset the rack
        Parameters
        ----------
        labels
        hole_pos

        Returns
        -------

        """
        self.reset()
        for idx, tube_label in enumerate(labels):
            self.insert_tube(hole_pos[idx], TubeType.get_tubetype_by_name(tube_label))

    def plot_rack(self, plot, rack=None):
        if rack is None:
            rack = self.rack_status

        num_tube_type_goal_pattern = np.max(self.goal_pattern).astype(np.int)
        for i in range(1, num_tube_type_goal_pattern + 1):
            tube_type_goal_pattern_idx = np.where(self.goal_pattern == i)
            if len(tube_type_goal_pattern_idx[0]) < 1:
                continue
            for j in range(len(tube_type_goal_pattern_idx[0])):
                polygon_tmp = self.slot_polygons[tube_type_goal_pattern_idx[0][j]][tube_type_goal_pattern_idx[1][j]]

                plot.fill(*polygon_tmp.exterior.xy, alpha=0.4, fc=TubeType.tube_type_color(i), ec='none')

        # plot border
        for polygon in list(chain.from_iterable(self.slot_polygons)):
            plot.plot(*polygon.exterior.xy, color="silver")

        tubes_index = np.where(rack > 0)
        for idx in range(len(tubes_index[0])):
            tube_position = self.slot_centers[tubes_index[0][idx]][tubes_index[1][idx]]
            tube_type = rack[tubes_index[0][idx], tubes_index[1][idx]]
            tube_color = TubeType.tube_type_color(tube_type)
            tube_polygon = Point(tube_position[0], tube_position[1]).buffer(
                np.min(self.slot_dim) / np.sqrt(2) / 2
            )
            plot.plot(*tube_polygon.exterior.xy, color=tube_color)

        return plot.get_img()

    def set_homomat(self, homomat: np.ndarray):
        assert isinstance(homomat, np.ndarray) and homomat.shape == (4, 4)
        self._rack_tf = homomat

    def get_homomat(self) -> np.ndarray:
        return self._rack_tf

    def get_slot_center_by_id(self, slot_id):
        return self.slot_centers[slot_id[0], slot_id[1]]

    def get_slot_bottom_center_by_id(self, slot_id):
        slot_center = self.get_slot_center_by_id(slot_id)
        slot_bottom_center = np.append(slot_center, self.bottom_height)
        return slot_bottom_center

    def convert_tube_homomat_gl(self, tube_homomat):
        return np.dot(self._rack_tf, tube_homomat)

    def gen_geo_model(self):
        if self._geom is not None:
            cm = self._geom.cm.copy()
            mat4 = self.get_homomat()
            mat4[:3, 3] = mat4[:3, 3] + mat4[:3, 2] * .055
            cm.set_homomat(mat4)
            return cm
        else:
            return None

    def gen_mesh_model(self, gen_tube=False, alpha=None, tube_color_dict=None):
        # plot the test tube rack
        rack_mesh_mdl = gm.GeometricModel(initor=self.model_path)
        rack_mesh_mdl.set_rgba(self.color)
        rack_mesh_mdl.set_homomat(self._rack_tf)
        if alpha is not None:
            rgba = rack_mesh_mdl.get_rgba()
            rack_mesh_mdl.set_rgba([rgba[0], rgba[1], rgba[2], alpha])
        # rack_mesh_mdl.show_localframe()
        if gen_tube:
            # gen the test tube model
            rack_status = self.rack_status
            mesh_model_collection = mc.ModelCollection(name="rack mesh model")
            mesh_model_collection.add_gm(rack_mesh_mdl)
            for ind in np.vstack(np.where(rack_status != 0)).T:
                tube_type = rack_status[ind[0], ind[1]]
                tube_homomat = self.convert_tube_homomat_gl(self.tubes_pose[ind[0], ind[1]])
                tube_prototype = TubeType.gen_tube_by_tubetype(tube_type)
                tube_mesh_mdl = tube_prototype.gen_mesh_model()
                if isinstance(tube_color_dict, dict) and tube_type in tube_color_dict:
                    tube_mesh_mdl.set_rgba(tube_color_dict[tube_type])
                tube_mesh_mdl.set_homomat(tube_homomat)
                if alpha is not None:
                    rgba = tube_mesh_mdl.get_rgba()
                    tube_mesh_mdl.set_rgba([rgba[0], rgba[1], rgba[2], alpha])
                # tube_mesh_mdl.show_localframe()
                mesh_model_collection.add_gm(tube_mesh_mdl)
            return mesh_model_collection
        else:
            return rack_mesh_mdl

    def gen_collision_model(self, gen_tube=False, center=None, toggle_no_tube_pose=False):
        rack_cm_mdl = cm.CollisionModel(initor=self.model_path)
        rack_cm_mdl.set_rgba(self.color)
        rack_cm_mdl.set_homomat(self._rack_tf)
        if gen_tube:
            # gen the test tube collision model
            rack_status = self.rack_status
            tube_cm_model_collection = mc.ModelCollection(name="rack mesh model")
            if center is not None:
                bounds = rack_status.shape
                bounds_x = (0, bounds[0] - 1)
                bounds_y = (0, bounds[1] - 1)
                for x in range(center[0] - 1, center[0] + 2):
                    if x < bounds_x[0] or x > bounds_x[1]:
                        continue
                    for y in range(center[1] - 1, center[1] + 2):
                        if y < bounds_y[0] or y > bounds_y[1]:
                            continue
                        tube_type = rack_status[x, y]
                        if tube_type < 1:
                            continue
                        if toggle_no_tube_pose:
                            tube_pose = np.eye(4)
                        else:
                            tube_pose = self.tubes_pose[x, y]
                        tube_homomat = self.convert_tube_homomat_gl(tube_pose)
                        tube = TubeType.gen_tube_by_tubetype(tube_type)
                        tube_cm_mdl = tube.gen_collision_model()
                        tube_cm_mdl.set_homomat(tube_homomat)
                        tube_cm_model_collection.add_cm(tube_cm_mdl)
                return rack_cm_mdl, tube_cm_model_collection
            else:
                for ind in np.vstack(np.where(rack_status != 0)).T:
                    tube_type = rack_status[ind[0], ind[1]]
                    tube_homomat = self.convert_tube_homomat_gl(self.tubes_pose[ind[0], ind[1]])
                    tube = TubeType.gen_tube_by_tubetype(tube_type)
                    tube_cm_mdl = tube.gen_collision_model()
                    tube_cm_mdl.set_homomat(tube_homomat)
                    tube_cm_model_collection.add_cm(tube_cm_mdl)
            return rack_cm_mdl, tube_cm_model_collection
        else:
            return rack_cm_mdl

    def copy(self):
        return copy.deepcopy(self)


from huri.vision.pnt_utils import RACK_HARD_TEMPLATE, RACK_SOFT_TEMPLATE

# TODO merge slot geom and tube rack
Rack_Hard_Proto = TubeRack(
    # model_path=fs.workdir_model / "tubestand_light_thinner.stl",
    model_path=fs.workdir_model / "rack.stl",
    bottom_height=Mm(0),
    dim=np.array([0.097, 0.192]),
    slot_dim=np.array([0.017, 0.0165]),
    slot_center_x_set=np.array([-0.038, -0.019, 0., 0.019, 0.038]),
    # slot_center_y_set=np.array(
    # [-0.08325, -0.06475, -0.04625, -0.02775, -0.00925, 0.00925, 0.02775, 0.04625, 0.06475, 0.08325]),
    slot_center_y_set=np.array([-0.081, -0.063, -0.045, -0.027, -0.009, 0.009, 0.027, 0.045,
                                0.063, 0.081]),
    color=np.array([0, .3, 1, 1]),
    geom=rack_hard_geom,
    pcd_template=RACK_HARD_TEMPLATE)

Rack_Hard_Proto3x3 = TubeRack(
    # model_path=fs.workdir_model / "tubestand_light_thinner.stl",
    model_path=fs.workdir_model / "rack3x3.stl",
    bottom_height=Mm(0),
    dim=np.array([0.097, 0.192]),
    slot_dim=np.array([0.017, 0.0165]),
    slot_center_x_set=np.array([-0.019, 0., 0.019, ]),
    # slot_center_y_set=np.array(
    # [-0.08325, -0.06475, -0.04625, -0.02775, -0.00925, 0.00925, 0.02775, 0.04625, 0.06475, 0.08325]),
    slot_center_y_set=np.array([-0.018, 0, 0.018]),
    color=np.array([0, .3, 1, 1]),
    geom=rack_hard_geom,
    pcd_template=RACK_HARD_TEMPLATE)

Rack_Soft_Proto = TubeRack(
    model_path=fs.workdir_model / "tubestand_light_thinner.stl",
    dim=np.array([0.097, 0.192]),
    slot_dim=np.array([0.017, 0.0165]),
    slot_center_x_set=np.array([-0.0354, -0.0177, 0., 0.0177, 0.0354]),
    slot_center_y_set=np.array(
        [-0.08325, -0.06475, -0.04625, -0.02775, -0.00925, 0.00925, 0.02775, 0.04625, 0.06475, 0.08325]),
    color=np.array([0, .3, 1, 1]),
    geom=rack_soft_geom,
    pcd_template=RACK_SOFT_TEMPLATE)


class ArrangeTubeSolSeq:
    def __init__(self, rack: TubeRack, solution: np.ndarray, tf: np.ndarray):
        if len(solution) == 0:
            raise NoSolutionError(rack.rack_status, rack.goal_pattern)
        self.rack_tube_pos = rack.tubes_pose.copy()
        self.solutions = solution.copy()
        self.rack = rack
        self.rack.set_homomat(tf)

    def gen_sol_seq(self, step_ind):
        rack_tmp = copy.deepcopy(self.rack)
        solutions = self.solutions.copy()
        total_step_n = len(solutions)
        if step_ind == total_step_n - 1:
            return None
        # figure out the init slot and goal slot
        move_map = solutions[step_ind + 1] - solutions[step_ind]
        init_slot_ind = np.concatenate(np.where(move_map < 0))
        goal_slot_ind = np.concatenate(np.where(move_map > 0))

        rack_tf = rack_tmp.get_homomat()
        print(f"------------------{rack_tf}")
        # remove the tube at init slot
        tube_type, init_homomat_gl = rack_tmp.remove_slot(init_slot_ind)
        # init_homomat_gl = rack_tmp.get_slot_homomat(init_slot_ind)
        # generate the collision model for the obstacle to be moved
        rack_cm_mdl, tube_cm_model_collection = rack_tmp.gen_collision_model(gen_tube=True, toggle_no_tube_pose=True)
        # generate the collision model for the tube to be moved
        # print(tube_type, init_slot_ind, goal_slot_ind)
        tube_moved = TubeType.gen_tube_by_tubetype(tube_type)
        tube_moved.set_homomat(init_homomat_gl)

        # insert the tube to the goal slot
        print(step_ind)
        print(solutions[step_ind + 1] - solutions[step_ind])
        print(solutions[step_ind])
        print(solutions[step_ind + 1])
        print(rack_tmp.rack_status)
        print(goal_slot_ind)
        rack_tmp.insert_tube(slot_id=goal_slot_ind, tube=tube_moved)
        goal_homomat_gl = rack_tmp.get_slot_homomat(goal_slot_ind)
        goal_homomat_gl[:3, :3] = init_homomat_gl[:3, :3]

        # grasp direction
        local_3x3 = get_3x3_local_state(rack_tmp.rack_status, init_slot_ind)
        satisfied_cons_mask = get_satisfied_constraints(local_3x3)
        u_dir = rack_tf[:3, 0]
        d_dir = -rack_tf[:3, 0]
        l_dir = rack_tf[:3, 1]
        r_dir = -rack_tf[:3, 1]
        ul_dir = rm.unit_vector(u_dir + l_dir)
        ur_dir = rm.unit_vector(u_dir + r_dir)
        dl_dir = rm.unit_vector(d_dir + l_dir)
        dr_dir = rm.unit_vector(d_dir + r_dir)
        z_dir = rack_tf[:3, 2]
        lean_factor = 0.15
        num_of_satisfied_cons_mask = np.count_nonzero(satisfied_cons_mask)
        if num_of_satisfied_cons_mask == 1:
            constraint = np.argwhere(satisfied_cons_mask).item()
            if constraint == 0:
                grasp_dir = z_dir
            if constraint == 1:
                grasp_dir = z_dir
            elif constraint == 2:
                grasp_dir = rm.unit_vector(dr_dir * lean_factor + z_dir)
            elif constraint == 3:
                grasp_dir = rm.unit_vector(dl_dir * lean_factor + z_dir)
            elif constraint == 4:
                grasp_dir = rm.unit_vector(ur_dir * lean_factor + z_dir)
            elif constraint == 5:
                grasp_dir = rm.unit_vector(ul_dir * lean_factor + z_dir)
        elif num_of_satisfied_cons_mask == 6:
            grasp_dir = z_dir
        else:
            local_3x3_obstacle = np.zeros_like(local_3x3)
            local_3x3_obstacle[local_3x3 > 0] = 1
            u, d, l, r = local_3x3_obstacle[0, 1], local_3x3_obstacle[2, 1], local_3x3_obstacle[1, 0], \
                local_3x3_obstacle[1, 2]
            ul, ur, dl, dr = local_3x3_obstacle[0, 0], local_3x3_obstacle[0, 2], local_3x3_obstacle[2, 0], \
                local_3x3_obstacle[2, 2]
            dir = u * u_dir + d * d_dir + l * l_dir + r * r_dir + ul * ul_dir + ur * ur_dir + dl * dl_dir + dr * dr_dir
            if np.linalg.norm(dir) > 1e-5:
                grasp_dir = rm.unit_vector(rm.unit_vector(dir) * lean_factor + z_dir)
            else:
                grasp_dir = z_dir
            # dir = u * u_dir + d * d_dir + l * l_dir + r * r_dir + ul * ul_dir + ur * ur_dir + dl * dl_dir + dr * dr_dir
            # dir = u * u_dir + d * d_dir + l * l_dir + r * r_dir
            # if np.linalg.norm(dir) > 1e-5:
            #     grasp_dir = rm.unit_vector(rm.unit_vector(dir) * lean_factor + z_dir)
            # else:
            #     grasp_dir = z_dir

        # insert direction
        local_3x3 = get_3x3_local_state(solutions[step_ind + 1], goal_slot_ind)
        satisfied_cons_mask = get_satisfied_constraints(local_3x3)
        lean_factor = .03
        num_of_satisfied_cons_mask = np.count_nonzero(satisfied_cons_mask)
        if num_of_satisfied_cons_mask == 1:
            constraint = np.argwhere(satisfied_cons_mask).item()
            if constraint == 0:
                insert_dir = z_dir
            if constraint == 1:
                insert_dir = z_dir
            elif constraint == 2:
                insert_dir = rm.unit_vector(dr_dir * lean_factor + z_dir)
            elif constraint == 3:
                insert_dir = rm.unit_vector(dl_dir * lean_factor + z_dir)
            elif constraint == 4:
                insert_dir = rm.unit_vector(ur_dir * lean_factor + z_dir)
            elif constraint == 5:
                insert_dir = rm.unit_vector(ul_dir * lean_factor + z_dir)
        elif num_of_satisfied_cons_mask == 6:
            insert_dir = z_dir
        else:
            local_3x3_obstacle = np.zeros_like(local_3x3)
            local_3x3_obstacle[local_3x3 > 0] = 1
            u, d, l, r = local_3x3_obstacle[0, 1], local_3x3_obstacle[2, 1], local_3x3_obstacle[1, 0], \
                local_3x3_obstacle[1, 2]

            x_dir = rack_tf[:3, 0]
            y_dir = rack_tf[:3, 1]
            z_dir = rack_tf[:3, 2]
            # if bool(u) and bool(d) and not bool(l) and not bool(r):
            #     insert_dir = rm.unit_vector(y_dir * lean_factor + z_dir)
            # elif not bool(u) and not bool(d) and bool(l) and bool(r):
            #     insert_dir = rm.unit_vector(x_dir * lean_factor + z_dir)
            # else:
            # insert_dir = (u * x_dir - d * x_dir + l * y_dir - r * y_dir) * lean_factor + z_dir
            local_3x3_obstacle = np.zeros_like(local_3x3)
            local_3x3_obstacle[local_3x3 > 0] = 1
            u, d, l, r = local_3x3_obstacle[0, 1], local_3x3_obstacle[2, 1], local_3x3_obstacle[1, 0], \
                local_3x3_obstacle[1, 2]
            ul, ur, dl, dr = local_3x3_obstacle[0, 0], local_3x3_obstacle[0, 2], local_3x3_obstacle[2, 0], \
                local_3x3_obstacle[2, 2]
            dir = u * u_dir + d * d_dir + l * l_dir + r * r_dir + ul * ul_dir + ur * ur_dir + dl * dl_dir + dr * dr_dir
            if np.linalg.norm(dir) > 1e-5:
                insert_dir = rm.unit_vector(rm.unit_vector(dir) * lean_factor + z_dir)
            else:
                insert_dir = z_dir

        # generate the infomation for each step
        # TODO: consider the return values of the TaskSol (Should it include the collision model?)
        sol = TaskSol(tube=tube_moved,
                      obs_rack_cm=rack_cm_mdl,
                      obs_tube_cm_collection=tube_cm_model_collection,
                      init_homomat=init_homomat_gl,
                      goal_homomat=goal_homomat_gl,
                      init_slot_id=init_slot_ind,
                      goal_slot_id=goal_slot_ind,
                      grasp_direction=-grasp_dir,
                      insert_direction=-insert_dir,
                      goal_rack_status=solutions[step_ind + 1])
        return sol

    def update_rack(self, new_rack: TubeRack):
        self.rack = new_rack
        self.rack_tube_pos = new_rack.tubes_pose.copy()

    def update_rack_pose(self, new_rack: TubeRack):
        self.rack.set_homomat(new_rack.get_homomat())
        # self.rack_tube_pos = new_rack.tubes_pose.copy()

    def get_last_rack_status(self):
        return self.solutions[self.step_cnt - 1]

    def __len__(self):
        return len(self.solutions)

    def __iter__(self):
        self.step_cnt = 0
        return self

    def __next__(self) -> TaskSol:
        if self.step_cnt >= len(self.solutions) - 1:
            raise StopIteration
        r = self.gen_sol_seq(self.step_cnt)
        if r is None:
            raise Exception("Solution error")
        self.step_cnt += 1
        return r

    @property
    def current(self):
        return self.gen_sol_seq(self.step_cnt - 1)


class TubeLocator:
    """Class to determine the place of the tube on the rack.

    Attributes:
        rack_dim: the dimension of the tube stand
        rack_slot_dim: the dimension of the tube stand slot
        rack_slot_centers: the coordinate of the centers of all tube rack slots
        rack_slot_polygons: the polygon representation of all tube rack slots
        tube_rack_tf: the transformation of the tube rack
    """

    def __init__(self, tube_rack: TubeRack, tube_rack_tf=np.eye(3)):
        """
        Initialization of the test tube locator
        Parameters:
            tube_rack: the definition of the tube rack. (definitions/tuberack_def.py)
            tube_rack_tf: the transformation of the tube rack
        """
        self.rack = tube_rack
        self.rack_dim = tube_rack.dim
        self.rack_shape = tube_rack.shape
        self.rack_slot_dim = tube_rack.slot_dim
        self.rack_slot_centers = tube_rack.slot_centers
        self.rack_slot_polygons = tube_rack.slot_polygons
        self.rack_tf = tube_rack_tf

    @property
    def rack_tf(self) -> np.ndarray:
        return self.rack.get_homomat()

    @rack_tf.setter
    def rack_tf(self, homomat: np.ndarray):
        self.rack.set_homomat(homomat)

    def plot_rack_grids(self, canvas):
        for polygon in list(chain.from_iterable(self.rack_slot_polygons)):
            canvas.plot(*polygon.exterior.xy, color=".8")

    def locate_tube_coord_from_pcd(self, tube_pcd_racktf, tube_label, detect_pos=False, canvas=None):
        """Find coordinate of the test tube on the tube rack from point cloud data"""
        # tube point cloud under tube rack coord
        # tube_pcd_racktf = rm.homomat_transform_points(homomat=rm.homomat_inverse(homomat=self.rack_tf),
        #                                               points=tube_pcd)

        # find out the convex hull of the tube and project to the tube rack slot
        tube_pcd_racktf_tm = Trimesh(vertices=tube_pcd_racktf)  # trimesh of the pcd under rack tf
        # TODO check if this revision has bug or not
        # tube_pcd_racktf_hull = tube_pcd_racktf_tm.convex_hull.vertices  # convex hull of the pcd under rack tf
        tube_pcd_racktf_hull = tube_pcd_racktf_tm.vertices  # convex hull of the pcd under rack tf
        tube_pcd_racktf_hull_2d = tube_pcd_racktf_hull[:, :2]
        tube_pcd_racktf_polygon = Polygon(tube_pcd_racktf_hull_2d).convex_hull
        xmin_tube_pcd_polygon, xmax_tube_pcd_polygon = np.min(tube_pcd_racktf_hull_2d[:, 0]), \
            np.max(tube_pcd_racktf_hull_2d[:, 0])
        ymin_tube_pcd_polygon, ymax_tube_pcd_polygon = np.min(tube_pcd_racktf_hull_2d[:, 1]), \
            np.max(tube_pcd_racktf_hull_2d[:, 1])
        # possible coordinate of the tube
        tube_coord_range_x = range(
            max(int(xmin_tube_pcd_polygon / self.rack_slot_dim[0] + self.rack_shape[0] / 2), 0),
            int(min(np.round(xmax_tube_pcd_polygon / self.rack_slot_dim[0] + self.rack_shape[0] / 2) + 1,
                    self.rack_shape[0]))
        )
        tube_coord_range_y = range(
            min(max(int(ymin_tube_pcd_polygon / self.rack_slot_dim[1] + self.rack_shape[1] / 2), 0),
                self.rack_shape[1] - 1),
            int(min(np.round(ymax_tube_pcd_polygon / self.rack_slot_dim[1] + self.rack_shape[1] / 2) + 1,
                    self.rack_shape[1]))
        )
        # the polygon of rack slots inside the possible coordinate range
        rack_slot_polygons = {}
        for _x_id in tube_coord_range_x:
            for _y_id in tube_coord_range_y:
                rack_slot_polygons[(_x_id, _y_id)] = self.rack_slot_polygons[_x_id, _y_id]
        # plot the relationship between tube and rack
        if canvas is not None:
            if canvas.is_plot_grid:
                if not hasattr(canvas, "plot_grid"):
                    canvas.plot_grid = True
                    self.plot_rack_grids(canvas=canvas)
            color = TubeType.tube_type_color(tube_type=tube_label)
            canvas.plot(*tube_pcd_racktf_polygon.exterior.xy, color=color)
        # find out the possible coordinate and its probability
        possible_coord = []
        possible_coord_prob = []  # the probability of tube in different slots
        possible_coord_prob_confidence = []  # the confidence of the tube in the slot
        tubepcd_racktf_polygon_area = tube_pcd_racktf_polygon.area
        for rack_slot_coord, rack_slot_polygon in rack_slot_polygons.items():
            if tube_pcd_racktf_polygon.intersects(rack_slot_polygon):
                prob = tube_pcd_racktf_polygon.intersection(rack_slot_polygon).area / tubepcd_racktf_polygon_area
                # if prob > 0.1:
                possible_coord.append(rack_slot_coord)
                possible_coord_prob.append(prob)
                possible_coord_prob_confidence.append(prob * tubepcd_racktf_polygon_area / rack_slot_polygon.area)
                # uncommet to debug
                # else:
                #    print(id(tube_label), prob)
        # confident slot coordinate
        if len(possible_coord) < 1:
            return None, None, None
        confident_slot_coord = possible_coord[np.argmax(possible_coord_prob)]
        confident_slot_confidence = possible_coord_prob_confidence[np.argmax(possible_coord_prob)]
        # TODO: use the border constraints of the slot to find the position of the test tube
        slot_bottom_coord = self.rack.get_slot_bottom_center_by_id(slot_id=confident_slot_coord)
        # slot_bottom_coord_global = rm.homomat_transform_points(self.rack_tf, slot_bottom_coord)
        tube_proto_type = TubeType.gen_tube_by_tubetype(tube_label)
        # pose_analyer = TestTubePoseAnalyzer(pcd=tube_pcd,
        #                                     test_tube_radius=tube_proto_type.radius,
        #                                     tubepos_in_rack=hole_bottom_coord_global)
        if detect_pos:
            pose_analyer = TestTubePoseAnalyzer(pcd=tube_pcd_racktf, test_tube_radius=tube_proto_type.radius,
                                                tubepos_in_rack=slot_bottom_coord)
            try:
                tube_rack_pose_rot, _ = pose_analyer.get_solution()
            except Exception as e:
                print(f"Exception: {e}")
                tube_rack_pose_rot = np.eye(3)
        else:
            tube_rack_pose_rot = np.eye(3)
        tube_rack_pose = rm.homomat_from_posrot(slot_bottom_coord, tube_rack_pose_rot)
        tube_pose = np.dot(self.rack_tf, tube_rack_pose)
        return tube_pose, confident_slot_coord, confident_slot_confidence


if __name__ == "__main__":
    raise TubeTypeNotDefined("TEST")
