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
from huri.definitions.rack_geom import rectangle_polygon, rack_hard_geom, rack_soft_geom, RackGeom

from huri.definitions.rack_def import TubeRack

TaskSol = namedtuple("TaskSol",
                     ["tube", "obs_rack_cm", "obs_tube_cm_collection",
                      "init_homomat", "goal_homomat", "init_slot_id", "goal_slot_id",
                      "grasp_direction",
                      "insert_direction",
                      "init_local_geometry",
                      "goal_rack_status"])

TestTube.tube_collision_sz = .001


class ArrangeTubeSolSeq:
    def __init__(self, rack: TubeRack, solution: np.ndarray, tf: np.ndarray):
        if len(solution) == 0:
            raise Exception("Task planning has no solution")
        self.rack_tube_pos = rack.tubes_pose.copy()
        self.solutions = solution.copy()
        self.rack = rack
        self.tf = tf.copy()

    def gen_sol_seq(self, step_ind):
        rack_tmp = self.rack.copy()
        rack_tmp.set_homomat(self.tf)
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
        rack_cm_mdl, tube_cm_model_collection = rack_tmp.gen_collision_model(gen_tube=True)
        # generate the collision model for the tube to be moved
        print(tube_type, init_slot_ind, goal_slot_ind)
        tube_moved = TubeType.gen_tube_by_tubetype(tube_type)
        tube_moved.set_homomat(init_homomat_gl)

        # insert the tube to the goal slot
        print(solutions[step_ind + 1] - solutions[step_ind])
        print(solutions[step_ind])
        print(solutions[step_ind + 1])
        print(rack_tmp.rack_status)
        rack_tmp.insert_tube(slot_id=goal_slot_ind, tube=tube_moved)
        goal_homomat_gl = rack_tmp.get_slot_homomat(goal_slot_ind)
        goal_homomat_gl[:3, :3] = init_homomat_gl[:3, :3]

        # grasp direction
        init_local_geometry = np.pad(rack_tmp.rack_status, 1)[init_slot_ind[0]:init_slot_ind[0] + 3,
                              init_slot_ind[1]:init_slot_ind[1] + 3].copy()
        init_local_geometry[init_local_geometry > 0] = 1
        u, d, l, r = init_local_geometry[0, 1], init_local_geometry[2, 1], init_local_geometry[1, 0], \
            init_local_geometry[1, 2]
        ul, ur, dl, dr = init_local_geometry[0, 0], init_local_geometry[0, 2], init_local_geometry[2, 0], \
            init_local_geometry[2, 2]

        u_dir = rack_tf[:3, 0]
        d_dir = -rack_tf[:3, 0]
        l_dir = rack_tf[:3, 1]
        r_dir = -rack_tf[:3, 1]
        ul_dir = rm.unit_vector(u_dir + l_dir)
        ur_dir = rm.unit_vector(u_dir + r_dir)
        dl_dir = rm.unit_vector(d_dir + l_dir)
        dr_dir = rm.unit_vector(d_dir + r_dir)

        z_dir = rack_tf[:3, 2]
        lean_factor = .1
        dir = u * u_dir + d * d_dir + l * l_dir + r * r_dir + ul * ul_dir + ur * ur_dir + dl * dl_dir + dr * dr_dir
        # dir = u * u_dir + d * d_dir + l * l_dir + r * r_dir
        if np.linalg.norm(dir) > 1e-5:
            grasp_dir = rm.unit_vector(rm.unit_vector(dir) * lean_factor + z_dir)
        else:
            grasp_dir = z_dir

        # insert direction
        local_geometry = np.pad(rack_tmp.rack_status, 1)[goal_slot_ind[0]:goal_slot_ind[0] + 3,
                         goal_slot_ind[1]:goal_slot_ind[1] + 3].copy()
        local_geometry[local_geometry > 0] = 1
        u, d, l, r = local_geometry[0, 1], local_geometry[2, 1], local_geometry[1, 0], local_geometry[1, 2]

        x_dir = rack_tf[:3, 0]
        y_dir = rack_tf[:3, 1]
        z_dir = rack_tf[:3, 2]
        lean_factor = .1

        if bool(u) and bool(d) and not bool(l) and not bool(r):
            insert_dir = rm.unit_vector(y_dir * lean_factor + z_dir)
        elif not bool(u) and not bool(d) and bool(l) and bool(r):
            insert_dir = rm.unit_vector(x_dir * lean_factor + z_dir)
        else:
            insert_dir = rm.unit_vector((u * x_dir - d * x_dir + l * y_dir - r * y_dir) * lean_factor + z_dir)

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
                      init_local_geometry=init_local_geometry,
                      goal_rack_status=solutions[step_ind + 1])
        return sol

    def update_rack(self, new_rack: TubeRack):
        self.rack = new_rack
        self.rack_tube_pos = new_rack.tubes_pose.copy()

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
