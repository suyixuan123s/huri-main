__VERSION__ = "0.0.1_seperate"

import time
from collections import namedtuple

from huri.core.common_import import wd, ym, fs, np
from huri.definitions.rack_def import TubeRack
from huri.definitions.utils_structure import MotionElement, MotionBatch, MotionBatchPPP
from huri.definitions.rack_def import TaskSol
from huri.components.task_planning.tube_puzzle_learning_solver import DQNSolver
from huri.examples.task_planning.a_star import TubePuzzle
import copy
from huri.definitions.tube_def import TubeType
from huri.core.print_tool import text_pd, print_with_border

TaskSol = namedtuple("TaskSol",
                     ["tube", "obs_rack_cm", "obs_tube_cm_collection",
                      "init_homomat", "goal_homomat", "init_slot_id", "goal_slot_id",
                      "goal_rack_status",
                      "rack_tf_from", "rack_tf_to",
                      "rack_from_id", "rack_to_id"])

# GOAL_PATTERN = np.array([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
#                          [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
#                          [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
#                          [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
#                          [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]])

S_GOAL_PATTERN = np.vstack([np.ones((5, 10), dtype=np.int) * i for i in [2, 1, 3, 0]])
GOAL_PATTERN = np.array([[2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                         [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                         [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


class ArrangeTubeSolSeq:
    def __init__(self, rack_1: TubeRack, rack_2: TubeRack, rack_3: TubeRack, rack_4: TubeRack, solution: np.ndarray):
        if len(solution) == 0:
            raise Exception("Task planning has no solution")
        self.solutions = solution.copy()
        self.rack_1 = rack_1.copy()
        self.rack_2 = rack_2.copy()
        self.rack_3 = rack_3.copy()
        self.rack_4 = rack_4.copy()

    def gen_sol_seq(self, step_ind):
        solutions = self.solutions.copy()
        total_step_n = len(solutions)
        if step_ind == total_step_n - 1:
            return None
        # figure out the init slot and goal slot
        # TODO design the way to move the test tube
        move_map = solutions[step_ind + 1] - solutions[step_ind]
        init_slot_ind = np.concatenate(np.where(move_map < 0))
        goal_slot_ind = np.concatenate(np.where(move_map > 0))

        # from rack id
        from_rack_id = int(init_slot_ind[0] / 5)
        init_slot_ind_new = init_slot_ind.copy()
        init_slot_ind_new[0] = init_slot_ind[0] % 5
        rack_tmp = [self.rack_1, self.rack_2, self.rack_3, self.rack_4][from_rack_id].copy()
        # to rack id
        to_rack_id = int(goal_slot_ind[0] / 5)
        goal_slot_ind_new = goal_slot_ind.copy()
        goal_slot_ind_new[0] = goal_slot_ind[0] % 5
        to_rack_tmp = [self.rack_1, self.rack_2, self.rack_3, self.rack_4][to_rack_id].copy()

        print(f"------------------{rack_tmp.get_homomat()}")
        # remove the tube at init slot
        tube_type, init_homomat_gl = rack_tmp.remove_slot(init_slot_ind_new)
        # init_homomat_gl = rack_tmp.get_slot_homomat(init_slot_ind)
        # generate the collision model for the obstacle to be moved
        rack_cm_mdl, tube_cm_model_collection = rack_tmp.gen_collision_model(gen_tube=True)
        for _ in [self.rack_2, self.rack_3, self.rack_4]:
            rack_cm, tube_mc = _.gen_collision_model(gen_tube=True)
            tube_cm_model_collection._cm_list.append(rack_cm)
            tube_cm_model_collection._cm_list += tube_mc.cm_list
        # generate the collision model for the tube to be moved
        print(tube_type)
        tube_moved = TubeType.gen_tube_by_tubetype(tube_type)
        tube_moved.set_homomat(init_homomat_gl)

        # insert the tube to the goal slot
        to_rack_tmp.insert_tube(slot_id=goal_slot_ind_new, tube=tube_moved)
        goal_homomat_gl = to_rack_tmp.get_slot_homomat(goal_slot_ind_new)
        goal_homomat_gl[:3, :3] = init_homomat_gl[:3, :3]

        # generate the infomation for each step
        # TODO: consider the return values of the TaskSol (Should it include the collision model?)
        sol = TaskSol(tube=tube_moved,
                      obs_rack_cm=rack_cm_mdl,
                      obs_tube_cm_collection=tube_cm_model_collection,
                      init_homomat=init_homomat_gl,
                      goal_homomat=goal_homomat_gl,
                      init_slot_id=init_slot_ind,
                      goal_slot_id=goal_slot_ind,
                      goal_rack_status=solutions[step_ind + 1],
                      rack_tf_from=rack_tmp.get_homomat(),
                      rack_tf_to=to_rack_tmp.get_homomat(),
                      rack_from_id=from_rack_id,
                      rack_to_id=to_rack_id, )
        return sol

    def update_rack(self, new_rack_1: TubeRack, new_rack_2: TubeRack, new_rack_3: TubeRack, new_rack_4: TubeRack, ):
        self.rack_1 = new_rack_1
        self.rack_2 = new_rack_2
        self.rack_3 = new_rack_3
        self.rack_4 = new_rack_4

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


def task_planning(tube_rack_1: TubeRack,
                  tube_rack_2: TubeRack,
                  tube_rack_3: TubeRack,
                  tube_rack_4: TubeRack,
                  infeasible_dict=None) -> ArrangeTubeSolSeq:
    rack_status = np.vstack([_.rack_status for _ in [tube_rack_1, tube_rack_2, tube_rack_3, tube_rack_4]])
    a = TubePuzzle(elearray=rack_status, goalpattern=GOAL_PATTERN)
    if infeasible_dict is None:
        infeasible_dict = {}
    is_find, result = a.atarSearch(infeasible_dict=infeasible_dict)
    if is_find:
        task_sols = ArrangeTubeSolSeq(rack_1=tube_rack_1,
                                      rack_2=tube_rack_2,
                                      rack_3=tube_rack_3,
                                      rack_4=tube_rack_4,
                                      solution=result, )
    else:
        raise Exception("No results")
    return task_sols


def main(open_jaw_width=.034,
         depth_sensor_debug_path=None,
         debug=False,
         retry_num=3,  # number of retry when grasping failed
         goal_place_offset_dis=.04,
         is_vision_feedback=True,
         toggle_save=False, ):
    # init task planning solver
    task_solver = DQNSolver()
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    ## Motion planning init
    motion_batch = MotionBatch()
    grasp_cache_dict = {}  # restore grasps
    infeasible_info_dict = {}  # restore infeasible action during task planning
    infeasible_local_pattern = {}
    rack_state_history = []
    bad_status_weight_matrix = {}

    tube_rack_1, tube_rack_2, tube_rack_3, tube_rack_4 = fs.load_pickle("rack.pkl")

    task_sols = task_planning(tube_rack_1=tube_rack_1,
                              tube_rack_2=tube_rack_2,
                              tube_rack_3=tube_rack_3,
                              tube_rack_4=tube_rack_4,
                              infeasible_dict=infeasible_info_dict, )
    sol_length = len(task_sols)
    for sol_id, sol in enumerate(task_sols):
        print(f"------- Remaining unsolved path length is {sol_length - sol_id - 1} -------")
        print_with_border("Rack State matrix", width=38)
        print(text_pd(tube_rack_1.rack_status))
        # generate and execute the pick and place motions
        common_grasp_iter_info = None
        is_replanning_flag = False
        rack_state_history.append(tube_rack_1.rack_status.copy())
        # moved tube
        moved_tube = sol.tube
        moved_tube_cm = moved_tube.gen_collision_model()
        # get grasp of moved tube
        if grasp_cache_dict.get(moved_tube.type, None) is None:
            grasp_cache_dict[moved_tube.type] = fs.load_json(path=moved_tube.grasps_path)
        grasp_info_list = grasp_cache_dict[moved_tube.type]

        # collision model of the moved tube
        init_slot_id = sol.init_slot_id
        goal_slot_id = sol.goal_slot_id
        for i in sol.obs_tube_cm_collection.cm_list:
            i.attach_to(base)
            i.show_cdprimit()
        base.run()
        print(sol.obs_tube_cm_collection.cm_list)
        print("?")


if __name__ == '__main__':
    main(open_jaw_width=.034,
         debug=False,
         depth_sensor_debug_path=None,
         retry_num=3,
         goal_place_offset_dis=.04,
         is_vision_feedback=True,
         toggle_save=False)
