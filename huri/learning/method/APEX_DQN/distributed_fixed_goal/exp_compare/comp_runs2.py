""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20231204osaka

"""
import time

from huri.components.task_planning.tube_puzzle_learning_solver import D3QNSolver, AStarSolver
import huri.core.file_sys as fs
from tqdm import tqdm
import numpy as np


V = 30

def load_comparasion_data(path: fs.Path):
    p = fs.Path(path)
    goal_pattern_path = p.joinpath('goal_pattern.pkl')
    state_list_path = p.joinpath('state_list.pkl')
    if not (goal_pattern_path.exists() and state_list_path.exists()):
        raise Exception("Examine the files, goal_pattern.pkl or state_list.pkl is missing")

    goal_pattern = fs.load_pickle(goal_pattern_path)
    state_list = fs.load_pickle(state_list_path)
    d3qn_solver = D3QNSolver(goal_pattern)
    astar_solver = AStarSolver(goal_pattern)
    return goal_pattern, state_list, d3qn_solver, astar_solver


def run_comps(goal_pattern, state_list, d3qn_solver: D3QNSolver, astar_solver: AStarSolver, seg_len:int = 100):
    goal_pattern = np.asarray(goal_pattern)
    num_class = len(np.unique(goal_pattern[goal_pattern > 0]))
    d3qn_solver_solution_dict_noAstarrefiner = {}
    d3qn_solver_solution_dict = {}
    astar_solver_solution_dict = {}
    # ------------------------------------
    d3qn_solver_time_dict_noAstarrefiner = {}
    d3qn_solver_time_dict = {}
    astar_solver_time_dict = {}

    d3qn_solver_noAstarrefiner_success_inds_dict = {}
    d3qn_solver_success_inds_dict = {}
    astar_solver_success_inds_dict = {}

    seg_len_counter = 1
    for training_level, states in state_list.items():
        if training_level != V:
            continue
        d3qn_solver_solution_dict_noAstarrefiner[training_level] = []
        d3qn_solver_solution_dict[training_level] = []
        astar_solver_solution_dict[training_level] = []

        d3qn_solver_time_dict_noAstarrefiner[training_level] = []
        d3qn_solver_time_dict[training_level] = []
        astar_solver_time_dict[training_level] = []

        d3qn_solver_noAstarrefiner_success_inds_dict[training_level] = []
        d3qn_solver_success_inds_dict[training_level] = []
        astar_solver_success_inds_dict[training_level] = []

        print(f"Start Training Level {training_level}/{len(state_list)}")
        for s_id, s in tqdm(enumerate(states), total=len(states)):
            a = time.time()
            path = d3qn_solver.solve(s, toggle_Astar_refiner=False)[0]
            b = time.time()
            if len(path) > 0:
                d3qn_solver_noAstarrefiner_success_inds_dict[training_level].append(s_id)
            d3qn_solver_time_dict_noAstarrefiner[training_level].append(b - a)
            d3qn_solver_solution_dict_noAstarrefiner[training_level].append(path)
            ##---------------------------
            a = time.time()
            path = d3qn_solver.solve(s, toggle_Astar_refiner=True)[1]
            b = time.time()
            if len(path) > 0:
                d3qn_solver_success_inds_dict[training_level].append(s_id)
            d3qn_solver_time_dict[training_level].append(b - a)
            d3qn_solver_solution_dict[training_level].append(path)
            ##---------------------------
            a = time.time()
            path = astar_solver.solve(s, max_iter=1500)
            b = time.time()
            if len(path) > 0:
                astar_solver_success_inds_dict[training_level].append(s_id)
            astar_solver_time_dict[training_level].append(b - a)
            astar_solver_solution_dict[training_level].append(path)
            ##---------------------------

        if training_level >= seg_len*seg_len_counter or training_level == max(state_list.keys()):
            seg_len_counter+= 1
            training_level_up_index = min(d3qn_solver_solution_dict_noAstarrefiner.keys())
            save_path = fs.Path(f'data/{goal_pattern.shape[0]}_'
                                f'{goal_pattern.shape[1]}_'
                                f'{num_class}/eval_results'
                                f'_{training_level_up_index}_{training_level}.pkl')
            fs.dump_pickle([d3qn_solver_solution_dict_noAstarrefiner,
                            d3qn_solver_solution_dict,
                            astar_solver_solution_dict,
                            d3qn_solver_time_dict_noAstarrefiner,
                            d3qn_solver_time_dict,
                            astar_solver_time_dict,
                            d3qn_solver_noAstarrefiner_success_inds_dict,
                            d3qn_solver_success_inds_dict,
                            astar_solver_success_inds_dict,
                            ], save_path)
            print(f"Save segmentation data from training level {training_level_up_index}-{training_level}."
                  f"Save the file to {save_path}")
            # ------------------------------------
            d3qn_solver_solution_dict_noAstarrefiner = {}
            d3qn_solver_solution_dict = {}
            astar_solver_solution_dict = {}
            # ------------------------------------
            d3qn_solver_time_dict_noAstarrefiner = {}
            d3qn_solver_time_dict = {}
            astar_solver_time_dict = {}
            # ------------------------------------
            d3qn_solver_noAstarrefiner_success_inds_dict = {}
            d3qn_solver_success_inds_dict = {}
            astar_solver_success_inds_dict = {}
            # ------------------------------------
    print(f"Finished running")


if __name__ == '__main__':
    # run_comps(*load_comparasion_data('data/5_10_3'))
    run_comps(*load_comparasion_data('data/5_10_4'),seg_len=10)
    # run_comps(*load_comparasion_data('data/5_10_5'),seg_len=10)
