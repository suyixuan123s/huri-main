""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20231204osaka

"""
import time

from huri.components.task_planning.tube_puzzle_learning_solver import D3QNSolver
import huri.core.file_sys as fs

import numpy as np


def load_comparasion_data(path: fs.Path):
    p = fs.Path(path)
    goal_pattern_path = p.joinpath('goal_pattern.pkl')
    state_list_path = p.joinpath('state_list.pkl')
    if not (goal_pattern_path.exists() and state_list_path.exists()):
        raise Exception("Examine the files, goal_pattern.pkl or state_list.pkl is missing")

    goal_pattern = fs.load_pickle(goal_pattern_path)
    state_list = fs.load_pickle(state_list_path)
    return goal_pattern, state_list


def run_comps(goal_pattern,
              state_list,
              d3qn_solver: D3QNSolver,
              seg_len: int = 100,
              save_id=0,
              toggle_A_post_processing=True):
    save_path_parent = fs.Path(f'./A{save_id}')
    save_path_parent.mkdir(exist_ok=True)
    goal_pattern = np.asarray(goal_pattern)
    num_class = len(np.unique(goal_pattern[goal_pattern > 0]))
    d3qn_solver_solution_dict_noAstarrefiner = {}
    d3qn_solver_solution_dict = {}
    # ------------------------------------
    d3qn_solver_time_dict_noAstarrefiner = {}
    d3qn_solver_time_dict = {}

    d3qn_solver_noAstarrefiner_success_inds_dict = {}
    d3qn_solver_success_inds_dict = {}

    seg_len_counter = 1
    for training_level, states in state_list.items():
        d3qn_solver_solution_dict_noAstarrefiner[training_level] = []
        d3qn_solver_solution_dict[training_level] = []

        d3qn_solver_time_dict_noAstarrefiner[training_level] = []
        d3qn_solver_time_dict[training_level] = []

        d3qn_solver_noAstarrefiner_success_inds_dict[training_level] = []
        d3qn_solver_success_inds_dict[training_level] = []

        print(f"Start Training Level {training_level}/{len(state_list)}")
        for s_id, s in enumerate(states):
            a = time.time()
            path = d3qn_solver.solve(s, toggle_Astar_refiner=False)[0]
            b = time.time()
            if len(path) > 0:
                d3qn_solver_noAstarrefiner_success_inds_dict[training_level].append(s_id)
            d3qn_solver_time_dict_noAstarrefiner[training_level].append(b - a)
            d3qn_solver_solution_dict_noAstarrefiner[training_level].append(path)
            if toggle_A_post_processing:
                ##---------------------------
                a = time.time()
                path = d3qn_solver.solve(s, toggle_Astar_refiner=True)[1]
                b = time.time()
                if len(path) > 0:
                    d3qn_solver_success_inds_dict[training_level].append(s_id)
                d3qn_solver_time_dict[training_level].append(b - a)
                d3qn_solver_solution_dict[training_level].append(path)

        if training_level >= seg_len * seg_len_counter or training_level == max(state_list.keys()):
            seg_len_counter += 1
            training_level_up_index = min(d3qn_solver_solution_dict_noAstarrefiner.keys())
            save_path = save_path_parent.joinpath(
                f'eval_results_{save_id}_{training_level_up_index}_{training_level}.pkl')
            fs.dump_pickle([d3qn_solver_solution_dict_noAstarrefiner,
                            d3qn_solver_solution_dict,
                            d3qn_solver_time_dict_noAstarrefiner,
                            d3qn_solver_time_dict,
                            d3qn_solver_noAstarrefiner_success_inds_dict,
                            d3qn_solver_success_inds_dict,
                            ], save_path)
            print(f"Save segmentation data from training level {training_level_up_index}-{training_level}."
                  f"Save the file to {save_path}")
            # ------------------------------------
            d3qn_solver_solution_dict_noAstarrefiner = {}
            d3qn_solver_solution_dict = {}
            # ------------------------------------
            d3qn_solver_time_dict_noAstarrefiner = {}
            d3qn_solver_time_dict = {}
            # ------------------------------------
            d3qn_solver_noAstarrefiner_success_inds_dict = {}
            d3qn_solver_success_inds_dict = {}
            # ------------------------------------
    print(f"Finished running")


if __name__ == '__main__':
    # run_comps(*load_comparasion_data('data/5_10_3'))
    # run_comps(*load_comparasion_data('data/5_10_4'),seg_len=10)
    goal_pattern = np.array([[1, 0, 2, 0, 3, 0, 4, 0, 5, 0],
                             [1, 0, 2, 0, 3, 0, 4, 0, 5, 0],
                             [1, 0, 2, 0, 3, 0, 4, 0, 5, 0],
                             [1, 0, 2, 0, 3, 0, 4, 0, 5, 0],
                             [1, 0, 2, 0, 3, 0, 4, 0, 5, 0]])
    solver1_R_T = D3QNSolver(goal_pattern, toggle_init_net=False)
    solver1_R_T._net = solver1_R_T._init_net(
        r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp\data_efficiency\A_R_T\data\model_best_47.chkpt')
    solver1_R = D3QNSolver(goal_pattern, toggle_init_net=False)
    solver1_R._net = solver1_R._init_net(
        'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp\data_efficiency\A_R_2\data\model_best_47.chkpt')
    solver1_T = D3QNSolver(goal_pattern, toggle_init_net=False)
    solver1_T._net = solver1_T._init_net(
        r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp\data_efficiency\A_T\data\model_best_47.chkpt')

    solver1_B = D3QNSolver(goal_pattern, toggle_init_net=False)
    solver1_B._net = solver1_B._init_net(
        r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp\data_efficiency\baseline\data\model_best_47.chkpt')

    run_comps(*load_comparasion_data(
        r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp_compare\data\5_10_5'),
              save_id=0,
              d3qn_solver=solver1_R_T,
              seg_len=10, )

    # run_comps(*load_comparasion_data(
    #     r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp_compare\data\5_10_5'),
    #           save_id=1,
    #           d3qn_solver=solver1_R,
    #           seg_len=10, )
    #
    # run_comps(*load_comparasion_data(
    #     r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp_compare\data\5_10_5'),
    #           save_id=2,
    #           d3qn_solver=solver1_T,
    #           seg_len=10, )

    # run_comps(*load_comparasion_data(
    #     r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp_compare\data\5_10_5'),
    #           save_id=3,
    #           d3qn_solver=solver1_B,
    #           seg_len=10, )
