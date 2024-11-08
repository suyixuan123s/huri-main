import time

import numpy as np
from huri.learning.env.arrangement_planning_rack.env import RackArrangementEnv, RackStatePlot
import huri.core.file_sys as fs
from huri.learning.method.DQN.eval.exp_eval.a_star import TubePuzzle


def a_star_agent(init_state, goal_pattern, max_iter_cnt=1000):
    # st = time.time()
    tp = TubePuzzle(init_state.copy())
    tp.goalpattern = goal_pattern.copy()
    # ed = time.time()
    is_finished, path = tp.atarSearch(max_iter_cnt=max_iter_cnt)
    if is_finished:
        return True, len(path)
    else:
        return False, len(path)


goal_pattern = np.array([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]])

final_difficulty = 30
total_test_epsiode = 100
success_path = [0] * final_difficulty
total_path = [0] * final_difficulty
for num_of_obj in range(23, final_difficulty + 1):
    print(f"Difficult level {num_of_obj}")
    test_data = fs.load_pickle(f"data/test_data_{num_of_obj}.pkl")
    diff_id = num_of_obj - 1
    success_path[diff_id] = 0
    for init_matrix in test_data:
        is_success, p_len = a_star_agent(init_matrix, goal_pattern=goal_pattern)
        if is_success:
            success_path[diff_id] += 1
            total_path[diff_id] += p_len - 1
    print("succ path ", success_path[diff_id])
    print("avg path len ", total_path[diff_id] / success_path[diff_id])
    print("----------------")
print(success_path)
