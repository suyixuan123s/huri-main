import numpy as np
import huri.core.file_sys as fs
from huri.components.task_planning.tube_puzzle_learning_solver import DQNSolver

# initialize the environment

goal_pattern = np.array([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]])

solver = DQNSolver()

final_difficulty = 50
total_test_epsiode = 100
success_path = [0] * final_difficulty
total_path = [0] * final_difficulty
time_info = [0] * final_difficulty
for num_of_obj in range(31, final_difficulty + 1):
    print(f"Difficult level {num_of_obj}")
    test_data = fs.load_pickle(f"data/test_data_{num_of_obj}.pkl")
    diff_id = num_of_obj - 1
    success_path[diff_id] = 0
    for init_matrix in test_data:
        path = solver.solve(current_state=init_matrix,
                            goal_pattern=goal_pattern, )
        if len(path) > 0:
            success_path[diff_id] += 1
            total_path[diff_id] += len(path) - 1
    print("succ path ", success_path[diff_id])
    print("avg path len ", total_path[diff_id] / success_path[diff_id])
    print("time spending", time_info[diff_id])
    print("----------------")
print(success_path)
