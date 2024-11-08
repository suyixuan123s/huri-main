from numpy import array
import numpy as np

a = array([[0, 0, 1, 2, 0, 0, 3, 2, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0, 3, 0],
       [2, 0, 1, 2, 3, 2, 1, 1, 2, 0],
       [0, 0, 3, 0, 1, 1, 3, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


b = array([[0, 0, 1, 2, 0, 0, 3, 2, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
       [2, 0, 1, 2, 3, 2, 1, 1, 2, 0],
       [0, 0, 3, 0, 1, 1, 3, 1, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])


print(np.array_equal(a, b))
print(a-b)

# rack_state_history = [array([[1, 1, 1, 0, 0, 2, 0, 3, 3, 0],
#             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#             [1, 0, 1, 2, 0, 2, 0, 0, 0, 0],
#             [1, 0, 2, 2, 0, 2, 0, 3, 0, 0],
#             [0, 0, 1, 0, 0, 2, 0, 0, 3, 3]]), array([[1, 1, 1, 0, 0, 2, 0, 3, 3, 0],
#                                                      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#                                                      [1, 0, 1, 2, 0, 2, 0, 0, 0, 0],
#                                                      [1, 0, 2, 2, 0, 2, 0, 3, 0, 0],
#                                                      [0, 0, 0, 0, 0, 2, 1, 0, 3, 3]])]
# from huri.components.task_planning.tube_puzzle_learning_solver import DQNSolver
# task_solver = DQNSolver()
# current_state = np.array([[1, 1, 1, 0, 0, 2, 0, 3, 3, 0],
#                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#                  [1, 0, 1, 2, 0, 2, 0, 0, 0, 0],
#                  [1, 0, 2, 2, 0, 2, 0, 3, 0, 0],
#                  [0, 0, 0, 0, 0, 0, 1, 0, 3, 3]])
# for prev_state in rack_state_history:
#     print(task_solver.action_between_states(current_state, prev_state))
