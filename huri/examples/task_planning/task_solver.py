import numpy as np
from huri.learning.env.arrangement_planning_rack.env import RackStatePlot
from huri.components.task_planning.tube_puzzle_learning_solver import DQNSolver
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

# initialize the environment

init_state = np.array([[0, 0, 0, 2, 0, 0, 2, 0, 0, 2],
                       [1, 0, 3, 0, 3, 2, 0, 3, 0, 0],
                       [0, 0, 1, 2, 0, 0, 0, 1, 0, 0],
                       [0, 1, 0, 0, 3, 1, 2, 1, 0, 1],
                       [0, 0, 3, 0, 0, 0, 0, 0, 2, 0]])

goal_pattern = np.array([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]])
ss1 = np.array([[1, 0, 0, 2, 0, 0, 2, 3, 0, 2],
                [1, 0, 0, 0, 0, 2, 0, 3, 0, 0],
                [0, 0, 1, 2, 0, 2, 0, 0, 0, 0],
                [1, 1, 0, 0, 3, 1, 2, 0, 0, 3],
                [1, 0, 0, 0, 0, 0, 0, 0, 3, 0]])
ss2 = np.array([[1, 0, 0, 2, 0, 0, 2, 3, 0, 0],
                [1, 0, 0, 2, 0, 2, 0, 3, 0, 0],
                [0, 0, 1, 2, 0, 2, 0, 0, 0, 0],
                [1, 1, 0, 0, 3, 1, 2, 0, 0, 3],
                [1, 0, 0, 0, 0, 0, 0, 0, 3, 0]])

qaq = np.array([[1, 1, 1, 0, 0, 2, 0, 3, 3, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 2, 0, 2, 0, 0, 0, 0],
                [1, 0, 2, 2, 0, 2, 0, 3, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 3, 3]])

qaq1 = np.array([[1, 1, 1, 0, 0, 2, 0, 3, 3, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1, 0, 1, 2, 0, 2, 0, 0, 0, 0],
                 [1, 0, 2, 2, 0, 2, 0, 3, 0, 0],
                 [1, 0, 0, 0, 0, 2, 0, 0, 3, 3]])

qaq2 = np.array([[1, 1, 1, 2, 0, 2, 0, 3, 3, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1, 0, 1, 2, 0, 2, 0, 0, 0, 0],
                 [1, 0, 0, 2, 0, 2, 0, 3, 0, 0],
                 [1, 0, 0, 0, 0, 2, 0, 0, 3, 3]])

solver = DQNSolver()
task_solver = solver
infeasible_local_pattern = {}
init_slot_id = (3, 2)
current_state = qaq1
next_state = qaq2
is_remove_all_init_slot_id_actions = True
is_init_feasible = False
if not is_init_feasible:
    infeasible_local = infeasible_local_pattern.get(tuple(init_slot_id), {})
    local_info = str(np.pad(current_state, 1)[init_slot_id[0]:init_slot_id[0] + 3,
                     init_slot_id[1]:init_slot_id[1] + 3])
    local_infeasible_actions = infeasible_local.get(local_info, [])
    local_infeasible_actions.extend(task_solver.action_between_states(current_state,
                                                                      next_state,
                                                                      remove_action_group=is_remove_all_init_slot_id_actions))
    infeasible_local[local_info] = local_infeasible_actions
    infeasible_local_pattern[tuple(init_slot_id)] = infeasible_local

path = solver.solve(current_state=qaq1,
                    goal_pattern=goal_pattern,
                    # infeasible_dict={str(ss1): solver.action_between_states(ss1, ss2)},
                    infeasible_dict={
                        # str(qaq): [32, 82, 132, 182, 232, 282, 332, 382, 432, 482, 532, 582, 632, 682, 732, 782, 832,
                        #            882, 932, 982, 1032, 1082, 1132, 1182, 1232, 1282, 1332, 1382, 1432, 1482, 1532,
                        #            1582, 1632, 1682, 1732, 1782, 1832, 1882, 1932, 1982, 2032, 2082, 2132, 2182, 2232,
                        #            2282, 2332, 2382, 2432, 2482],
                        str(qaq1): [32, 82, 132, 182, 232, 282, 332, 382, 432, 482, 532, 582, 632, 682, 732, 782, 832,
                                    882, 932, 982, 1032, 1082, 1132, 1182, 1232, 1282, 1332, 1382, 1432, 1482, 1532,
                                    1582, 1632, 1682, 1732, 1782, 1832, 1882, 1932, 1982, 2032, 2082, 2132, 2182, 2232,
                                    2282, 2332, 2382, 2432, 2482],
                        str(qaq2): [32, 82, 132, 182, 232, 282, 332, 382, 432, 482, 532, 582, 632, 682, 732, 782, 832,
                                    882, 932, 982, 1032, 1082, 1132, 1182, 1232, 1282, 1332, 1382, 1432, 1482, 1532,
                                    1582, 1632, 1682, 1732, 1782, 1832, 1882, 1932, 1982, 2032, 2082, 2132, 2182, 2232,
                                    2282, 2332, 2382, 2432, 2482],
                    },
                    infeasible_local_pattern=infeasible_local_pattern,
                    toggle_result=False)

for i in range(len(path) - 1):
    state = path[i + 1]
    state_pr = path[i]
    movemap = state - state_pr
    move_to_ind = np.concatenate(np.where(movemap > 0))
    move_from_ind = np.concatenate(np.where(movemap < 0))
    print(f"{i + 1}: Move from [{tuple(move_from_ind.tolist())}, {tuple(move_to_ind.tolist())}]")
# # visualize the result
if len(path) > 1:
    plotter = RackStatePlot(goal_pattern=goal_pattern)
    fig = plotter.plot_states(path, row=11)

    plt.show()
