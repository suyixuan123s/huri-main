from huri.learning.env.arrangement_planning_rack_gc_apex.env import RackStatePlot, RackArrangementEnv, RackState
import cv2
import numpy as np
from huri.learning.A_start_teacher.A_star_teacher import E, F
from huri.components.utils.matlibplot_utils import Plot

GOAL_PATTERN = np.array([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]])
rack_size = (5, 10)
num_classes = 3
observation_space_dim = (num_classes * 2, *rack_size)
action_space_dim = np.prod(rack_size) ** 2
env = RackArrangementEnv(rack_size=rack_size,
                         num_classes=num_classes,
                         observation_space_dim=observation_space_dim,
                         action_space_dim=action_space_dim,
                         is_goalpattern_fixed=False,
                         is_curriculum_lr=False)
env.goal_pattern = RackState(GOAL_PATTERN)
drawer = RackStatePlot(GOAL_PATTERN)
init_state = np.array([[2, 3, 1, 0, 2, 2, 0, 1, 3, 0],
                       [1, 0, 0, 0, 0, 1, 3, 3, 0, 0],
                       [2, 2, 3, 0, 1, 0, 0, 2, 3, 0],
                       [0, 1, 0, 2, 0, 0, 2, 0, 2, 3],
                       [3, 3, 2, 0, 3, 3, 0, 1, 3, 1]])
print(len(init_state[init_state > 0]))

s1 = env.reset_state(init_state)

demo_list = [s1.state]
demo_rs_list = [s1]
set_p = False
window_name = 'a'

# TO DO
# Solve the bug
import huri.core.file_sys as fs
while 1:
    p = drawer.plot_states(demo_list, plot_movable=True, plot_fillable=True, row=12, )
    img = p.get_img()
    cv2.imshow(window_name, img)
    if set_p is False:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        set_p = True
    cv2.waitKey(10)

    human_input = input("Please select a tube \n")
    row, column = human_input[0], human_input[1]
    if row == '-' or column == '-':
        demo_list = demo_list[:-1]
        demo_rs_list = demo_rs_list[:-1]
        continue
    m_row, m_column = int(row), 9 - int(column)
    human_input = input("Please select a hole \n")
    row, column = human_input[0], human_input[1]
    f_row, f_column = int(row), 9 - int(column)
    action_code = env.to_action(pick_id=(m_row, m_column), place_id=(f_row, f_column))
    if action_code not in demo_rs_list[-1].feasible_action_set:
        print("Action Failed!!")
        continue
    s, _, _, _ = env.step(action_code)
    demo_list.append(s.state)
    demo_rs_list.append(s)
    fs.dump_pickle(demo_list, "demo",reminder=False)
    cv2.imwrite("demo.jpg", img)
