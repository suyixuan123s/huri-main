import numpy as np
from huri.learning.env.arrangement_planning_rack.env import RackArrangementEnv, RackStatePlot
import huri.core.file_sys as fs

# initialize the environment
num_tube_classes = 3
rack_size = (5, 10)
action_space_dim = np.prod(rack_size) ** 2
observation_space_dim = (1, *rack_size)
env = RackArrangementEnv(rack_size=rack_size,
                         num_classes=num_tube_classes,
                         observation_space_dim=observation_space_dim,
                         action_space_dim=action_space_dim,
                         is_curriculum_lr=True,
                         is_goalpattern_fixed=True,
                         seed=1988,
                         difficulty=26)
env.goal_pattern = np.array([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                             [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                             [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                             [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                             [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]])

final_difficulty = 51
total_test_epsiode = 100
for num_of_obj in range(31, final_difficulty + 1):
    test_data = []
    env.difficulty = num_of_obj + 1
    for i in range(total_test_epsiode):
        init_state = env.reset()
        print(init_state)
        test_data.append(init_state.state)
    print("generated finished")
    fs.dump_pickle(test_data, f"data/test_data_{num_of_obj}.pkl")
