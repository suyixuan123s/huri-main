import numpy as np
from huri.learning.env.rack_v3.env import RackArrangementEnv, RackStatePlot
import huri.core.file_sys as fs
from huri.learning.env.rack_v3 import create_env

# initialize the environment
num_tube_classes = 1
rack_size = (3, 6)
env_meta_difficult: RackArrangementEnv = create_env(rack_sz=rack_size,
                                                    num_tube_class=num_tube_classes,
                                                    seed=888,
                                                    toggle_curriculum=True,
                                                    toggle_goal_fixed=False,
                                                    scheduler='GoalRackStateScheduler3',
                                                    num_history=1)

total_test_epsiode = 100
for num_of_obj in range(1, 18):
    test_data = []
    env_meta_difficult.scheduler.state_level = num_of_obj
    print()
    for i in range(total_test_epsiode):
        init_state = env_meta_difficult.reset()
        print(init_state)
        test_data.append([init_state.state, env_meta_difficult.goal_pattern])
    print("generated finished")
    fs.dump_pickle(test_data, f"./data/test_data_{num_of_obj}.pkl", reminder=False)
