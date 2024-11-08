""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20231204osaka

"""
import huri.core.file_sys as fs
from huri.components.exe.version.utils import create_directory
from huri.learning.method.APEX_DQN.distributed_fixed_goal.env import create_fixe_env, RackArrangementEnv, \
    RackStatePlot, RackState
import numpy as np


def generate_data(goal_pattern, num_data=100):
    goal_pattern = np.asarray(goal_pattern)
    # generate folder:
    rack_size = goal_pattern.shape
    num_class = len(np.unique(goal_pattern[goal_pattern > 0]))
    num_possible_obj = np.count_nonzero(goal_pattern > 0)
    env = create_fixe_env(rack_sz=rack_size,
                          goal_pattern=goal_pattern,
                          num_tube_class=num_class,
                          num_history=1,
                          seed=np.random.randint(0, 1000000),
                          scheduler='RackStateScheduler',
                          toggle_curriculum=True)
    save_path = create_directory(f'data/{goal_pattern.shape[0]}_'
                                 f'{goal_pattern.shape[1]}_'
                                 f'{num_class}')

    goal_pattern_saved_path = save_path.joinpath('goal_pattern.pkl')
    state_list_saved_path = save_path.joinpath('state_list.pkl')
    states_list = {}
    for training_level in range(1, num_possible_obj + 1):
        env.scheduler.set_training_level(training_level)
        states_tmp = []
        for i in range(num_data):
            state = env.reset()
            states_tmp.append(state)
        states_list[training_level] = states_tmp
    fs.dump_pickle(goal_pattern, goal_pattern_saved_path, reminder=False)
    fs.dump_pickle(states_list, state_list_saved_path, reminder=False)
    print("Save finished")


if __name__ == '__main__':
    goal = {
        0: np.array([[1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ],
                     [1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ],
                     [1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ],
                     [1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ],
                     [1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ]]),
        1: np.array([[1, 1, 1, 1, 0, 0, 2, 2, 2, 2],
                     [1, 1, 1, 1, 0, 0, 2, 2, 2, 2],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [3, 3, 3, 3, 0, 0, 4, 4, 4, 4],
                     [3, 3, 3, 3, 0, 0, 4, 4, 4, 4]]),
        2: np.array([[1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
                     [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
                     [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
                     [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
                     [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ]]),
    }
    generate_data(goal[0])
    generate_data(goal[1])
    generate_data(goal[2])
