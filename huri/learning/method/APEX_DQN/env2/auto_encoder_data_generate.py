import copy

import cv2

from huri.learning.A_start_teacher.A_star_teacher import TubePuzzle
from huri.learning.env.rack_v2.env import RackArrangementEnv, RackStatePlot, \
    StripeRackStateScheduler, RackState
import numpy as np
import multiprocessing as mp
from os import getpid
from tqdm import tqdm

from huri.learning.env.rack_v2.utils import (get_possible_actions,
                                             get_fillable_movable,
                                             isdone,
                                             get_random_states,
                                             get_random_goal_pattern)

num_tube_classes = 1
# rack_size = (3, 5)
rack_size = (5, 10)
# A_star_max_iter_cnt = 300

A_star_max_iter_cnt = 60


def plot(states, goal):
    drawer = RackStatePlot(goal)
    fig = drawer.plot_states(states).get_img()
    cv2.imshow("window", fig)
    cv2.waitKey(10)


def generate_training_data(args, toggle_debug=False):
    (state_level, iter_num, env) = args
    env.set_seed(np.random.randint(2 ** 31 - 1))
    proc_id = getpid()
    print(f'I am number {proc_id}, {(state_level, iter_num)}')
    rss = StripeRackStateScheduler(num_classes=num_tube_classes, rack_size=rack_size, state_level=1)
    train_data = []
    for sl in tqdm(range(1, state_level + 1)):
        for _ in range(iter_num):
            goal = rss.gen_goal()
            state = rss.gen_state(goal)
            # s_fillable, s_movable = get_fillable_movable(state.state)
            s_action = state.feasible_action_set
            train_data.append([state.state, goal.state, s_action])
        rss.state_level = sl
        print(f"{proc_id} finished state{sl}")
    print(f"{proc_id} finished")

    return train_data


if __name__ == "__main__":
    import huri.core.file_sys as fs

    num_actor = 14
    # each_state_data_num = 10000
    each_state_data_num = 10000

    num_worker_pool = min(int(num_actor), int(mp.cpu_count() / 2))
    print(f"Number of avaliable CPU: {mp.cpu_count()}. Number of cpu for each pool: ", num_worker_pool)

    num_tube_classes = 2
    rack_size = (5, 10)
    action_space_dim = np.prod(rack_size) ** 2
    observation_space_dim = (1, rack_size[0], rack_size[1])
    observation_space_dim_nn = (1, *rack_size)
    env = RackArrangementEnv(rack_size=rack_size,
                             num_classes=num_tube_classes,
                             observation_space_dim=observation_space_dim,
                             action_space_dim=action_space_dim,
                             is_curriculum_lr=True,
                             is_goalpattern_fixed=False, )
    action_vec = np.zeros(action_space_dim, dtype=np.float32)

    with mp.Pool(num_actor) as p:
        training_data = sum(p.map(generate_training_data,
                                  [(20, int(each_state_data_num / num_worker_pool), env.copy()), ] * num_worker_pool),
                            [])
    fs.dump_pickle(training_data, path=f"ae_data_{rack_size[0]}_{rack_size[1]}.pkl")
