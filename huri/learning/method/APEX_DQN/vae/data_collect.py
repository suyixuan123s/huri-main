import cv2
from huri.learning.env.rack_v3.env import RackArrangementEnv, RackStatePlot, RackState
import numpy as np
import multiprocessing as mp
from os import getpid
from tqdm import tqdm

num_actor = 14
num_tube_classes = 3
rack_size = (5, 10)
data_num = 50000

each_data_num = int(data_num / num_actor)


def plot(states, goal):
    drawer = RackStatePlot(goal)
    fig = drawer.plot_states(states).get_img()
    cv2.imshow("window", fig)
    cv2.waitKey(10)


def generate_training_data(env):
    env.set_seed(np.random.randint(2 ** 31 - 1))
    proc_id = getpid()
    train_data = []
    for _ in tqdm(range(1, each_data_num)):
        state: RackState = env.reset()
        goal: RackState = env.goal_pattern
        train_data.append((state, state.to_abs_state))
        train_data.append((goal, goal.to_abs_state))

    print(f"{proc_id} finished")

    return train_data


if __name__ == "__main__":
    import huri.core.file_sys as fs

    num_worker_pool = min(int(num_actor), int(mp.cpu_count() / 2))
    print(f"Number of avaliable CPU: {mp.cpu_count()}. Number of cpu for each pool: ", num_worker_pool)

    num_tube_classes = num_tube_classes
    rack_size = rack_size
    observation_space_dim, action_space_dim = RackState.get_obs_act_dim_by_size(rack_size)
    env = RackArrangementEnv(rack_size=rack_size,
                             num_classes=num_tube_classes,
                             observation_space_dim=observation_space_dim,
                             action_space_dim=action_space_dim,
                             is_curriculum_lr=False,
                             is_goalpattern_fixed=False, )

    with mp.Pool(num_actor) as p:
        training_data = sum(p.map(generate_training_data,
                                  [env.copy() for num_worker_pool in range(num_actor)]),
                            [])
    fs.dump_pickle(training_data, path=f"ae_data_{rack_size[0]}_{rack_size[1]}.pkl")
