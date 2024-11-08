import copy

import cv2

from huri.learning.A_start_teacher.A_star_teacher import TubePuzzle
from huri.learning.env.rack_v3 import create_env, RackArrangementEnv, RackState
import numpy as np
import multiprocessing as mp
from os import getpid
from tqdm import tqdm
import hydra
import huri.core.file_sys as fs
from collections.abc import Iterable
from typing import List

A_star_max_iter_cnt = 60


def plot(states, goal):
    drawer = RackStatePlot(goal)
    fig = drawer.plot_states(states).get_img()
    cv2.imshow("window", fig)
    cv2.waitKey(10)


def path2obs(path: List[np.ndarray], history: int) -> List[np.ndarray]:
    assert isinstance(path, Iterable) and isinstance(history, int), f"Input type errors:" \
                                                                    f"Path should be iterable, {type(path)} and" \
                                                                    f"history should be integer, {type(history)}"
    assert history >= 2, "History should at least be 2"
    obs_shape = path[0].shape
    obs = np.zeros((history, *obs_shape))
    obs[-1] = path[-1].copy()  # Path[-1] is assumed to be the goal pattern
    observations = []
    for i in range(len(path) - 1):
        obs_tmp = obs.copy()
        obs_tmp[-2] = path[i].copy()
        for j in range(min(i, history - 2)):
            obs_tmp[-2 - (j + 1)] = path[i - (j + 1)].copy()
        observations.append(obs_tmp)
    return observations


def generate_training_data(args, toggle_debug=False):
    (train_level, iter_num, env) = args
    assert isinstance(train_level, int) and isinstance(iter_num, int) and isinstance(env, RackArrangementEnv)
    train_level = min(train_level, np.prod(env.rack_size) - 2)
    env.set_seed(np.random.randint(2 ** 31 - 1))
    proc_id = getpid()
    print(f'I am number {proc_id}, Train Level & Iteration Number is :{(train_level, iter_num)}')
    rss = env.scheduler
    train_data = []
    for tl in tqdm(range(1, train_level + 1)):
        rss.set_training_level(tl)
        for _ in range(iter_num):
            goal = rss.gen_goal()
            state = rss.gen_state(goal)
            a = TubePuzzle(elearray=state.state, goalpattern=goal.state)
            is_find, path = a.atarSearch(infeasible_dict={}, max_iter_cnt=A_star_max_iter_cnt)
            if is_find:
                if toggle_debug:
                    plot(path, goal)
                train_data.append([goal.state, path, path2obs(path, env.observation_space_dim[0])])
        print(f"{proc_id} finished state{tl}")
    print(f"{proc_id} finished")

    dataset_raw = train_data
    dataset = []
    action_priors = np.zeros(env.action_space_dim, dtype=np.float32)
    # action_vec[:] = -np.inf
    for _ in tqdm(dataset_raw, desc=f'{proc_id} Dataset Raw', total=len(dataset_raw)):
        goal_p, path_traj, obs_traj = _
        path_traj, reward_traj, action_traj = env.evaluate_traj(path_traj, goal_p)
        # print(path_traj)
        for i in range(len(action_traj)):
            priors = action_priors.copy()
            # print(path_traj[i], env._expr_action(action_traj[i]), action_traj[i])
            priors[RackState(path_traj[i]).feasible_action_set] = 1
            priors[action_traj[i]] = 10
            priors = priors / sum(priors)
            dataset.append([obs_traj[i], priors, reward_traj[i]])
        # print("-"*20)
    return dataset


@hydra.main(config_path='../params', config_name='params_datacollect', version_base='1.3')
def collect_data(cfg):
    num_actor = cfg['num_actors']
    num_worker_pool = min(int(num_actor), int(mp.cpu_count() / 2))
    each_state_data_num = cfg['data_num']
    env_meta = create_env(rack_sz=cfg['env']['rack_sz'],
                          num_tube_class=cfg['env']['num_tube_class'],
                          seed=cfg['env']['seed'],
                          # num_history=cfg['env']['num_obs_history'],
                          num_history=4,
                          toggle_curriculum=cfg['env']['toggle_curriculum'],
                          toggle_goal_fixed=cfg['env']['toggle_goal_fixed'])
    print(f"Number of avaliable CPU: {mp.cpu_count()}. Number of cpu for each pool: {num_worker_pool}. "
          f"Rack size is ({env_meta.rack_size[0]},{env_meta.rack_size[1]})")
    with mp.Pool(num_actor) as p:
        training_data = sum(p.map(generate_training_data,
                                  [(cfg['train_level_max'], int(each_state_data_num / num_worker_pool),
                                    env_meta.copy()), ] * num_worker_pool),
                            [])
    fs.dump_pickle(training_data, path=f"demo_training_data_{env_meta.rack_size[0]}_{env_meta.rack_size[1]}.pkl",
                   reminder=False)


if __name__ == "__main__":
    collect_data()
