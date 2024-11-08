""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230720osaka

"""
import copy
import time

import ray
import hydra
import huri.core.file_sys as fs
from huri.learning.env.rack_v3 import create_env
from huri.learning.method.APEX_DQN.distributed_fixed_goal.pipeline import Actor, SharedState
from huri.learning.method.APEX_DQN.distributed_fixed_goal.reanalyzer import Reanalyzer, refined_path_to_traj_recur
from huri.learning.method.APEX_DQN.distributed_fixed_goal.network import DDQN2 as DDQN
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from huri.learning.method.APEX_DQN.distributed_fixed_goal.env import create_fixe_env, GOAL_PATTERN_5x10_1

GOAL_PATTERN = GOAL_PATTERN_5x10_1


@hydra.main(config_path='../params', config_name='20230517_5x10_4.yaml', version_base='1.3')
def main(cfg):
    ray.init(local_mode=True, )
    print("cfg information: ", cfg)
    save_path = fs.workdir_learning / 'method' / 'APEX_DQN' / 'distributed' / 'run'
    env_meta = create_fixe_env(rack_sz=cfg['env']['rack_sz'],
                               goal_pattern=cfg['env']['goal_pattern'],
                               num_tube_class=cfg['env']['num_tube_class'],
                               seed=cfg['env']['seed'],
                               toggle_curriculum=cfg['env']['toggle_curriculum'],
                               num_history=1)
    replay_buffer = ray.remote(PrioritizedReplayBuffer).remote(capacity=10,
                                                               alpha=0.6)
    # traj_list = fs.load_pickle('../traj_list_important.pkl')
    traj_list = fs.load_pickle('../traj_list_1.pkl')
    # failed_path = fs.load_pickle(
    #     r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\test\debug_data\debug_failed_path2.pkl')
    failed_path = fs.load_pickle(
        r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\test\debug_data\human_traj_saved.pkl')
    failed_path = [i.state for i in failed_path]
    traj = refined_path_to_traj_recur(env_meta,
                                      failed_path,
                                      env_meta.action_space_dim, num_categories=env_meta.num_classes,
                                      goal_pattern=env_meta.goal_pattern,
                                      toggle_debug=False, gamma=cfg['rl']['gamma'], n_step=1)
    traj_list = [traj] * 10
    # traj_list = fs.load_p  ickle('debug_traj_list2.pkl') * 30
    ckpnt = {'training_level': 10, }
    ckpnt_traj_list = {'trajectory_list': traj_list, }
    shared_traj_buffer = SharedState.options(num_cpus=1).remote(ckpnt_traj_list)
    cfg['rl']['eps_decay'] = 0
    shared_state = SharedState.remote(ckpnt)
    print("Start reanalyzing ...")
    for i in range(1):
        reanalyzer = Reanalyzer.remote(uid=i,
                                       env=env_meta.copy(),
                                       replay_buffer=replay_buffer,
                                       shared_state=shared_state,
                                       log_path=save_path.joinpath('log'),
                                       shared_traj_buffer=shared_traj_buffer,
                                       pop_buff_len=1,
                                       toggle_visual=True,
                                       toggle_log_debug=False)

        reanalyzer.start.remote()

    while True:
        buff_len = ray.get(replay_buffer.__len__.remote())
        trajectory_list_len = len(ray.get(shared_state.get_info.remote('trajectory_list')))
        # if buff_len > 1:
        #     a = ray.get(replay_buffer.sample.remote(1, .6))
        #     print(a)
        print(buff_len, trajectory_list_len)
        time.sleep(.5)


if __name__ == '__main__':
    main()
