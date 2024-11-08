"""

Author: Hao Chen (chen960216@gmail.com)
Created: 20230727osaka

"""

if __name__ == '__main__':
    """ 

    Author: Hao Chen (chen960216@gmail.com)
    Created: 20230720osaka

    """
    processes = []
    import torch

    import copy
    import time

    import ray
    import hydra
    import huri.core.file_sys as fs
    from huri.learning.env.rack_v3 import create_env2
    from huri.learning.method.APEX_DQN.distributed_2.pipeline import Eval, SharedState
    from huri.learning.method.APEX_DQN.distributed_2.network import DDQN
    from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    import numpy as np


    @hydra.main(config_path='../params', config_name='20230517_3x6_2.yaml', version_base='1.3')
    def main(cfg):
        ray.init(_node_ip_address='100.80.147.16',
                 runtime_env={"working_dir": r'E:\learning\.',
                              "pip": ['shapely', 'numba', 'gym', 'open3d', 'lz4']})
        save_path = fs.workdir_learning / 'method' / 'APEX_DQN' / 'distributed_2' / 'run'
        env_meta = create_env2(rack_sz=cfg['env']['rack_sz'],
                              num_tube_class=cfg['env']['num_tube_class'],
                              seed=cfg['env']['seed'],
                              toggle_curriculum=cfg['env']['toggle_curriculum'],
                              toggle_goal_fixed=cfg['env']['toggle_goal_fixed'],
                              scheduler='GoalRackStateScheduler3',
                              num_history=1)
        input_shape = env_meta.observation_space_dim
        num_actions = env_meta.action_space_dim
        num_heads = env_meta.head_space_dim
        network = DDQN(input_shape,
                       num_actions,
                       num_heads,
                       num_category=cfg['env']['num_tube_class'],
                       num_filters=32,
                       num_res_block=5,
                       num_fc_units=64)
        ckpnt = {'train_steps': 1,
                 'weights':
                     torch.load(
                         r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_2\run\data\model_last.chkpt')[
                         'dqn_state_dict'],
                 'weights2':
                 'training_level': 5,
                 'trajectory_list': []
                 }
        ckpnt['weights'] = {k: v.cpu() for k, v in ckpnt['weights'].items()}
        shared_state = SharedState.remote(ckpnt)

        eval = Eval.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(),
                                                               soft=False)).remote(env=env_meta,
                                                                                   net=copy.deepcopy(network),
                                                                                   cfg=cfg['eval'],
                                                                                   shared_state=shared_state,
                                                                                   save_path=save_path.joinpath('data'),
                                                                                   log_path=None,
                                                                                   toggle_visual=True)

        v = env_meta.copy()
        # v.is_goalpattern_fixed = True
        # GOAL = np.array([[1, 1, 0, 1, 0, 0],
        #                  [1, 1, 0, 1, 0, 0],
        #                  [1, 1, 0, 1, 0, 0],
        #                  [1, 1, 0, 1, 0, 0],
        #                  [1, 1, 0, 1, 0, 0],
        #                  [1, 1, 0, 1, 0, 0]])
        # GOAL = np.array([[1, 0, 0, 0, 0, 2],
        #                  [1, 0, 0, 0, 0, 2],
        #                  [1, 0, 0, 0, 0, 2], ])
        # v.set_goal_pattern(GOAL)
        #

        processes.extend([shared_state,eval])
        v.scheduler.set_training_level(ckpnt['training_level'])
        eval.single_test.remote(v, toggle_sync_training=False)
        while True:
            time.sleep(1)


    if __name__ == '__main__':
        try:
            main()
        except:
            for p in processes:
                ray.kill(p)
