""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230720osaka

"""
import copy
import time
import ray
import cv2
import hydra
import huri.core.file_sys as fs
from huri.learning.env.rack_v3 import create_env
from huri.learning.method.APEX_DQN.distributed.pipeline import Actor, SharedState
from huri.learning.method.APEX_DQN.distributed.reanalyzer import Reanalyzer, extract_path_from_traj, \
    rm_ras_actions_recur3, RackStatePlot, refined_path_to_transitions_recur
from huri.learning.method.APEX_DQN.distributed.network import DDQN
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from huri.components.utils.img_utils import combine_images2
import numpy as np


@hydra.main(config_path='../params', config_name='20230517', version_base='1.3')
def main(cfg):
    ray.init(_node_ip_address='100.80.147.16',
             runtime_env={"working_dir": r'E:\learning\.',
                          "pip": ['shapely', 'numba', 'gym', 'open3d', 'lz4']})
    save_path = fs.workdir_learning / 'method' / 'APEX_DQN' / 'distributed' / 'run'
    env_meta = create_env(rack_sz=cfg['env']['rack_sz'],
                          num_tube_class=cfg['env']['num_tube_class'],
                          seed=cfg['env']['seed'],
                          toggle_curriculum=cfg['env']['toggle_curriculum'],
                          toggle_goal_fixed=cfg['env']['toggle_goal_fixed'],
                          num_history=1)
    input_shape = env_meta.observation_space_dim
    num_actions = env_meta.action_space_dim
    network = DDQN(input_shape, num_actions, num_filters=10, num_fc_units=128)
    replay_buffer = ray.remote(PrioritizedReplayBuffer).remote(capacity=int(cfg['rl']['replay_sz']),
                                                               alpha=0.6)
    ckpnt = {'weights': network.state_dict(),
             'training_level': 28,
             'trajectory_list': []
             }
    cfg['rl']['eps_decay'] = 0
    shared_state = SharedState.remote(ckpnt)
    actor = Actor.remote(actor_id=1,
                         env=env_meta,
                         net=copy.deepcopy(network),
                         cfg=cfg['rl'],
                         replay_buffer=replay_buffer,
                         shared_state=shared_state,
                         log_path=save_path.joinpath('log'),
                         toggle_visual=False)

    actor.start.remote()
    while True:
        traj_buff_len = ray.get(shared_state.get_info_len.remote('trajectory_list'))
        buff_len = ray.get(replay_buffer.__len__.remote())
        print(">==========", buff_len, traj_buff_len)
        if traj_buff_len > 0:
            traj = ray.get(shared_state.get_info_pop.remote('trajectory_list'))
        else:
            time.sleep(.5)
            continue
        traj.action_dim = env_meta.action_space_dim
        redundant_path, redundant_abs_state_paired_str, goal_pattern = extract_path_from_traj(traj)
        a = time.time()
        refined_path, refined_path_her = rm_ras_actions_recur3(redundant_path,
                                                               h=4,
                                                               goal_pattern=redundant_path[-1],
                                                               infeasible_dict={}, )
        b= time.time()
        print(f"time consumption: {b-a}")
        # exit(0)
        if np.array_equal(redundant_path[-1], goal_pattern):
            # print("! Successful ")
            pass
        else:
            # print(f" GOAL CHANGED, LEN from {len(redundant_path)} => {max([len(r) for r in refined_path])}")
            pass
        for i in range(len(refined_path)):
            # print(len(refined_path[i]), len(redundant_path) - i)
            refined_transitions = refined_path_to_transitions_recur(env_meta, refined_path[i], redundant_path[-1])
            for t in refined_transitions:
                _t_paired = np.concatenate((t['goal'].squeeze(0), t['state'].squeeze(0), t['next_state'].squeeze(0)))
                if str(_t_paired) not in redundant_abs_state_paired_str:
                    # print(t['state'], t['next_state'], t['reward'])
                    replay_buffer.add.remote(t)
                    redundant_abs_state_paired_str.append(str(_t_paired))
                else:
                    # print("Not appended")
                    pass

        drawer = RackStatePlot(redundant_path[-1])
        img_list = []
        img_list.extend([drawer.plot_states(path_r, row=22).get_img() for path_r in refined_path])
        # cv2.imshow("debug_her_label", combine_images2(img_list[-4:], columns=3))
        # cv2.waitKey(0)

        # HER

        goal_select_ids = np.arange(1, len(redundant_path) - 1)
        goal_pattern_set = redundant_path[goal_select_ids]
        for i in range(len(refined_path_her)):
            if len(refined_path_her[i]) >= len(redundant_path) - i:
                continue
            if len(refined_path_her[i]) < 2:
                continue
            goal_state_tmp_np = goal_pattern_set[i]
            refined_transitions = refined_path_to_transitions_recur(env_meta, refined_path_her[i],
                                                                    goal_state_tmp_np)
            if not np.array_equal(goal_state_tmp_np, refined_path_her[i][-1]):
                raise Exception("ERROR")
            rsp = RackStatePlot(goal_state_tmp_np, )
            img = rsp.plot_states(refined_path_her[i]).get_img()
            # cv2.imshow(f"goal_her_plot", img)
            # cv2.waitKey(0)
            for t in refined_transitions:
                _t_paired = np.concatenate(
                    (t['goal'].squeeze(0), t['state'].squeeze(0), t['next_state'].squeeze(0)))
                if str(_t_paired) not in redundant_abs_state_paired_str:
                    replay_buffer.add.remote(t)
                    redundant_abs_state_paired_str.append(str(_t_paired))
                else:
                    print("Not appended 222 ")


if __name__ == '__main__':
    main()
