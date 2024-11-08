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
    rm_ras_actions_recur3, RackStatePlot, refined_path_to_transitions_recur, a_star_solve, synthetic_traj
from huri.learning.method.APEX_DQN.distributed.network import DDQN
from ray.rllib.utils.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from huri.components.utils.img_utils import combine_images2
import numpy as np
import itertools
from huri.components.utils.img_utils import combine_images
from huri.learning.method.APEX_DQN.distributed.pipeline import Trajectory


def extract_path_from_state_list_goal(state_list, goal_pattern):
    goal_pattern = np.array(goal_pattern)
    return np.array([np.array(i) for i in state_list]), \
        [str(np.concatenate((goal_pattern, state_list[i], state_list[i + 1]))) for i in
         range(len(state_list) - 1)], \
        goal_pattern


@hydra.main(config_path='../params', config_name='20230517_3x6_2.yaml', version_base='1.3')
def main(cfg):
    env_meta = create_env(rack_sz=cfg['env']['rack_sz'],
                          num_tube_class=cfg['env']['num_tube_class'],
                          seed=cfg['env']['seed'],
                          toggle_curriculum=cfg['env']['toggle_curriculum'],
                          toggle_goal_fixed=cfg['env']['toggle_goal_fixed'],
                          num_history=1)
    env_meta.scheduler.set_training_level(3)
    # input_shape = env_meta.observation_space_dim
    # num_actions = env_meta.action_space_dim
    # network = DDQN(input_shape, num_actions, num_filters=10, num_fc_units=128)
    # replay_buffer = ray.remote(PrioritizedReplayBuffer).remote(capacity=int(cfg['rl']['replay_sz']),
    #                                                            alpha=0.6)
    # ckpnt = {'weights': network.state_dict(),
    #          'training_level': 28,
    #          'trajectory_list': []
    #          }
    # cfg['rl']['eps_decay'] = 0
    # shared_state = SharedState.remote(ckpnt)
    # actor = Actor.remote(actor_id=1,
    #                      env=env_meta,
    #                      net=copy.deepcopy(network),
    #                      cfg=cfg['rl'],
    #                      replay_buffer=replay_buffer,
    #                      shared_state=shared_state,
    #                      log_path=save_path.joinpath('log'),
    #                      toggle_visual=False)

    # actor.start.remote()
    while True:
        # traj_buff_len = ray.get(shared_state.get_info_len.remote('trajectory_list'))
        # buff_len = ray.get(replay_buffer.__len__.remote())
        # print(">==========", buff_len, traj_buff_len)
        # if traj_buff_len > 0:
        #     traj = ray.get(shared_state.get_info_pop.remote('trajectory_list'))
        # else:
        #     time.sleep(.5)
        #     continue
        # traj.action_dim = env_meta.action_space_dim
        # redundant_path, redundant_abs_state_paired_str, goal_pattern = extract_path_from_traj(traj)
        # ------------
        # state_list, goal_pattern = fs.load_pickle(
        #     r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed\run\data\20231116203455_eval_3.pkl')
        traj_list = fs.load_pickle('debug_traj_list2.pkl')
        goal_pattern = np.array(traj_list[0].goal)
        state_list = np.array(traj_list[0].states)
        rsp = RackStatePlot(goal_pattern, )
        # Uncomment if you want  to use randomly generated trajectory
        # Trajectory.action_dim = env_meta.action_space_dim
        # state = env_meta.reset()
        # goal_pattern = env_meta.goal_pattern
        # print(env.scheduler.state_level, env.is_curriculum_lr)
        # state_list = [state]
        # # dqn agent
        # traj = Trajectory(goal_pattern)
        # for t in itertools.count(1):
        #     action = env_meta.sample()
        #     if action is None:
        #         if t < 2:
        #             is_state_goal_feasible = False
        #         break
        #     if action is None or action < 0:
        #         next_state, reward, done = None, -10, True
        #     else:
        #         next_state, reward, done, _ = env_meta.step(action)
        #     traj.add_transition(state, action, reward, next_state, done)
        #     state = next_state  # state = next_state
        #     # accumulate rewards
        #     state_list.append(state)
        #     if done:
        #         break
        #     if t % 100 == 0:
        #         break
        # # synthetic_traj(traj, env_meta, np.random.randint(1, 4), toggle_debug=True)
        # fs.dump_pickle(traj, 'trajectory.pkl')
        img = rsp.plot_states(state_list, row=10).get_img()
        cv2.imshow(f"original", img)
        cv2.waitKey(0)

        redundant_path, redundant_abs_state_paired_str, goal_state_np = extract_path_from_state_list_goal(state_list,
                                                                                                          goal_pattern)

        if not np.array_equal(redundant_path[-1], goal_state_np):
            path = a_star_solve(redundant_path[-1], goal_state_np, )
            if len(path) > 0:
                # find the path to goal
                print("Find the path to goal")
                redundant_path = np.concatenate((redundant_path, path[1:]))
                # redundant_abs_state_paired_str = [  # update the redundant_abs_state_paired_str
                #     str(np.concatenate((goal_state_np, path[i], path[i + 1])))
                #     for i in range(len(path) - 1)]
                img = rsp.plot_states(redundant_path, row=13).get_img()
                cv2.imshow(f"complete the goal", img)
                cv2.waitKey(0)
            else:
                print("No path to goal")

        a = time.time()
        refined_path, refined_path_her = rm_ras_actions_recur3(redundant_path,
                                                               h=10,
                                                               goal_pattern=redundant_path[-1],
                                                               infeasible_dict={},
                                                               max_refine_num=800)
        # if not np.array_equal(redundant_path[-1], goal_state_np):
        #     goal_state_np = redundant_path[-1]
        #     # update the redundant_abs_state_paired_str
        #     redundant_abs_state_paired_str = [
        #         str(np.concatenate((goal_state_np, redundant_path[i], redundant_path[i + 1])))
        #         for i in range(len(redundant_path) - 1)]

        for i in range(len(refined_path)):
            store_transitions = []
            imgs_list = []
            if len(refined_path[i]) >= len(redundant_path) - i:
                continue
            if len(refined_path[i]) < 2:
                continue
            refined_transitions = refined_path_to_transitions_recur(env_meta, refined_path[i], goal_state_np)
            if len(refined_transitions) > 0:
                print(f"Add refined data into replay buffer: {len(refined_transitions) + 1}/{len(redundant_path)}")
            for t_id, t in enumerate(refined_transitions):
                _t_paired = np.concatenate((t['goal'], t['state'], t['next_state']))
                if str(_t_paired) not in redundant_abs_state_paired_str:
                    store_transitions.append(t_id)
                    redundant_abs_state_paired_str.append(str(_t_paired))
                    rsp = RackStatePlot(t['goal'], )
                    img = rsp.plot_states([t['state'], t['next_state']], ).get_img()
                    imgs_list.append(img)
            print("Number of Added data:", len(store_transitions), store_transitions)

            rsp = RackStatePlot(goal_state_np, )
            img = rsp.plot_states(refined_path[i], ).get_img()
            cv2.imshow(f"refined_states", img)
            if len(imgs_list) > 0:
                cv2.imshow(f"added", combine_images(imgs_list))
            cv2.waitKey(0)
            print("@" * 100)

        goal_select_ids = np.arange(1, len(redundant_path) - 1)
        goal_pattern_set = redundant_path[goal_select_ids]
        for i in range(len(refined_path_her)):
            store_transitions = []
            imgs_list = []
            if len(refined_path_her[i]) >= len(redundant_path) - i:
                continue
            if len(refined_path_her[i]) < 2:
                continue
            goal_state_tmp_np = goal_pattern_set[i]
            refined_transitions = refined_path_to_transitions_recur(env_meta, refined_path_her[i],
                                                                    goal_state_tmp_np)
            for t_id, t in enumerate(refined_transitions):
                _t_paired = np.concatenate((t['goal'], t['state'], t['next_state']))
                if str(_t_paired) not in redundant_abs_state_paired_str:
                    store_transitions.append(t_id)
                    redundant_abs_state_paired_str.append(str(_t_paired))
                    rsp = RackStatePlot(t['goal'], )
                    img = rsp.plot_states([t['state'], t['next_state']], ).get_img()
                    imgs_list.append(img)
            print("Number of Added data:", len(store_transitions), store_transitions)

            rsp = RackStatePlot(goal_state_tmp_np, )
            img = rsp.plot_states(refined_path_her[i]).get_img()
            cv2.imshow(f"goal_her_plot_{0}", img)
            if len(imgs_list) > 0:
                cv2.imshow(f"goal_her_plot_2{0}", combine_images(imgs_list))
            cv2.waitKey(0)


if __name__ == '__main__':
    main()
