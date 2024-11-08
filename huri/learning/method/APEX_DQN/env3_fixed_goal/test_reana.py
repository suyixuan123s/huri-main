from env_tst import env, observation_space_dim, action_space_dim, num_tube_classes
from main import create_agent
from huri.learning.method.APEX_DQN.env3.actor import Actor, Eval, HERActor
import copy
from huri.learning.utils import select_device, LOGGER
import torch
import huri.core.file_sys as fs
import torch.multiprocessing as mp
import numpy as np
from huri.learning.A_start_teacher.A_star_teacher import rm_ras_actions_recur, rm_ras_actions_recur2, rm_ras_actions_recur3
from huri.learning.env.rack_v3.env import from_action
def extract_path_from_buffer(trainsition_buffer):
    return np.array([trainsition_buffer[0][1], *[i[-2] for i in trainsition_buffer]]), \
        [np.concatenate((i[1], i[-2])) for i in trainsition_buffer], trainsition_buffer[0][0]

if __name__ == "__main__":
    device = select_device()

    net = create_agent(observation_space_dim, action_space_dim, num_tube_classes, device)

    eval_net_path = fs.workdir_learning / "run" / f"dqn_debug" / "model_last.pt"
    # eval_net_path = fs.Path("E:\huri_shared\huri\learning\method\APEX_DQN\env2\min_tst\\transfer_learning_weight.pt")
    net.load_state_dict(torch.load(str(eval_net_path)))

    mp_manager = mp.Manager()
    shared_state = mp_manager.dict()
    shared_mem = mp_manager.Queue()
    shared_mem_ = mp_manager.Queue()
    shared_state['state_level'] = env.scheduler.state_level
    shared_state['state_level'] = 5
    shared_state['class_level'] = 4

    shared_net = copy.deepcopy(net)
    shared_net.eval()
    shared_net.share_memory()

    shared_reanalyzer_mem = mp.Queue()

    env_actor= env.copy()
    actor_proc = HERActor(actor_id=0,
                                  net=copy.deepcopy(net),
                                  env=env_actor,
                                  batch_size=32,
                                  epsilon_decay=1e-5,
                                  max_epsilon=1,
                                  min_epsilon=.1,
                                  reset_num=100,
                                  target_update_freq=1000,
                                  shared_net=shared_net,
                                  shared_state=shared_state,
                                  shared_reanalyzer_mem=shared_reanalyzer_mem,
                                  shared_mem=shared_mem,
                                  device=device,
                                  toggle_visual=False)
    actor_proc.start()


    # replay buffer
    state_level, replaybuffer_need_refined = shared_reanalyzer_mem.get(block=True)
    redundant_abs_state, redundant_abs_state_paired, goal_pattern = extract_path_from_buffer(replaybuffer_need_refined)

    _redundant_abs_state_paired_str = [str(_) for _ in redundant_abs_state_paired]
    redundant_abs_state_paired_str = copy.deepcopy(_redundant_abs_state_paired_str)

    redundant_path = redundant_abs_state
    goal_state_np = goal_pattern

    refined_path, refined_path_her = rm_ras_actions_recur3(redundant_path,
                                                           h=8,
                                                           goal_pattern=redundant_path[-1],
                                                           infeasible_dict={}, )

    for i in range(len(refined_path)):
        if len(refined_path[i]) >= len(redundant_path) - i:
            continue
        if len(refined_path[i]) < 2:
            continue
        transitions = []
        for _i in np.arange(len(refined_path[i]) - 1):
            s_current, s_next = refined_path[i][_i], refined_path[i][_i + 1]

            from_action()
            action = env.action_between_states(s_current=s_current, s_next=s_next)
            is_finsihed = env.is_finished(s_next, goal_pattern)
            # check if it is a unfinished state
            # if _i == len(refined_path) - 2 and not is_finsihed:
            #     reward = -50
            #     print("!!!!!!!!!!", s_current, s_next)
            # else:
            reward = env._get_reward(is_finsihed, s_current, s_next, s_current)
            tran_tmp = [goal_pattern,
                        s_current,
                        action,
                        reward,
                        s_next,
                        is_finsihed]
            transitions.append(tran_tmp)
            if is_finsihed:
                break
    for _ in transitions:
        print("a")

    actor_proc.join()