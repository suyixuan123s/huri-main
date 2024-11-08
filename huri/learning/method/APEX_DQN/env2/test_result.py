from huri.learning.method.APEX_DQN.actor import Actor, Eval
from huri.learning.utils import select_device, LOGGER
from huri.learning.env.rack_v2.env import RackArrangementEnv, GOAL_PATTERN_5x10
from huri.learning.network.d3qn_attention import DuelingDQN5_
import numpy as np
import copy
import torch
import huri.core.file_sys as fs
import torch.multiprocessing as mp

if __name__ == "__main__":
    num_tube_classes = 1
    rack_size = (5, 10)
    action_space_dim = np.prod(rack_size) ** 2
    observation_space_dim = (rack_size[0], rack_size[1])
    observation_space_dim_nn = (1, *rack_size)
    env = RackArrangementEnv(rack_size=rack_size,
                             num_classes=num_tube_classes,
                             observation_space_dim=observation_space_dim,
                             action_space_dim=action_space_dim,
                             is_curriculum_lr=True,
                             is_goalpattern_fixed=False,
                             seed=888)
    # GOAL_PATTERN_5x10[GOAL_PATTERN_5x10 == 1] = 2
    # env.set_goal_pattern(GOAL_PATTERN_5x10)
    # env.scheduler.update_state_level()
    # env.scheduler.update_state_level()
    # env.scheduler.update_state_level()

    env1 = env.copy()

    device = select_device()

    net = DuelingDQN5_(obs_dim=observation_space_dim, action_dim=action_space_dim,
                       num_classes=num_tube_classes).to(device)

    # eval_net_path = fs.workdir_learning / "run" / f"dqn_debug" / "model_last.pt"
    eval_net_path = fs.Path("transfer_learning_weight.pt")
    net.load_state_dict(torch.load(str(eval_net_path)))

    shared_net = copy.deepcopy(net)
    shared_net.eval()
    shared_net.share_memory()
    mp_manager = mp.Manager()
    shared_state = mp_manager.dict()
    shared_state['state_level'] = 5
    shared_state['class_level'] = 1
    print(shared_state['state_level'])

    eval_proc = Eval(net=copy.deepcopy(net), env_test=env.copy(),
                     reset_num=2,
                     eval_num=20,
                     eval_interval=1,
                     shared_net=shared_net,
                     shared_state=shared_state,
                     device=device,
                     inc_diff_threshold=48,
                     toggle_visual=True,
                     save_model_path_best=None,
                     save_model_path_last=None)
    eval_proc.run()
