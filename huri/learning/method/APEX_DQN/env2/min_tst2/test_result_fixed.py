from huri.learning.method.APEX_DQN.actor import Actor, Eval
from huri.learning.utils import select_device, LOGGER
from huri.learning.method.APEX_DQN.env2.min_tst2.dqn_model import DuelingDQNMini, DuelingDQN, DuelingDQNAttentionMini
import numpy as np
import copy
import torch
import huri.core.file_sys as fs
import torch.multiprocessing as mp
from env_tst_fixed_p import env, num_tube_classes, rack_size, action_space_dim, observation_space_dim

if __name__ == "__main__":
    env1 = env.copy()

    device = select_device()

    net = DuelingDQNAttentionMini(obs_dim=observation_space_dim, action_dim=action_space_dim,
                                  num_classes=num_tube_classes).to(device)

    eval_net_path = fs.workdir_learning / "run" / f"dqn_debug" / "model_last.pt"
    # eval_net_path = fs.Path("E:\huri_shared\huri\learning\method\APEX_DQN\env2\min_tst\\transfer_learning_weight.pt")
    net.load_state_dict(torch.load(str(eval_net_path)))

    shared_net = copy.deepcopy(net)
    shared_net.eval()
    shared_net.share_memory()
    mp_manager = mp.Manager()
    shared_state = mp_manager.dict()
    shared_state['state_level'] = 4
    shared_state['class_level'] = 1
    print(shared_state['state_level'])

    eval_proc = Eval(net=copy.deepcopy(net), env_test=env.copy(),
                     reset_num=20,
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