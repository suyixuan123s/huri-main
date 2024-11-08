from huri.learning.method.APEX_DQN.env3.actor import Actor, Eval
from huri.learning.utils import select_device, LOGGER
from main import create_agent
import numpy as np
import copy
import torch
import huri.core.file_sys as fs
import torch.multiprocessing as mp
from env_tst import env, num_tube_classes, rack_size, action_space_dim, observation_space_dim

if __name__ == "__main__":
    env1 = env.copy()

    start = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
    goal = np.array([[0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    env1.reset_state_goal(start, goal)

    device = select_device()

    net = create_agent(observation_space_dim, action_space_dim, num_tube_classes, device)

    eval_net_path = fs.workdir_learning / "run" / f"dqn_debug" / "model_last.pt"
    # eval_net_path = fs.Path("E:\huri_shared\huri\learning\method\APEX_DQN\env2\min_tst\\transfer_learning_weight.pt")
    net.load_state_dict(torch.load(str(eval_net_path)))

    shared_net = copy.deepcopy(net)
    shared_net.eval()
    shared_net.share_memory()
    mp_manager = mp.Manager()
    shared_state = mp_manager.dict()
    shared_state['state_level'] = 2
    shared_state['class_level'] = env.num_classes
    print(shared_state['state_level'])

    eval_proc = Eval(net=copy.deepcopy(net), env_test=env.copy(),
                     reset_num=3,
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
