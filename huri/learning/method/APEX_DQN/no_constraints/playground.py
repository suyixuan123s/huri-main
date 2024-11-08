from policy.actor import Playground
import torch
import huri.core.file_sys as fs

import hydra
from main import create_env, create_agent
from utils import SharedState
import numpy as np


@hydra.main(config_path='params', config_name='20230517', version_base='1.3', )
def main(cfg):
    shared_state = SharedState()
    shared_state['state_level'] = cfg['eval']['state_level']
    shared_state['class_level'] = cfg['eval']['class_level']
    shared_state['state_level'] = 1

    env_meta = create_env(rack_sz=cfg['env']['rack_sz'],
                          num_tube_class=cfg['env']['num_tube_class'],
                          seed=cfg['env']['seed'],
                          toggle_curriculum=cfg['env']['toggle_curriculum'],
                          toggle_goal_fixed=cfg['env']['toggle_goal_fixed'])
    # env_meta.set_goal_pattern(np.array([[0, 0, 0],
    #                                     [0, 1, 0],
    #                                     [1, 0, 0]]))
    net = create_agent(env_meta.observation_space_dim,
                       env_meta.action_space_dim,
                       n_classes=env_meta.num_classes,
                       device=cfg['rl']['device'])
    save_path = fs.workdir_learning / "run" / f"dqn_debug"
    chkpnt = torch.load(save_path.joinpath('model.chkpt'))
    net.load_state_dict(chkpnt['dqn_state_dict'])
    cfg['eval']['reset_num'] = 5
    eval_proc = Playground(net=net,
                           env=env_meta.copy(),
                           cfg=cfg['eval'],
                           shared_state=shared_state.get_dict(),
                           toggle_visual=True)
    eval_proc.run()


if __name__ == "__main__":
    main()
