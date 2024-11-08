""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230703osaka

"""
import hydra
import numpy as np
from huri.learning.utils import select_device
from huri.learning.env.rack_v3 import RackArrangementEnv, create_env, RackStatePlot
import cv2


@hydra.main(config_path='../params', config_name='params20230531', version_base='1.3')
def eval(cfg):
    env_meta = create_env(rack_sz=cfg['env']['rack_sz'],
                          num_tube_class=cfg['env']['num_tube_class'],
                          seed=cfg['env']['seed'],
                          # num_history=cfg['env']['num_obs_history'],
                          num_history=4,
                          toggle_curriculum=cfg['env']['toggle_curriculum'],
                          toggle_goal_fixed=cfg['env']['toggle_goal_fixed'])
    eval_level = 1
    env = env_meta.copy(toggle_reset=True)
    env.scheduler.set_training_level(eval_level)
    env.reset()

    goal_pattern = env.goal_pattern
    state = env.state
    drawer = RackStatePlot(goal_pattern)
    print(f"Possible actions len: {len(state.feasible_action_set)}")
    all_states = {}
    all_states['origin'] = state
    for act in np.sort(state.feasible_action_set):
        env_tmp = env.copy(toggle_reset=False)
        env_tmp.step(act)
        all_states[f'action {act}'] = env_tmp.state
    p = drawer.plot_states(all_states, row=12, toggle_arrows=False)
    img = p.get_img()
    p.save_fig('result2.jpg', dpi=300)
    # cv2.imshow("a", img)
    # cv2.waitKey(0)


if __name__ == '__main__':
    eval()
