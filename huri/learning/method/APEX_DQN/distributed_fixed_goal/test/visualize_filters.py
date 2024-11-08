"""

Author: Hao Chen (chen960216@gmail.com)
Created: 20230727osaka

"""
processes = []
import torch
import hydra
from huri.learning.env.rack_v3 import create_env
from huri.learning.method.APEX_DQN.distributed.network import DDQN2 as DDQN
import numpy as np
import os
import matplotlib.patches as patches
from huri.learning.method.APEX_DQN.distributed.utils import to_onehot, swap_values
from huri.learning.env.rack_v3.env import RackArrangementEnv, RackStatePlot, RackState
from huri.learning.method.APEX_DQN.distributed.utils import abs_state_np, category_feasible_action, dummy_abs_state
from huri.components.utils.matlibplot_utils import Plot
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
import seaborn as sns

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import defaultdict

color = ['green', 'red', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'white', 'purple', 'pink', 'brown',
         'orange']
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 9  # You can adjust the size as needed
# font_properties = {'family': 'Arial', 'fontsize': 12, 'weight': 'bold'}
font_properties = {'family': 'Arial', 'fontsize': 12, }
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')


def draw(dqn, env, goal, state, num_columns=5):
    with torch.no_grad():
        dqn.eval()
        feasible_action_set = state.feasible_action_set
        input_value = np.array(abs_state_np(state.state, goal.state, env.num_classes))[None, ...]
        output = dqn.forward_features(
            dummy_abs_state(input_value, [torch.as_tensor(c, dtype=torch.float32, device=device).unsqueeze(0) for c in
                                          category_feasible_action(state.state, env.action_space_dim, env.num_classes)],
                            device
                            )).detach()
        num_filters = output.shape[1]
        num_rows = int(num_filters / num_columns) + int(num_filters % num_columns > 0)
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(20, num_rows * 2))
        axes = axes.flatten()
        for i in range(num_filters):
            ax = axes[i]
            sns.heatmap(output.data[0, i].cpu().numpy(), ax=ax, cmap='viridis', cbar=False, annot=False)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_xticks([])

        # Hide any unused subplots
        for i in range(num_filters, len(axes)):
            axes[i].axis('off')

        Plot(fig=fig).save_fig("filters.jpg")
        print("Finish drawing")



@hydra.main(config_path='../run/params/', config_name='params.yaml', version_base='1.3')
def main(cfg):
    env_meta = create_env(rack_sz=cfg['env']['rack_sz'],
                          num_tube_class=cfg['env']['num_tube_class'],
                          seed=cfg['env']['seed'],
                          toggle_curriculum=cfg['env']['toggle_curriculum'],
                          toggle_goal_fixed=cfg['env']['toggle_goal_fixed'],
                          scheduler='GoalRackStateScheduler3',
                          num_history=1)
    input_shape = env_meta.observation_space_dim
    num_actions = env_meta.action_space_dim
    network = dqn = DDQN(input_shape,
                         num_actions,
                         num_category=cfg['env']['num_tube_class'],
                         num_filters=cfg['ddqn']['num_filters'],
                         num_res_block=cfg['ddqn']['num_res_block'],
                         num_fc_units=cfg['ddqn']['num_fc_units'], )
    network.load_state_dict(
        torch.load(
            r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed\run\data\model_last.chkpt')[
            # r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed\exp5\run\data\model_best_6.chkpt')[
            'dqn_state_dict'])

    network.to(device)

    # Update the default rc settings
    from collections import defaultdict

    env = env_meta.copy()
    for i in range(100):
        print('\n' * 50)
        env.scheduler.set_training_level(3)
        # state = env.reset()
        # goal = env.goal_pattern

        state = env.reset()
        goal = env.goal_pattern
        print(repr(state), repr(goal))
        rsp = RackStatePlot(goal_pattern=goal)
        rsp.plot_states([state]).save_fig("state.jpg")
        draw(network, env, goal, state)
        # draw(network, env, goal_c, state_c)

        input("Wait: ... ")


if __name__ == '__main__':
    main()
