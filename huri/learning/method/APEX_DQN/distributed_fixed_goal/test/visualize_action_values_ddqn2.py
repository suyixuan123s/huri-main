"""

Author: Hao Chen (chen960216@gmail.com)
Created: 20230727osaka

"""
import time

processes = []
import torch
import hydra
from huri.learning.env.rack_v3 import create_env
from huri.learning.method.APEX_DQN.distributed.network import DDQN2 as DDQN
import numpy as np
import os
import matplotlib.patches as patches
from huri.learning.method.APEX_DQN.distributed.utils import to_onehot, swap_values, abs_state
from huri.learning.env.rack_v3.env import RackArrangementEnv, RackStatePlot, RackState
from huri.learning.method.APEX_DQN.distributed.utils import abs_state, category_feasible_action
from huri.components.utils.matlibplot_utils import Plot, plt
import cv2
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
import seaborn as sns

# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
from collections import defaultdict

color = ['green', 'red', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'white', 'purple', 'pink', 'brown',
         'orange']
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 9  # You can adjust the size as needed
# font_properties = {'family': 'Arial', 'fontsize': 12, 'weight': 'bold'}
font_properties = {'family': 'Arial', 'fontsize': 12, }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def draw(dqn, env, goal, state):
    with torch.no_grad():
        dqn.eval()
        feasible_action_set = state.feasible_action_set
        input_value = abs_state(torch.as_tensor(state.state, dtype=torch.float32, device=device).unsqueeze(0),
                                torch.as_tensor(env.goal_pattern.state, dtype=torch.float32, device=device).unsqueeze(
                                    0),
                                env.num_classes,
                                [torch.as_tensor(c, dtype=torch.float32, device=device) for c in
                                 category_feasible_action(state.state, env.action_space_dim)]
                                )
        print(input_value)
        a = time.time()
        dqn_action_value = dqn(input_value, toggle_debug=True).detach()
        b = time.time()
        print("time consumption is ", b - a)
    selected_action = feasible_action_set[dqn_action_value.squeeze()[feasible_action_set].argmax()].item()
    action_values = dqn_action_value.cpu().numpy().squeeze()
    action_values_indices = [(*env._expr_action(feasible_action_set[i]), action_values[feasible_action_set[i]]) for
                             i in range(len(feasible_action_set))]
    all_values = [v for _, _, v in action_values_indices if v != 0]
    min_val = min(all_values)
    max_val = max(all_values)

    print("Selected action:", env._expr_action(selected_action))
    # Create a dictionary to hold the groups
    groups = defaultdict(list)
    for coord in action_values_indices:
        groups[coord[0]].append(coord)

    # Determine the number of subplots needed
    n_subplots = len(groups)
    n_cols = int(np.ceil(np.sqrt(n_subplots)))
    n_rows = int(np.ceil(n_subplots / n_cols))

    # Create a figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    if n_subplots == 1:
        axes = [axes]

    # Flatten the axes array for easy indexing
    axes_flat = axes.flatten()

    for idx, (pick_coord, group) in enumerate(groups.items()):
        ax = axes_flat[idx]
        # Create a grid initialized to 0
        grid_size_x, grid_size_y = env.rack_size
        grid = np.zeros((grid_size_x, grid_size_y))
        # Populate the grid with counts
        for _, (x, y), v in group:
            grid[x, y] = v  # Increment the count for the grid cell at (x, y)

        # Create the heatmap
        im = ax.imshow(grid, cmap='viridis', origin='lower', vmin=min_val, vmax=max_val)
        # ax.colorbar(label='Intensity')
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_xticks(range(grid_size_y))
        ax.set_yticks(range(grid_size_x))
        ax.set_xlabel('Y Coordinate')
        ax.set_ylabel('X Coordinate')
        ax.set_title(f'Pick Coord: {pick_coord}')

        # Optionally, add the counts on the squares
        for (j, i), value in np.ndenumerate(grid):
            if value > 0:  # Only add text when there's a count
                ax.text(i, j, f'{value:.3f}', ha='center', va='center', color='white',
                        fontdict=font_properties)

        # draw pick
        circle_radius = 0.3
        circle = patches.Circle((pick_coord[1], pick_coord[0]), circle_radius,
                                edgecolor='none',
                                facecolor=color[state[pick_coord[0], pick_coord[1]]])  # Choose your color

        ax.add_patch(circle)
        # draw the goal
        for i in goal.num_classes:
            axis = np.where(goal.state == i)
            for _ in range(len(axis[0])):
                rect_x, rect_y = 0.9, 0.9
                rect = patches.Rectangle((axis[1][_] - rect_x / 2, axis[0][_] - rect_y / 2), rect_x, rect_y,
                                         linewidth=4,
                                         edgecolor=color[i],
                                         facecolor='none', transform=ax.transData)
                ax.add_patch(rect)

    # Hide any unused subplots
    for ax in axes_flat[n_subplots:]:
        ax.axis('off')
    return Plot(fig=fig)


@hydra.main(config_path='../params', config_name='20230517_3x6_2.yaml', version_base='1.3')
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

    # Update the default rc settings
    from collections import defaultdict
    network.load_state_dict(
        torch.load(r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed\run\data\model_last.chkpt')[
            'dqn_state_dict'])
    network.to(device)
    env = env_meta.copy()
    for i in range(100):
        print('\n' * 50)
        env.scheduler.set_training_level(7)
        state = env.reset()
        goal = env.goal_pattern

        m = np.concatenate((state.state.copy(), env.goal_pattern.state.copy()))
        m_swap = swap_values(m, env.num_classes)
        state_c, goal_c = RackState(m_swap[0: len(state.state)]), RackState(
            m_swap[len(state.state): 2 * len(state.state)])
        fig1 = draw(network, env, goal, state)
        print("-" * 30)
        fig2 = draw(network, env, goal_c, state_c)


        cv2.imshow('state', fig1.get_img())
        cv2.imshow('state2', fig2.get_img())
        # Adjust the layout
        # plt.tight_layout()
        # plt.show()
        cv2.waitKey(0)

    # # Create a bar chart to visualize the action values
    # plt.figure(figsize=(10, 6))  # Set the figure size as desired.
    # plt.bar(range(len(action_values)), action_values, color='skyblue')
    # plt.xlabel('Action Index')
    # plt.ylabel('Action Value')
    # plt.title('DQN Action Values Visualization')
    # plt.show()


if __name__ == '__main__':
    main()
