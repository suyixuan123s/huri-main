""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20231212osaka

"""

import huri.core.file_sys as fs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.lines as mlines
import matplotlib.ticker as ticker

font_size = 40
legend_size = 30
rcParams['font.size'] = 30
csfont = {'fontname': 'Times New Roman'}
plt.title('title', **csfont)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.family'] = 'Times New Roman'
# Set the font size for the axes labels
# plt.rcParams['axes.labelsize'] = 22
# Set the font size for the numbers on the axes
# plt.rcParams['xtick.labelsize'] = 22
# plt.rcParams['ytick.labelsize'] = 22
SCALE = 2


def success_rate(success_dict):
    success_rate_dict = {}
    for training_level in range(1, max(success_dict.keys()) + 1):
        success_rate_dict[training_level] = len(success_dict[training_level]) / 100
    return success_rate_dict


def average_time(time_dict):
    average_time_dict = {}
    for training_level in range(1, max(time_dict.keys()) + 1):
        average_time_dict[training_level] = np.average(time_dict[training_level])
    return average_time_dict


def average_len(len_dict):
    average_time_dict = {}
    for training_level in range(1, max(len_dict.keys()) + 1):
        average_time_dict[training_level] = np.average([len(_) for _ in len_dict[training_level]])
    return average_time_dict


def plot_success_rate(data_list, toggle_label=False):
    # Storing dataframes in a list

    plt.figure(figsize=(12 / 1.5, 7 / 1.5))
    font_size = 25
    # Plot each dataset with a distinct marker and color
    markers = ['o', 's', 'x', '^', '']
    colors = ['b', 'g', 'r', 'm', 'y']
    labels = ['Baseline + R. + T.', 'Baseline + R. + T.', 'No Reanalyzer', 'Proposed', 'No Eval']

    for i, df in enumerate(data_list):
        if toggle_label:
            plt.plot(df.keys(), df.values(), marker=markers[i], color=colors[i], label=labels[i], linewidth=3)
        else:
            plt.plot(df.keys(), df.values(), marker=markers[i], color=colors[i], linewidth=3)

    # plt.title(f'Goal Pattern ({INDEX}): Success Rate', fontsize=font_size)
    plt.xlabel('#Objects', fontsize=font_size)
    # plt.ylabel('Successful Rate', fontsize=font_size)
    plt.xlim([0, len(data_list[0].keys())])
    plt.ylim([0, 1.01])
    if toggle_label:
        plt.legend(fontsize=20)
    plt.grid(True)

    plot_path = 'successful_rate.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')


def plot_time_consumption(data_list, toggle_label=False):
    # Storing dataframes in a list

    plt.figure(figsize=(12 / SCALE, 7 / SCALE))
    font_size = 25
    # Plot each dataset with a distinct marker and color
    markers = ['o', 's', 'x', '^', '']
    colors = ['b', 'g', 'r', 'm', 'y']
    labels = ['RL solver', 'A* solver', 'RL solver with A* relabeler', 'Proposed', 'No Eval']
    # labels = ['RL solver', 'A* solver', 'Proposed', 'No Eval']

    for i, df in enumerate(data_list):
        if toggle_label:
            plt.plot(df.keys(), df.values(), marker=markers[i], color=colors[i], label=labels[i], linewidth=3)
        else:
            plt.plot(df.keys(), df.values(), marker=markers[i], color=colors[i], linewidth=3)
        print(df.values())

    # plt.title(f'Goal Pattern ({INDEX}): Time Consumption', fontsize=font_size)
    plt.xlabel('#Tubes', fontsize=font_size)
    plt.ylabel('Second (s)', fontsize=font_size)
    plt.xlim([0, len(data_list[0].keys())])
    plt.ylim([0, 10])
    if toggle_label:
        plt.legend(fontsize=20)
    plt.grid(True)
    plot_path = 'time_consumption.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')


def plot_average_len(data_list, toggle_label=False):
    # Storing dataframes in a list

    plt.figure(figsize=(12 / 1.5, 7 / 1.5))
    # Plot each dataset with a distinct marker and color
    # markers = ['o', 's', 'x', '^', '']
    colors = ["#009E73", "#0072B2", "#00d198", "#56B4E9", "#D55E00", "#CC79A7", "#F0E442"]
    labels = ['Baseline+R.+T.', 'Baseline+R.', 'Baseline+T.', 'Baseline']

    for i, df in enumerate(data_list):
        if toggle_label:
            plt.plot(df.keys(), df.values(), color=colors[i], label=labels[i], linewidth=4)
        else:
            # plt.plot(df.keys(), df.values(), marker=markers[i], color=colors[i], linewidth=3)
            plt.plot(df.keys(), df.values(), color=colors[i], linewidth=5)

    # plt.title(f'Goal Pattern ({INDEX}): Average Success Solution Length', fontsize=font_size)
    plt.xlabel('#Tubes', fontsize=font_size)
    plt.ylabel('Length', fontsize=font_size)
    plt.xlim([0, len(data_list[0].keys())])
    plt.ylim([0, 35])
    if toggle_label:
        plt.legend(fontsize=legend_size, handlelength=.1, borderpad=0, frameon=False, labelspacing=0, )
    ax = plt.gca()
    # Get lines and labels from the plot
    lines, labels = ax.get_legend_handles_labels()

    # Create custom legend handles as square markers
    custom_handles = [mlines.Line2D([], [], color=line.get_color(), marker='s', linestyle='None',
                                    markersize=20, label=label) for line, label in zip(lines, labels)]
    # Set major tick locator for denser grid
    ax.xaxis.set_major_locator(ticker.MultipleLocator(8))  # Adjust interval for x-axis
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(8))  # Adjust interval for y-axis
    if False:
        legend = plt.legend(handles=custom_handles, fontsize=legend_size, handlelength=.1, borderpad=0, frameon=True,
                            labelspacing=0,
                            borderaxespad=0, loc='upper left')
        legend.get_frame().set_edgecolor('none')

        # Set the modified bounding box
        legend.set_bbox_to_anchor((0.06, 0.94), transform=plt.gca().transAxes)

    plt.grid(True)
    grid_color = 'gray'  # Color of the grid lines
    grid_line_width = 3  # Width of the grid lines
    # Get current axes
    ax.grid(linewidth=grid_line_width, linestyle='--')
    # Customize the spines (borders)
    spine_width = 3  # Specify the width of the border here
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
    plot_path = 'average_len.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')


def load_comparasion_data(path: fs.Path, op='success_rate'):
    p = fs.Path(path)
    eval_path = list(p.glob('**/eval_results*pkl'))
    if len(eval_path) < 1:
        raise Exception("Examine the files, goal_pattern.pkl or state_list.pkl is missing")
    d3qn_solver_solution_dict_noAstarrefiner, \
        d3qn_solver_solution_dict, \
        astar_solver_solution_dict, \
        d3qn_solver_time_dict_noAstarrefiner, \
        d3qn_solver_time_dict, \
        astar_solver_time_dict, \
        d3qn_solver_noAstarrefiner_success_inds_dict, \
        d3qn_solver_success_inds_dict, \
        astar_solver_success_inds_dict = {}, {}, {}, {}, {}, {}, {}, {}, {}
    for p in eval_path:
        [d3qn_solver_solution_dict_noAstarrefiner_tmp,
         d3qn_solver_solution_dict_tmp,
         d3qn_solver_time_dict_noAstarrefiner_tmp,
         d3qn_solver_time_dict_tmp,
         d3qn_solver_noAstarrefiner_success_inds_dict_tmp,
         d3qn_solver_success_inds_dict_tmp,
         ] = fs.load_pickle(p)
        d3qn_solver_solution_dict_noAstarrefiner.update(d3qn_solver_solution_dict_noAstarrefiner_tmp)
        d3qn_solver_solution_dict.update(d3qn_solver_solution_dict_tmp)
        d3qn_solver_time_dict_noAstarrefiner.update(d3qn_solver_time_dict_noAstarrefiner_tmp)
        d3qn_solver_time_dict.update(d3qn_solver_time_dict_tmp)
        d3qn_solver_noAstarrefiner_success_inds_dict.update(d3qn_solver_noAstarrefiner_success_inds_dict_tmp)
        d3qn_solver_success_inds_dict.update(d3qn_solver_success_inds_dict_tmp)
    if op == 'success_rate':
        return success_rate(d3qn_solver_solution_dict_noAstarrefiner)
    else:
        return average_len(d3qn_solver_solution_dict_noAstarrefiner)


def plot_len_comparison(data_list):
    data_time = [load_comparasion_data(dl, 'avg_len') for dl in data_list]
    plot_average_len(data_time, toggle_label=False)
    data_time = [load_comparasion_data(dl, ) for dl in data_list]
    plot_success_rate(data_time, toggle_label=True)


if __name__ == "__main__":
    INDEX = 'a'
    # load_comparasion_data(
    #     r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp_compare\data\5_10_5')
    # INDEX = 'b'
    # load_comparasion_data(
    #     r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp_compare\data\5_10_3')
    # INDEX = 'c'
    plot_len_comparison(
        [r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp_utils\exp_baseline\A0',
         r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp_utils\exp_baseline\A1',
         r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp_utils\exp_baseline\A2',
         r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp_utils\exp_baseline\A3'],
    )
