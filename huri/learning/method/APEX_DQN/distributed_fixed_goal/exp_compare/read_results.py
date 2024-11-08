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
COLORS = ["#009E73", "#E69F00", ]
LABELS = ['RL Solver', 'A* Solver', 'A* Solver (Success)', 'RL + Postprocessing']

def plot_img(data_list,
             colors,
             labels,
             fig_size=(12 / 1.5, 7 / 1.5),
             xlabel=None,
             ylabel=None,
             x_limit_max=None,
             y_limit_max=None,
             divide=None,
             divide_y=None,
             plot_path='plot.png',
             legend_bbox=(0.06, 0.25),
             toggle_label=False):
    plt.figure(figsize=fig_size)
    # Plot each dataset with a distinct marker and color
    for i, df in enumerate(data_list):
        if toggle_label:
            plt.plot(df.keys(), df.values(), color=colors[i], label=labels[i], linewidth=5)
        else:
            # plt.plot(df.keys(), df.values(), marker=markers[i], color=colors[i], linewidth=3)
            plt.plot(df.keys(), df.values(), color=colors[i], linewidth=5)
        # plt.title(f'Goal Pattern ({INDEX}): Average Success Solution Length', fontsize=font_size)
        if isinstance(xlabel, str):
            plt.xlabel(xlabel, fontsize=font_size)
        if isinstance(ylabel, str):
            plt.ylabel(ylabel, fontsize=font_size)
        if x_limit_max is not None:
            plt.xlim([0, x_limit_max])
        if y_limit_max is not None:
            plt.ylim([0, y_limit_max])

    if toggle_label:
        plt.legend(fontsize=legend_size, handlelength=.1, borderpad=0, frameon=False, labelspacing=0, )

    ## -----------------
    if PLOT_FLOT:
        from matplotlib.ticker import FuncFormatter
        # Define a formatter function
        def one_decimal_format(x, pos):
            """ Format with one decimal place """
            return f'{x:.1f}'
        # Set the formatter for the y-axis
        plt.gca().yaxis.set_major_formatter(FuncFormatter(one_decimal_format))
    ## -----------------

    ax = plt.gca()
    # Get lines and labels from the plot
    lines, labels = ax.get_legend_handles_labels()

    # Create custom legend handles as square markers
    custom_handles = [mlines.Line2D([], [], color=line.get_color(), marker='s', linestyle='None',
                                    markersize=20, label=label) for line, label in zip(lines, labels)]
    # Set major tick locator for denser grid
    ax.xaxis.set_major_locator(ticker.MultipleLocator(8))  # Adjust interval for x-axis
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(8))  # Adjust interval for y-axis

    legend = plt.legend(handles=custom_handles,
                        fontsize=legend_size,
                        handlelength=.1, borderpad=0, frameon=True,
                        labelspacing=0,
                        borderaxespad=0, loc='upper left')
    legend.get_frame().set_edgecolor('none')

    # Set the modified bounding box
    legend.set_bbox_to_anchor(legend_bbox, transform=plt.gca().transAxes)

    plt.grid(True)
    grid_color = 'gray'  # Color of the grid lines
    grid_line_width = 3  # Width of the grid lines
    # Get current axes
    ax.grid(linewidth=grid_line_width, linestyle='--')
    # Customize the spines (borders)
    spine_width = 3  # Specify the width of the border here
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
    if divide is not None:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(divide))  # Adjust interval for x-axis
    if divide_y is not None:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(divide_y))  # Adjust interval for x-axis

    plot_path = plot_path
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')


def success_rate(success_dict):
    success_rate_dict = {}
    for training_level in sorted(success_dict.keys()):
        success_rate_dict[training_level] = len(success_dict[training_level]) / 100
    return success_rate_dict


def average_time(time_dict):
    average_time_dict = {}
    for training_level in sorted(time_dict.keys()):
        average_time_dict[training_level] = np.average(time_dict[training_level])
    return average_time_dict


def average_success_time(time_dict, success_dict):
    average_time_dict = {}
    for training_level in sorted(time_dict.keys()):
        time = np.average([time_dict[training_level][v] for v in success_dict[training_level]])
        average_time_dict[training_level] = time
    return average_time_dict


def average_len(len_dict):
    average_time_dict = {}
    for training_level in sorted(len_dict.keys()):
        average_time_dict[training_level] = np.average([len(_) for _ in len_dict[training_level]])
    return average_time_dict


def average_success_len(len_dict, success_dict):
    average_time_dict = {}
    for training_level in sorted(len_dict.keys()):
        llen = [len(_) for _ in len_dict[training_level]]
        # if len(llen) < 1:
        #     continue
        lllen = np.average([llen[v] for v in success_dict[training_level]])
        average_time_dict[training_level] = lllen
    return average_time_dict


def plot_success_rate(data_list, toggle_label=False):
    # Storing dataframes in a list
    colors = ['cyan', 'magenta', ]
    colors = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]
    labels = ['RL Solver', 'A* Solver']

    plot_img(data_list[:len(labels)], colors=colors, labels=labels,
             fig_size=(6 / 1.5, 7 / 1.5),
             xlabel='#Tubes',
             plot_path='successful_rate.png',
             x_limit_max=len(data_list[0].keys()),
             y_limit_max=1.1,
             divide=7,
             toggle_label=toggle_label)

    # data_list[0][31] = 1
    # data_list[0][32] = 1
    # data_list[1][31] = 0
    # data_list[1][32] = 0
    # plot_img(data_list[:len(labels)], colors=colors, labels=labels,
    #          xlabel='#Tubes',
    #          plot_path='successful_rate.png',
    #          x_limit_max=len(data_list[0].keys()),
    #          y_limit_max=1.1,
    #          divide=7,
    #          toggle_label=True)
    #
    # plot_img(data_list[:len(labels)], colors=colors, labels=labels,
    #          xlabel='#Tubes',
    #          plot_path='successful_rate.png',
    #          x_limit_max=len(data_list[0].keys()),
    #          y_limit_max=1.1,
    #          divide=6,
    #          toggle_label=True)


def plot_time_consumption(data_list,
                          fig_size=(6 / 1.5, 7 / 1.5),
                          ylabel='Second (s)',
                          y_limit_max=9.5,
                          divide_y=3,
                          toggle_label=False,
                          plot_path='time_consumption.png'):
    # Storing dataframes in a list
    # Plot each dataset with a distinct marker and color
    colors = ['cyan', 'magenta', '']
    colors = ["#E69F00", "#009E73"]
    labels = ['RL Solver', 'RL + Postprocessing']
    plot_img(data_list, colors=colors, labels=labels,
             xlabel='#Tubes',
             ylabel=ylabel,
             fig_size=fig_size,
             plot_path=plot_path,
             legend_bbox=(0.06, 0.94),
             x_limit_max=len(data_list[0].keys()),
             y_limit_max=y_limit_max,
             divide=7,
             divide_y=divide_y,
             # y_limit_max=9,
             toggle_label=toggle_label)


def plot_time_consumption_A(data_list,
                            fig_size=(6 / 1.5, 7 / 1.5),
                            ylabel='Second (s)',
                            y_limit_max=9.5,
                            divide_y=3,
                            toggle_label=False,
                            plot_path='time_consumption.png'):
    # Storing dataframes in a list
    # Plot each dataset with a distinct marker and color
    colors = ['cyan', 'magenta', '']
    colors = ["#E69F00", "#56B4E9", "#009E73", "#009E73", "#F0E442"]
    labels = ['RL Solver', 'A* Solver', 'A* Solver (Success)', 'RL + Postprocessing']
    plot_img(data_list, colors=colors, labels=labels,
             xlabel='#Tubes',
             ylabel=ylabel,
             fig_size=fig_size,
             plot_path=plot_path,
             legend_bbox=(0.06, 0.94),
             x_limit_max=len(data_list[0].keys()),
             y_limit_max=y_limit_max,
             divide=7,
             divide_y=divide_y,
             # y_limit_max=9,
             toggle_label=toggle_label)


def plot_average_len(data_list,
                     fig_size=(6 / 1.5, 7 / 1.5),
                     ylabel='',
                     y_limit_max=9.5,
                     divide_y=3,
                     toggle_label=False,
                     plot_path='average_len.png'):
    # Storing dataframes in a list
    colors = ["#E69F00", "#56B4E9", "#009E73", "#009E73", "#F0E442"]
    colors = ["#E69F00", "#009E73"]
    labels = ['RL Solver', 'A* Solver', 'A* Solver (Success)', 'RL + Postprocessing']
    plot_img(data_list, colors=colors, labels=labels,
             xlabel='#Tubes',
             ylabel=ylabel,
             fig_size=fig_size,
             plot_path=plot_path,
             legend_bbox=(0.06, 0.94),
             x_limit_max=len(data_list[0].keys()),
             y_limit_max=y_limit_max,
             divide=7,
             divide_y=divide_y,
             # y_limit_max=9,
             toggle_label=toggle_label)


def plot_average_len_A(data_list,
                       fig_size=(6 / 1.5, 7 / 1.5),
                       ylabel='Length',
                       y_limit_max=9.5,
                       divide_y=3,
                       toggle_label=False,
                       plot_path='average_len_A.png'):
    # Storing dataframes in a list
    colors = ["#E69F00", "#56B4E9", "#009E73", "#009E73", "#F0E442"]
    labels = ['RL Solver', 'A* Solver', 'A* Solver (Success)', 'RL + Postprocessing']
    plot_img(data_list, colors=colors, labels=labels,
             xlabel='#Tubes',
             ylabel=ylabel,
             fig_size=fig_size,
             plot_path=plot_path,
             legend_bbox=(0.06, 0.94),
             x_limit_max=len(data_list[0].keys()),
             y_limit_max=y_limit_max,
             divide=7,
             divide_y=divide_y,
             # y_limit_max=9,
             toggle_label=toggle_label)


def load_comparasion_data(path: fs.Path):
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
         astar_solver_solution_dict_tmp,
         d3qn_solver_time_dict_noAstarrefiner_tmp,
         d3qn_solver_time_dict_tmp,
         astar_solver_time_dict_tmp,
         d3qn_solver_noAstarrefiner_success_inds_dict_tmp,
         d3qn_solver_success_inds_dict_tmp,
         astar_solver_success_inds_dict_tmp,
         ] = fs.load_pickle(p)
        d3qn_solver_solution_dict_noAstarrefiner.update(d3qn_solver_solution_dict_noAstarrefiner_tmp)
        d3qn_solver_solution_dict.update(d3qn_solver_solution_dict_tmp)
        astar_solver_solution_dict.update(astar_solver_solution_dict_tmp)
        d3qn_solver_time_dict_noAstarrefiner.update(d3qn_solver_time_dict_noAstarrefiner_tmp)
        d3qn_solver_time_dict.update(d3qn_solver_time_dict_tmp)
        astar_solver_time_dict.update(astar_solver_time_dict_tmp)
        d3qn_solver_noAstarrefiner_success_inds_dict.update(d3qn_solver_noAstarrefiner_success_inds_dict_tmp)
        d3qn_solver_success_inds_dict.update(d3qn_solver_success_inds_dict_tmp)
        astar_solver_success_inds_dict.update(astar_solver_success_inds_dict_tmp)

    print("successful rate")
    data = [success_rate(d3qn_solver_noAstarrefiner_success_inds_dict), success_rate(astar_solver_success_inds_dict),
            success_rate(d3qn_solver_success_inds_dict), ]
    toggle_label = False
    plot_success_rate(data, toggle_label=toggle_label)
    print("average time")
    data_time = [average_time(d3qn_solver_time_dict_noAstarrefiner),
                 average_time(d3qn_solver_time_dict), ]
    # data_time = [average_time(d3qn_solver_time_dict_noAstarrefiner),
    #              average_time(astar_solver_time_dict)]
    plot_time_consumption(data_time, y_limit_max=None,
                          divide_y=None,
                          ylabel='',
                          toggle_label=toggle_label)

    data_time = [average_success_time(d3qn_solver_time_dict_noAstarrefiner, astar_solver_success_inds_dict),
                 average_success_time(astar_solver_time_dict, astar_solver_success_inds_dict),
                 average_success_time(d3qn_solver_time_dict, astar_solver_success_inds_dict), ]

    plot_time_consumption_A(data_time,
                            toggle_label=toggle_label,
                            y_limit_max=None,
                            divide_y=None,
                            plot_path='a_star_time_consumption.jpg')

    data_time = [average_len(d3qn_solver_solution_dict_noAstarrefiner),
                 average_len(d3qn_solver_solution_dict), ]
    plot_average_len(data_time,
                     y_limit_max=None,
                     divide_y=None,
                     toggle_label=toggle_label)

    data_time = [average_success_len(d3qn_solver_solution_dict_noAstarrefiner, astar_solver_success_inds_dict),
                 average_success_len(astar_solver_solution_dict, astar_solver_success_inds_dict),
                 average_success_len(d3qn_solver_solution_dict, astar_solver_success_inds_dict), ]
    plot_average_len_A(data_time,
                       y_limit_max=None,
                       divide_y=None,
                       toggle_label=toggle_label)


if __name__ == "__main__":
    # load_comparasion_data(
    #     r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp_compare\data\5_10_5')
    # INDEX = 'b'
    # PLOT_FLOT = False
    PLOT_FLOT = True
    load_comparasion_data(
        r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp_compare\data\5_10_4')
    # INDEX = 'c'
    # load_comparasion_data(
    #     r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp_compare\data\5_10_4')
