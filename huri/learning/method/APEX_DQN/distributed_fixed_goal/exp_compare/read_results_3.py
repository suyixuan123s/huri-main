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

from read_results_1 import *

font_size = 40
legend_size = 30
rcParams['font.size'] = 30
csfont = {'fontname': 'Times New Roman'}
plt.title('title', **csfont)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.family'] = 'Times New Roman'
COLORS = ["#009E73", "#E69F00", ]
LABELS = ['RL Solver', 'A* Solver', 'A* Solver (Success)', 'RL + Postprocessing']
PLOT_FLOT = True


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
    data = [success_rate(d3qn_solver_noAstarrefiner_success_inds_dict),
            success_rate(astar_solver_success_inds_dict), ]
    toggle_label = False
    plot_success_rate(data, toggle_label=toggle_label,
                      toggle_vline=True,
                      y_position_text="60%",
                      y_position=.6,
                      y_position_text_offset=.1
                      )
    print("average time")
    y_position = max(average_time(d3qn_solver_time_dict_noAstarrefiner).values())
    data_time = [
        average_success_time(d3qn_solver_time_dict_noAstarrefiner, astar_solver_success_inds_dict, success=False), ]
    # data_time = [average_time(d3qn_solver_time_dict_noAstarrefiner),
    #              average_time(astar_solver_time_dict)]
    plot_time_consumption(data_time,
                          y_limit_max=1.4,
                          divide_y=.5,
                          ylabel='',
                          toggle_label=toggle_label,
                          y_position=y_position,
                          y_position_text_offset=.15)

    data_time = [average_success_time(d3qn_solver_time_dict_noAstarrefiner, astar_solver_success_inds_dict, ),
                 average_success_time(astar_solver_time_dict, astar_solver_success_inds_dict, ), ]

    plot_time_consumption_A(data_time,
                            toggle_label=toggle_label,
                            x_limit_max=25,
                            y_limit_max=1.4,
                            divide_y=None,
                            plot_path='a_star_time_consumption.jpg', y_position=y_position,
                            y_position_text_offset=.15)

    data_time = [
        average_success_len(d3qn_solver_solution_dict_noAstarrefiner, astar_solver_success_inds_dict, success=False), ]
    plot_average_len(data_time,
                     y_limit_max=None,
                     divide_y=None,
                     toggle_label=toggle_label)

    data_time = [average_success_len(d3qn_solver_solution_dict_noAstarrefiner, astar_solver_success_inds_dict),
                 average_success_len(astar_solver_solution_dict, astar_solver_success_inds_dict), ]
    plot_average_len_A(data_time,
                       x_limit_max=25,
                       y_limit_max=None,
                       divide_y=None,
                       toggle_label=toggle_label)


if __name__ == "__main__":
    PLOT_FLOT = True
    load_comparasion_data(
        r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp_compare\data\5_10_5')
