""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20231130osaka

"""

if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib import rcParams
    import matplotlib.lines as mlines
    import matplotlib.ticker as ticker

    font_size = 40
    rcParams['font.size'] = 30
    legend_size = 30
    csfont = {'fontname': 'Times New Roman'}
    plt.title('title', **csfont)
    plt.rcParams["font.family"] = "Times New Roman"

    # Creating a plot with each dataset represented by a different line
    # file_paths = [r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp1\5x10x5_no_A_star_completer\log\eval_log.csv',
    #               r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp1\5x10x5_no_A_star_refiner\log\eval_log.csv',
    #               r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp1\5x10x5_no_reanalyzer\log\eval_log.csv',
    #               r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\finished\5x10x5\log\eval_log.csv',
    #               r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp1\5x10x5_no_curriculum\log\eval_log.csv']

    file_paths = [
        r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\finished\5x10x3\log\eval_log.csv',
        r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\finished\5x10x4\log\eval_log.csv',
        r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp\data_efficiency\A_R_T\log\eval_log.csv'
    ]
    range_list = [
        550000,
        650000,
        130000,
    ]
    tick_len =[
        260000,
        200000,
        60000
    ]
    idx = 0

    # Storing dataframes in a list
    dataframes = [pd.read_csv(path) for path in file_paths]
    df = dataframes[idx]
    plt.figure(figsize=(10 / 1.5, 7 / 1.5))

    # Plot each dataset with a distinct marker and color
    # markers = ['o', 's', 'x', '^', '']
    colors = ['b', 'g', 'r', 'y', 'y']
    colors = [ "#D55E00", "#CC79A7", "#F0E442"]
    # colors = ['b', 'g', 'r', 'y', 'y']
    # labels = ['No A* C', 'No A* R', 'No R', 'Proposed', 'No E']
    labels = ['Baseline+R.+T.', 'Baseline+R.', 'Baseline+T.', 'Baseline', ]
    # labels = ['Baseline+R.+T.', 'Baseline+R.', 'Baseline+T.', 'Baseline', ]


    plt.plot(df['train_steps'], df['eval_average_score'], color=colors[0],
             linewidth=5)

    # plt.axvline(x=df['train_steps'].values[-1], color='r', linestyle='--',
    #             linewidth=5)  # Green dash-dot horizontal line

    # plt.title('Ablation Study', fontsize=font_size)
    plt.xlabel('Training Steps', fontsize=font_size)
    # plt.ylabel('Avg. Reward', fontsize=font_size)
    plt.xlim([0, range_list[idx]])
    # plt.ylim([0, 55])

    ax = plt.gca()
    # Get lines and labels from the plot
    lines, labels = ax.get_legend_handles_labels()

    # Create custom legend handles as square markers
    custom_handles = [mlines.Line2D([], [], color=line.get_color(), marker='s', linestyle='None',
                                    markersize=20, label=label) for line, label in zip(lines, labels)]

    # plt.legend(handles=custom_handles,fontsize=legend_size, handlelength=.1, handletextpad=.5, borderpad=0, frameon=False, labelspacing=0,
    #            borderaxespad=1.0)
    plt.grid(True)
    grid_color = 'gray'  # Color of the grid lines
    grid_line_width = 3  # Width of the grid lines
    # Get current axes
    ax.grid(linewidth=grid_line_width, linestyle='--')
    # Customize the spines (borders)
    spine_width = 3  # Specify the width of the border here
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)

    # ax.xaxis.set_major_locator(ticker.MultipleLocator(60000))  # Adjust interval for x-axis
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_len[idx]))  # Adjust interval for x-axis
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(8))  # Adjust interval for y-axis

    # Specify the y-position for the horizontal line
    # y_position = 48  # Example value
    # Draw a horizontal line
    # plt.axhline(y=y_position, color='black', linestyle='--', linewidth=3)  # Green dash-dot horizontal line

    plot_path = 'comparative_analysis_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
