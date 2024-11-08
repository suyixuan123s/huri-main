""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20231130osaka

"""
import numpy as np

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
        r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp\data_efficiency\baseline\log\eval_log.csv',
        r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp\data_efficiency\ablation_study\num_post_processing\A3\log\eval_log.csv',
        r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp\data_efficiency\ablation_study\num_post_processing\A6\log\eval_log.csv',
        r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp\data_efficiency\ablation_study\num_post_processing\A9_3\log\eval_log.csv',
        # r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp\data_efficiency\ablation_study\num_post_processing\A12\log\eval_log.csv',
        # r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp\data_efficiency\ablation_study\num_post_processing\A12_2\log\eval_log.csv',
        r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp\data_efficiency\ablation_study\num_post_processing\A12_3\log\eval_log.csv',
        r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\exp\data_efficiency\ablation_study\num_post_processing\A15\log\eval_log.csv',
    ]

    # Storing dataframes in a list
    dataframes = [pd.read_csv(path) for path in file_paths]
    plt.figure(figsize=(12 / 1.5, 7 / 1.5))

    # Plot each dataset with a distinct marker and color
    # markers = ['o', 's', 'x', '^', '']
    # "#E69F00", "#56B4E9"
    colors = ["#F0E442", "#CC79A7", "#E69F00", "#D55E00", "#0072B2", "#009E73", "", "#"]
    colors = ['#56B4E9', '#0072B2', '#009E73', '#CC79A7', '#F0E442', '#00d198']
    # labels = ['No A* C', 'No A* R', 'No R', 'Proposed', 'No E']
    labels = ['0 Units', '3 Units', '6 Units', '9 Units', '12 Units', '15 Units']
    steps = np.array([df['train_steps'].values[-1] for df in dataframes])
    steps = (steps - steps[-1]) / steps[-1]
    print(steps)

    ax = plt.gca()
    # Specify the y-position for the horizontal line
    y_position = 48  # Example value
    # Draw a horizontal line
    # plt.axhline(y=y_position, color='orchid', linestyle='--', linewidth=3)  # Green dash-dot horizontal line
    # ax.text(10000/2, y_position +4, "Convergence", fontsize=30, fontname='Times New Roman', color='orchid', verticalalignment='center')

    for i, df in enumerate(dataframes):
        df['eval_average_score'].values[df['eval_average_score'].values < 0] = 0
        # plt.plot(df['train_steps'], df['eval_average_score'], marker=markers[i], color=colors[i], label=labels[i], linewidth=3)
        plt.plot(df['train_steps'], df['eval_average_score'], color=colors[i], label=labels[i],
                 linewidth=5)
        plt.axvline(x=df['train_steps'].values[-1], color=colors[i], linestyle='--',
                    linewidth=5)  # Green dash-dot horizontal line
        # plt.scatter(df['train_steps'].values[-1], df['eval_average_score'].values[-1], color=colors[i], s=300)  # s is the size of the marker
    for i, df in enumerate(dataframes[:3]):
        plt.axvline(x=df['train_steps'].values[-1], color=colors[i], linestyle='--', linewidth=5)

    # plt.title('Ablation Study', fontsize=font_size)
    plt.xlabel('Training Steps', fontsize=font_size)
    # plt.ylabel('Avg. Reward', fontsize=font_size)
    plt.xlim([0, 210000])
    plt.ylim([-1, 55])

    # Get lines and labels from the plot
    lines, labels = ax.get_legend_handles_labels()

    # Create custom legend handles as square markers
    custom_handles = [mlines.Line2D([], [], color=line.get_color(), marker='s', linestyle='None',
                                    markersize=20, label=label) for line, label in zip(lines, labels)]
    if False:
        legend = plt.legend(handles=custom_handles, fontsize=legend_size, handlelength=.1, borderpad=0, frameon=True,
                            labelspacing=0,
                            borderaxespad=0.3, loc='upper right')
        legend.get_frame().set_edgecolor('none')
    # plt.grid(True)
    grid_color = 'gray'  # Color of the grid lines
    grid_line_width = 3  # Width of the grid lines
    # Get current axes
    ax.grid(linewidth=grid_line_width, linestyle='--')
    # Customize the spines (borders)
    spine_width = 3  # Specify the width of the border here
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(50000))  # Adjust interval for x-axis
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(8))  # Adjust interval for y-axis

    plot_path = 'comparative_analysis_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
