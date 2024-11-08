import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
import numpy as np

def drawimage(ax, data_label, data_x, data_y, color='b', **kwargs):
    print(data_label)
    x = data_x
    y = data_y
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  #
    ax.plot(x, y, c=color, markersize=2,  **kwargs)
    ax.set_ylim(0, 140)
    ax.set_title(data_label, fontsize=11)
    ax.legend()

if __name__ == "__main__":
    # setup font
    rcParams['font.size'] = 13
    csfont = {'fontname': 'Times New Roman'}
    plt.title('title', **csfont)
    plt.rcParams["font.family"] = "Times New Roman"
    # data
    data1 = [14.52897358, 28.9011097, 42.9840655326843, 58.95859766, 70.32770872

]

    data2 = [23.79988527,47.55332208, 72.45833874, 98.26514864, 121.4426234
]
    x = np.arange(1, len(data1) + 1)*400
    # set up axis
    fig, ax = plt.subplots()

    # draw
    drawimage(ax, ' ', x, data1, label="New IK Solver")
    drawimage(ax, ' ', x, data2, color='r', label="Numerical IK Solver")


    # remove y axis and add grid
    def remove_yaxis(ax):
        ax.tick_params(axis=u'y', which=u'both', length=0)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.sharey(ax1)


    def addgrid(ax):
        ax.grid(color='0.8', ls='-.', lw=0.25)


    # list(map(remove_yaxis, axlist[1:5]))
    list(map(addgrid, [ax]))
    # setup the size of image
    fig.set_size_inches(5, 3)
    # setup the position of "Time(s)" and "No.") and x axis
    # fig.text(0.80, .94, 'No.', va='center', rotation=0, fontsize=8.5)
    # fig.text(0.38, 0.02, 'Number of test tubes', va='center', rotation=0, fontsize=8.5)
    # fig.text(0.005, .94, 'Time (s)', va='center', rotation=0, fontsize=8.5, fontfamily="Times New Roman")
    plt.xlabel("Number of IK Calculation")
    plt.ylabel("Time Consumption (second)")
    # plt.subplots_adjust(left=0.05, bottom=0.157, right=0.945,
    #                     top=0.886, hspace=0.202,
    #                     wspace=0.031)
    # axlist[-1].set_ylim(0, 3000)
    # setup the position of the picture (#Nodes)
    # pos1 = axlist[-1].get_position()
    # pos2 = [pos1.x0 + 0.046, pos1.y0, pos1.width, pos1.height]
    # print(pos2)
    # axlist[-1].set_position(pos2)
    plt.tight_layout()
    plt.savefig('result3.jpg', dpi=300)
    # plt.subplot_tool()
    # plt.show()
