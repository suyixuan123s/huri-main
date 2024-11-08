import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams


def drawimage(ax, data_label, data_x, data_y, color='b'):
    print(data_label)
    x = data_x
    y = data_y
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  #
    ax.plot(x, y, c=color, markersize=2)
    ax.set_ylim(0, 20)
    ax.set_title(data_label, fontsize=11)


if __name__ == "__main__":
    # setup font
    rcParams['font.size'] = 15
    csfont = {'fontname': 'Times New Roman'}
    plt.title('title', **csfont)
    plt.rcParams["font.family"] = "Times New Roman"
    # data
    data1 = [1,
             1.77,
             2.47,
             3.1122448979591835,
             3.847058823529412,
             4.52,
             4.722222222222222,
             4.9,
             5.517241379310345,
             5.6923076923076925,
             6.0,
             5.75
             ]
    print(len(data1))

    data2 = [1.0, 1.77, 2.46, 3.06, 3.68, 4.29, 4.61, 5.16, 5.94, 6.63, 7.35, 8.252525252525253]
    print(len(data2))
    x = range(1, len(data1) + 1)
    # set up axis
    fig, ax = plt.subplots()

    # draw
    drawimage(ax, ' ', x, data1)
    drawimage(ax, ' ', x, data2, color='r')


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
    plt.xlabel("Number of test tubes")
    plt.ylabel("Avg. path length")
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
    plt.savefig('result4.jpg', dpi=300)
    # plt.subplot_tool()
    # plt.show()
