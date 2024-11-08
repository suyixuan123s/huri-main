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
             1.47,
             2.09,
             2.59,
             3.44,
             3.76,
             4.53,
             5.12,
             6.14,
             6.7,
             7.282828282828283,
             7.707070707070707,
             8.75257731958763,
             9.197802197802197,  # 14
             9.953488372093023,
             10.642857142857142,
             11.169014084507042,  # 17
             11.46875,  # 18
             12.20754716981132,  # 19
             12.804878048780488,  # 20
             13.96875,
             14.678571428571429
             ]


    data2 = [1,
             1.47,
             2.09,
             2.59,
             3.44,  # 5
             3.77,
             4.54,
             5.13,
             6.16,
             6.75,
             7.35,
             7.84,
             8.91,
             9.45,
             10.26,
             11.09,
             11.77,
             12.2,
             13.34,
             14.53,
             15.55,
             16.84,
             17.79,
             18.8,
             20.44,
             22.42,
             24.242424242424242,  # 27
             29.072164948453608,  # 28
             30.90909090909091,
             35.577319587628864,
             ]
    data2 = data2[:22]
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
