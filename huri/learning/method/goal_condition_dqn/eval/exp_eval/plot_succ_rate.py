import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams


def drawimage(ax, data_label, data_x, data_y, color='b'):
    print(data_label)
    x = data_x
    y = data_y
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  #
    ax.plot(x, y, c=color, markersize=2)
    ax.set_ylim(0, 1.1)
    ax.set_title(data_label, fontsize=11)


if __name__ == "__main__":
    # setup font
    rcParams['font.size'] = 15
    csfont = {'fontname': 'Times New Roman'}
    plt.title('title', **csfont)
    plt.rcParams["font.family"] = "Times New Roman"
    # data
    data1 = [100 / 100,  # 1
             100 / 100,  # 2
             100 / 100,  # 3
             100 / 100,  # 4
             100 / 100,  # 5
             100 / 100,  # 6
             100 / 100,  # 7
             100 / 100,  # 8
             100 / 100,  # 9
             100 / 100,  # 10
             99 / 100,  # 11
             99 / 100,  # 12
             97 / 100,  # 13
             91 / 100,  # 14
             86 / 100,  # 15
             70 / 100,  # 16
             71 / 100,  # 17
             64 / 100,  # 18
             53 / 100,  # 19
             41 / 100,  # 20
             32 / 100,  # 21
             28 / 100,  # 22
             13 / 100,  # 23
             17 / 100,  # 24
             7 / 100,  # 25
             5 / 100,  # 26
             3 / 100,  # 27
             0 / 100,  # 28
             0 / 100,  # 29
             0 / 100,  # 30
             ] + [0] * 19

    data2 = [100 / 100,  # 1
             100 / 100,  # 2
             100 / 100,  # 3
             100 / 100,  # 4
             100 / 100,  # 5
             100 / 100,  # 6
             100 / 100,  # 7
             100 / 100,  # 8
             100 / 100,  # 9
             100 / 100,  # 10
             100 / 100,  # 11
             100 / 100,  # 12
             100 / 100,  # 13
             100 / 100,  # 14
             100 / 100,  # 15
             100 / 100,  # 16
             100 / 100,  # 17
             100 / 100,  # 18
             100 / 100,  # 19
             100 / 100,  # 20
             100 / 100,  # 21
             100 / 100,  # 22
             100 / 100,  # 23
             100 / 100,  # 24
             100 / 100,  # 25
             100 / 100,  # 26
             99 / 100,  # 27
             97 / 100,  # 28
             99 / 100,  # 29
             97 / 100,  # 30
             90 / 100,  # 31
             76 / 100,  # 32
             55 / 100,  # 33
             38 / 100,  # 34
             12 / 100,  # 35
             1 / 100,  # 36
             0 / 100,  # 37
             0 / 100,  # 38
             0 / 100,  # 39
             0 / 100,  # 40
             0 / 100,  # 41
             0 / 100,  # 42
             0 / 100,  # 43
             0 / 100,  # 44
             0 / 100,  # 45
             0 / 100,  # 46
             0 / 100,  # 47
             0 / 100,  # 48
             0 / 100,  # 49

             ]
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
    plt.ylabel("Success rate")
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
