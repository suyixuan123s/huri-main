import io
import pickle

import numpy as np
import matplotlib.pyplot as plt


def show():
    plt.show()


class Plot(object):
    """
    An utils function for plot
    """

    def __init__(self, w=12.8, h=9.6, is_axis=True, dpi=100, fig=None):
        # init figure
        self._init_fig(fig=fig, figsize=(w, h), dpi=dpi)
        self.figsize = (w, h)
        self.dpi = dpi
        self.is_axis = is_axis

        # Add a sub figure if figure is empty
        if len(self.fig.axes) < 1:
            self.add_subplot(1, 1, 1)
        self.axes = self.fig.axes
        if not is_axis:
            for ax in self.axes:
                ax.axis('off')

        self.add_subplot = self.fig.add_subplot

    def _init_fig(self, fig=None, *args, **kwargs):
        """
        init figure
        :param fig: matplot.figure.Figure or None.
                    if fig is None, generate a new figure
        :return:
        """
        if fig is None:
            self.fig = plt.figure(*args, **kwargs)
        else:
            self.fig = fig

    def add_subplot(self, *args, **kwargs):
        ax = self.fig.add_subplot(*args, **kwargs)
        if not self.is_axis:
            ax.axis('off')
        return ax

    def set_size_inches(self, width, height):
        self.fig.set_size_inches(width, height)

    def get_img(self):
        """
        Get a numpy format image from current figure
        :return: np.array
        """
        fig = self.fig
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return img[..., ::-1]

    def save_fig(self, path: str, dpi=None, is_tight_layout=True):
        if dpi is None:
            dpi = self.dpi
        if is_tight_layout:
            plt.tight_layout()
        plt.savefig(path, dpi=dpi)

    def clear_fig(self):
        """
        clear the plot in the figure
        :return:
        """
        for ax in self.axes:
            ax.clear()

    def show(self):
        img = self.get_img()
        # plot show numpy image
        plt.imshow(img, )
        # Show the plot
        plt.show()

    def copy(self):
        # use pickle to copy matplotlib.figure.Figure
        # https://stackoverflow.com/questions/45810557/pyplot-copy-an-axes-content-and-show-it-in-a-new-figure
        fig_copy = pickle.loads(pickle.dumps(self.fig))
        return Plot(fig=fig_copy)


class LivePlot(object):
    """
    A live plot
    """

    def __init__(self, update_time=.5):
        """

        :param update_time: Time to redraw the scene
        """
        self._update_time = update_time

        # make pyplot in interactive mode
        if not plt.isinteractive():
            plt.ion()
        plt.show()


if __name__ == "__main__":
    plt.ion()
    plt.show()
    p = []
    for i in range(10):
        p += [i]
        plt.plot(p)
        plt.pause(0.5)
