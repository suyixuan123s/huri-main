import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry.polygon import Polygon, Point
from matplotlib import transforms


def rectangle_polygon(rect_center, rect_extent):
    """Generate a rectangle (shapely.geometry.Polygon)"""
    lu = rect_center - rect_extent / 2
    ru = rect_center + np.array([rect_extent[0], -rect_extent[1]]) / 2
    lt = rect_center + np.array([-rect_extent[0], +rect_extent[1]]) / 2
    rt = rect_center + rect_extent / 2
    return Polygon([lu, ru, lt, rt]).convex_hull


class Plot:
    def __init__(self, x_size=1280, y_size=960, is_plot_grid=False):
        self.fig = plt.figure(figsize=(x_size / 100.0, y_size / 100.0), dpi=100.0)

        self.is_plot_grid = is_plot_grid

        # The first graph
        self.ax1 = self.fig.add_subplot(1, 1, 1)
        self.ax1.axis('off')
        self.plot = self.ax1.plot
        self.fill = self.ax1.fill

    def get_img(self):
        fig = self.fig
        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return img

    def clear_img(self):
        self.ax1.clear()

    def show(self):
        plt.show()


class Drawer:
    def __init__(self, goal_pattern):
        self.goal_pattern = goal_pattern
        self.racksize = goal_pattern.shape
        self.tubeholesize = np.array([6, 6])
        self.tubeholecenters = []
        self.tubeshapelycollections = []
        for x in np.arange(self.racksize[0]) * self.tubeholesize[0]:
            for y in np.arange(self.racksize[1]) * self.tubeholesize[1]:
                self.tubeholecenters.append([x, y])
                self.tubeshapelycollections.append(rectangle_polygon(np.array([x, y]), self.tubeholesize))

        self.pattern_index = np.ones((np.max(goal_pattern), np.prod(self.racksize)), dtype=np.int) * -1
        for i in range(1, np.max(goal_pattern) + 1):
            tmp_pattern_index = np.asarray(np.where(goal_pattern == i)).T
            idx = tmp_pattern_index[:, 0] * self.racksize[1] + tmp_pattern_index[:, 1]
            self.pattern_index[i - 1, :len(idx)] = idx

    def plot_state(self, rack_path, row=6):
        rack_path_len = len(rack_path)
        fig, axs = plt.subplots(int(np.ceil(rack_path_len / (row))), min(rack_path_len, row),
                                figsize=(min(rack_path_len, row) * 2, 3 * int(np.ceil(rack_path_len / (row)))))
        if rack_path_len == 1:
            axs = np.array([axs])
        [i.axis('off') for i in axs.ravel()]
        axs = axs.reshape(-1, min(rack_path_len, row))
        for rack_id, rack in enumerate(rack_path):
            for idx, polygon in enumerate(self.tubeshapelycollections):
                field_type = np.where(self.pattern_index == idx)[0]
                if len(field_type) > 0:
                    tube_type = field_type.item()
                    if tube_type == 0:
                        tube_color = "orangered"
                    elif tube_type == 1:
                        tube_color = "mediumorchid"
                    elif tube_type == 2:
                        tube_color = "royalblue"
                    elif tube_type == 3:
                        tube_color = "lightgreen"

                    axs[rack_id // row, rack_id % row].fill(*polygon.exterior.xy, alpha=0.2, fc=tube_color, ec='none')

                    # for fields_idx in range
                axs[rack_id // row, rack_id % row].plot(*polygon.exterior.xy, color="silver")
            tubes_index = np.asarray(np.where(rack > 0)).T
            for idx in tubes_index:
                tube_position = self.tubeholecenters[idx[0] * self.racksize[1] + idx[1]]
                tube_type = rack[idx[0], idx[1]]

                if tube_type == 0:
                    tube_color = "orangered"
                elif tube_type == 1:
                    tube_color = "mediumorchid"
                elif tube_type == 2:
                    tube_color = "royalblue"
                elif tube_type == 3:
                    tube_color = "lightgreen"

                tube_color = tube_color
                tube_polygon = Point(tube_position[0], tube_position[1]).buffer(
                    np.min(self.tubeholesize) / np.sqrt(2) / 2
                )
                axs[rack_id // row, rack_id % row].plot(*tube_polygon.exterior.xy, color=tube_color)
        # plt.axis('scaled')
        plt.show()


# def plot_state():
#     for polygon in tubeshapelycollections:
#         plt.plot(*polygon.exterior.xy)

if __name__ == "__main__":
    # plot_state()
    dr = Drawer(goal_pattern=np.array([
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2],
        [0, 0, 0, 0, 0, 0],
    ]))
    dr.plot_state(np.array([
        [1, 1, 1, 2, 2, 2],
        [0, 0, 0, 2, 2, 2],
        [1, 0, 1, 2, 2, 2],
        [0, 0, 0, 0, 0, 0],
    ]))
