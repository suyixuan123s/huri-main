import copy

from shapely.geometry import Polygon
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

from basis.trimesh.creation import extrude_polygon

from huri.math.units import Mm
from huri.core.common_import import np, cm


# Plots a Polygon to pyplot `ax`
def plot_polygon(ax, poly, **kwargs):
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection


def rectangle_polygon(rect_center, rect_extent):
    """Generate a rectangle (shapely.geometry.Polygon)"""
    lu = rect_center - rect_extent / 2
    ru = rect_center + np.array([rect_extent[0], -rect_extent[1]]) / 2
    lt = rect_center + np.array([-rect_extent[0], +rect_extent[1]]) / 2
    rt = rect_center + rect_extent / 2
    return Polygon([lu, ru, lt, rt]).convex_hull


class ShapeGeom2D:
    def __init__(self, dim=(Mm(16), Mm(16)), center_pos=(0, 0)):
        self._dim = np.array(dim)
        self._center_pos = np.array(center_pos)

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, dim=(Mm(16), Mm(16))):
        self._dim = np.array(dim)

    @property
    def center_pos(self):
        return self._center_pos

    @center_pos.setter
    def center_pos(self, pos=(0, 0)):
        self._center_pos = np.array(pos)

    @property
    def shape(self):
        raise NotImplementedError

    def copy(self):
        return copy.deepcopy(self)


class SlotGeom(ShapeGeom2D):
    @property
    def shape(self):
        return rectangle_polygon(self._center_pos, self._dim)


class RackGeom(ShapeGeom2D):
    def __init__(self, slot_instance: SlotGeom, dim=(Mm(96), Mm(192)),
                 slot_shape=(5, 10),
                 slot_gap_dis_x=Mm(4.5),
                 slot_gap_dis_y=Mm(2),
                 height=Mm(55)):
        super().__init__(dim=dim)
        self.slots_geoms = np.zeros(slot_shape, dtype=SlotGeom)
        self.slots_centers = np.zeros((*slot_shape, 2))

        slot_dim_x, slot_dim_y = slot_instance.dim
        center_id_x = (slot_shape[0] + 1) / 2 - 1
        center_id_y = (slot_shape[1] + 1) / 2 - 1

        for _i in np.arange(slot_shape[0]):
            for _j in np.arange(slot_shape[1]):
                self.slots_centers[_i, _j] = np.array([
                    (_i - center_id_x) * (slot_dim_x + slot_gap_dis_x),
                    (_j - center_id_y) * (slot_dim_y + slot_gap_dis_y)
                ])
                slot_geom_t = slot_instance.copy()
                slot_geom_t.center_pos = self.slots_centers[_i, _j]
                self.slots_geoms[_i, _j] = slot_geom_t

        # height of the rack
        self.height = height
        # rack's pcd
        self._pcd = None

    @property
    def shape(self):
        rack_shape = rectangle_polygon(self._center_pos, self._dim)
        for slot_geom in self.slots_geoms.flatten():
            rack_shape = rack_shape.difference(slot_geom.shape)
        return rack_shape

    @property
    def cm(self):
        return cm.CollisionModel(extrude_polygon(self.shape, Mm(.1)))

    @property
    def pcd(self):
        if self._pcd is None:
            self._pcd = self.cm.sample_surface(Mm(.2), nsample=10000)[0]
        return self._pcd


rack_hard_geom = RackGeom(slot_instance=SlotGeom(dim=(Mm(17), Mm(16.5))),
                          dim=(Mm(95.3), Mm(191)),
                          slot_gap_dis_x=Mm(2),
                          slot_gap_dis_y=Mm(1.5),
                          height=Mm(55))

rack_soft_geom = RackGeom(slot_instance=SlotGeom(dim=(Mm(16.5), Mm(16.5))),
                          dim=(Mm(95.5), Mm(192.5)),
                          slot_gap_dis_x=Mm(1.2),
                          slot_gap_dis_y=Mm(2),
                          height=Mm(53))

if __name__ == "__main__":
    # slot_geom = SlotGeom(dim=(Mm(15), Mm(16.5)))
    # rack_geom = RackGeom(slot_instance=slot_geom)
    # print(rack_geom.slots_centers)
    # rack_cm = rack_geom.cm
    # fig, ax = plt.subplots()

    from huri.core.common_import import *

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    # rack_soft_geom.cm.attach_to(base)
    gm.gen_pointcloud(rack_soft_geom.pcd).attach_to(base)
    print(rack_hard_geom.slots_centers)
    print(rack_soft_geom.slots_centers)
    base.run()

    # def forceAspect(ax, aspect=1):
    #     im = ax.get_images()
    #     extent = im[0].get_extent()
    #     ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)
    #
    #
    # plot_polygon(ax, rack_geom_shape, facecolor='lightblue', edgecolor='red')
    # # forceAspect(ax, aspect=rack_geom.dim[0] / rack_geom.dim[1])
    # # ax.set_aspect(rack_geom.dim[0] / rack_geom.dim[1])
    # plt.axis('equal')
    # plt.show()
    # print("?")
