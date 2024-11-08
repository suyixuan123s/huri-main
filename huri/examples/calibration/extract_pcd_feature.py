from huri.examples.calibration.calibration_constants import CALIB_MAT_PATH
import huri.core.utils as hcu
from huri.components.gui.tk_gui.base import GuiFrame
import basis.trimesh as tm
import numpy as np

color_group = [hcu.color_hex2oct(color) for color in hcu.color_hex["Number 3"]]


def extract_pcd(pcd, start_coord, extend, toggle=False):
    cp = lambda x, y: (x, y) if x < y else (y, x)
    lower_bd_0, upper_bd_0 = cp(start_coord[0], start_coord[0] + extend[0])
    lower_bd_1, upper_bd_1 = cp(start_coord[1], start_coord[1] + extend[1])
    lower_bd_2, upper_bd_2 = cp(start_coord[2], start_coord[2] + extend[2])
    pcd = np.asarray(pcd)
    extracted_pcd = pcd[
        (pcd[:, 0] > lower_bd_0) & (pcd[:, 0] < upper_bd_0) &
        (pcd[:, 1] > lower_bd_1) & (pcd[:, 1] < upper_bd_1) &
        (pcd[:, 2] > lower_bd_2) & (pcd[:, 2] < upper_bd_2)]
    if toggle:
        gm.gen_pointcloud(extracted_pcd, rgbas=[color_group[1]]).attach_to(base)
    return extracted_pcd


class GUIExtracter(GuiFrame):
    def __init__(self, root, pcd, topmost=False, close_hidden=False, resizable=False, hidden=False):
        super(GUIExtracter, self).__init__(root, topmost, close_hidden, resizable, hidden)
        self.pcd = pcd

        self.add_title("Extract the point cloud feature", pos=(0, 0))
        self.add_text("Start Coordination", pos=(1, 0))
        self.entry_start_coord = self.add_grid_entry(3, headers=["x", "y", "z"], pos=(2, 0))
        self.entry_start_coord.add_row(data=[0, 0, 0])
        self.add_text("Extend", pos=(3, 0))
        self.entry_extend_coord = self.add_grid_entry(3, headers=["x", "y", "z"], pos=(4, 0))
        self.entry_extend_coord.add_row(data=[0, 0, 0])
        self.add_button(text="Extract", command=self.extract_points, pos=(5, 0))
        self.plot_pcd_gm = None
        self.plot_box_gm = None

    def extract_points(self):
        start_coord_dict = self.entry_start_coord.read()[0]
        start_coord = np.array(
            [float(start_coord_dict['x']), float(start_coord_dict['y']), float(start_coord_dict['z'])])
        extend_coord_dict = self.entry_extend_coord.read()[0]
        extend_coord = np.array(
            [float(extend_coord_dict['x']), float(extend_coord_dict['y']), float(extend_coord_dict['z'])])
        points = extract_pcd(self.pcd, start_coord, extend_coord)
        if self.plot_pcd_gm is not None:
            self.plot_pcd_gm.remove()
        self.plot_pcd_gm = gm.gen_pointcloud(points, rgbas=[color_group[1]])
        self.plot_pcd_gm.attach_to(base)
        if self.plot_box_gm is not None:
            self.plot_box_gm.remove()
        self.plot_box_gm = gm.GeometricModel(
            tm.primitives.Box(box_center=start_coord + extend_coord / 2, box_extents=extend_coord / 2))
        self.plot_box_gm.attach_to(base)
        self.plot_box_gm.set_rgba([*color_group[3][:3], 0.5])


def test(yumi_s, pcd):
    # Init base
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    # Generate robot model
    yumi_s.gen_meshmodel().attach_to(base)
    # load affine mat
    affine_mat = np.asarray(
        fs.load_json(CALIB_MAT_PATH)['affine_mat'])
    # Transform and Plot Point Clouds
    pcd = rm.homomat_transform_points(affine_mat, points=pcd)
    gm.gen_pointcloud(pcd, [color_group[0]]).attach_to(base)

    window = GUIExtracter(base.tkRoot, pcd)

    # lft leg
    start_coord = np.array([0.095, 0.05, -0.03])
    extend_coord = np.array([0.055, 0.18, 0.1])

    # rgt leg
    start_coord = np.array([0.095, -0.23, -0.03])
    extend_coord = np.array([0.055, 0.18, 0.1])

    base.startTk()
    base.tkRoot.withdraw()
    base.run()


if __name__ == "__main__":
    from huri.core.common_import import *
    from huri.vision.phoxi_capture import SensorMarkerHandler, vision_pipeline
    import time

    DEBUG = True
    SAVE_PATH = None
    REAL_ROBOT = False
    pcd_trimesh_boundingbox = [None]

    yumi_s = ym.Yumi(enable_cc=True)
    if REAL_ROBOT:
        from huri.components.yumi_control.yumi_con import YumiController

        yumix = YumiController()
        yumi_s.fk(component_name="rgt_arm",
                  jnt_values=yumix.get_jnt_values(component_name="rgt_arm"))
        yumi_s.fk(component_name="lft_arm",
                  jnt_values=yumix.get_jnt_values(component_name="lft_arm"))
        yumi_s.jaw_to("rgt_hnd", 0.0)
    if DEBUG:
        filename = fs.workdir / "data" / "vision_exp" / "20220214-220155.pkl"
        pcd, img, _ = fs.load_pickle(filename)
    else:
        pcd, img = vision_pipeline(SensorMarkerHandler(),
                                   dump_path=fs.workdir / "data" / "vision_exp" / f"{time.strftime('%Y%m%d-%H%M%S')}.pkl")
    test(yumi_s, pcd)
