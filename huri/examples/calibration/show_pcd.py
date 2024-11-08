import numpy as np

from huri.core.common_import import *
from huri.vision.phoxi_capture import SensorMarkerHandler, vision_pipeline
from huri.examples.calibration.calibration_constants import CALIB_MAT_PATH


pcd_trimesh_boundingbox = [None]


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
    gm.gen_pointcloud(pcd, [[0, 1, 0, .3]]).attach_to(base)
    base.run()


if __name__ == "__main__":
    DEBUG = True
    REAL_ROBOT = False
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
        filename = fs.workdir / "data" / "vision_exp" / "20210920-164042.pkl"
        pcd, img = fs.load_pickle(filename)
    else:
        pcd, img = vision_pipeline(SensorMarkerHandler(),
                                   dump_path=fs.workdir / "data" / "vision_exp" / f"{time.strftime('%Y%m%d-%H%M%S')}.pkl")
    test(yumi_s, pcd)
