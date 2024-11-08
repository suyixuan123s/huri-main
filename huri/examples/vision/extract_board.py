import numpy as np

from huri.core.common_import import *
from huri.vision.phoxi_capture import SensorMarkerHandler, vision_pipeline
import huri.vision.pnt_utils as pntu
import basis.trimesh as tm
from huri.components.vision.extract import oriented_box_icp_general

REAL_ROBOT = False
AFFINE_MAT_PATH = fs.workdir / "data" / "calibration" / "affine_mat_20210727-162433_rrr.json"
pcd_trimesh_boundingbox = [None]
IP_ADR = "192.168.125.100:18300"


def get_board_pos(robot_tcp_pos, robot_tcp_rotmat, affine_mat, pcd):
    # Transform and Plot Point Clouds
    pcd = rm.homomat_transform_points(affine_mat, points=pcd)
    gm.gen_pointcloud(pcd, ).attach_to(base)
    # Remove
    # pcd = pcd[pcd[:, 2] > 0.15]
    # gm.gen_pointcloud(pcd, [[1, 0, 0, .3]]).attach_to(base)

    pcd_in_hand = rm.homomat_transform_points(
        rm.homomat_inverse(rm.homomat_from_posrot(robot_tcp_pos, robot_tcp_rotmat)),
        pcd)
    idx = np.where((pcd_in_hand[:, 2] > -0.03) & (pcd_in_hand[:, 2] < 0.12)
                   & (pcd_in_hand[:, 0] > -0.01) & (pcd_in_hand[:, 0] < 0.01)
                   & (pcd_in_hand[:, 1] > -0.08) & (pcd_in_hand[:, 1] < 0.08))
    pcd = pcd[idx]

    board_pcd, outlier = pntu.segment_table_plane(pcd, distance_threshold=0.005, ransac_n=3)
    # gm.gen_pointcloud(pcd, [[1, 1, 0, .3]]).attach_to(base)
    gm.gen_pointcloud(board_pcd, [[0, 1, 0, .3]]).attach_to(base)
    board_mdl = cm.CollisionModel(tm.primitives.Box(box_extents=[0.1, 0.15, 0.001]))
    board_mdl_pnts = board_mdl.sample_surface(radius=0.003)[0]
    transform = oriented_box_icp_general(board_pcd, board_mdl_pnts)

    board_mdl_pnts_trans = rm.homomat_transform_points(transform, points=board_mdl_pnts)
    gm.gen_pointcloud(board_mdl_pnts_trans, [[1, 1, 1, .3]]).attach_to(base)
    print(transform[:3, 3])
    gm.gen_sphere(transform[:3, 3], radius=0.01).attach_to(base)
    return transform


def test(yumi_s, pcd):
    # Init base
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    # Generate robot model
    yumi_s.gen_meshmodel(rgba=[0, 0, 1, 0.3]).attach_to(base)
    # load affine mat
    affine_mat = np.asarray(
        fs.load_json(AFFINE_MAT_PATH)['affine_mat'])

    manipulate_hnd = "rgt_hnd"

    pos, rotmat = yumi_s.get_gl_tcp(manipulator_name=manipulate_hnd)
    get_board_pos(pos, rotmat, affine_mat, pcd)
    base.run()


if __name__ == "__main__":
    yumi_s = ym.Yumi(enable_cc=True)
    if REAL_ROBOT:
        from huri.components.yumi_control.yumi_con import YumiController

        yumix = YumiController()
        yumi_s.fk(component_name="rgt_arm",
                  jnt_values=yumix.get_jnt_values(component_name="rgt_arm"))
        yumi_s.fk(component_name="lft_arm",
                  jnt_values=yumix.get_jnt_values(component_name="lft_arm"))
        yumi_s.jaw_to("rgt_hnd", 0.05)
    pcd, img = vision_pipeline(SensorMarkerHandler(IP_ADR), dump_path=None)
    test(yumi_s, pcd)
