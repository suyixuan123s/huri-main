import matplotlib
from time import strftime

from huri.core.file_sys import workdir
from huri.core.base_boost import render_img, spawn_window, gen_img_texture_on_render, set_img_texture
from huri.vision.phoxi_capture import (vision_pipeline,
                                       SensorMarkerHandler,
                                       vision_read_data,
                                       depth2gray_map,
                                       enhance_gray_img, )
from huri.core.utils import img_to_n_channel
from huri.core.common_import import rm, np
matplotlib.use('TkAgg')

IP_ADR = "192.168.125.100:18300"
SAVE_PATH_VISION = workdir / "examples/vision/paper" / f"tst.pkl"  # None: Do not save the data captured by phoxi
SAVE_PATH_REALRBT = workdir / "examples/vision/paper" / f"tst_rbt.pkl"
AFFINE_MAT_PATH = workdir / "data/calibration/qaqqq.json"
DEBUG = False


def extract_pixel_around_hand(robot_tcp_pos,
                              robot_tcp_rotmat,
                              affine_mat,
                              pcd,
                              img_shape,
                              extract_area=((-.03, .03), (.01, .05), (-.03, .03)),
                              toggle_debug=False):
    # Transform and Plot Point Clouds
    pcd = rm.homomat_transform_points(affine_mat, points=pcd)
    pcd_in_hand = rm.homomat_transform_points(
        rm.homomat_inverse(rm.homomat_from_posrot(robot_tcp_pos, robot_tcp_rotmat)),
        pcd)
    idx = np.where((pcd_in_hand[:, 0] > extract_area[0][0]) & (pcd_in_hand[:, 0] < extract_area[0][1])
                   & (pcd_in_hand[:, 1] > extract_area[1][0]) & (pcd_in_hand[:, 1] < extract_area[1][1])
                   & (pcd_in_hand[:, 2] > extract_area[2][0]) & (pcd_in_hand[:, 2] < extract_area[2][1]))[0]
    h, w = img_shape[0], img_shape[1]
    idx_in_pixel = np.unravel_index(idx, (h, w))
    h1, h2 = idx_in_pixel[0].min(), idx_in_pixel[0].max()
    w1, w2 = idx_in_pixel[1].min(), idx_in_pixel[1].max()
    if toggle_debug:
        from huri.core.common_import import gm
        pcd_r = pcd[idx]
        # gm.gen_pointcloud(pcd_in_hand, rgbas=[[0, 1, 0, .5]]).attach_to(base)
        gm.gen_pointcloud(pcd_r).attach_to(base)
        gm.gen_frame().attach_to(base)
    return h1, w1, h2, w2, idx


if __name__ == "__main__":
    from huri.core.common_import import *
    from huri.components.yumi_control.yumi_con import YumiController

    # 3D environment
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    # init the simulation robot
    yumi_robot = ym.Yumi(enable_cc=True)

    if not DEBUG:
        yumi_con = YumiController(debug=False)
        jnt_vals_rgt = yumi_con.get_jnt_values(component_name="rgt_arm")
        jnt_vals_lft = yumi_con.get_jnt_values(component_name="lft_arm")
        fs.dump_pickle([jnt_vals_rgt, jnt_vals_lft], SAVE_PATH_REALRBT)
    else:
        jnt_vals_rgt, jnt_vals_lft = fs.load_pickle(SAVE_PATH_REALRBT)
    yumi_robot.fk("rgt_arm", jnt_vals_rgt)
    yumi_robot.fk("lft_arm", jnt_vals_lft)
    if DEBUG:
        pcd, img, depth_img = vision_read_data(SAVE_PATH_VISION)
    else:
        pcd, img, depth_img = vision_pipeline(streamer=SensorMarkerHandler(IP_ADR),
                                              dump_path=SAVE_PATH_VISION)
    affine_mat = np.asarray(
        fs.load_json(AFFINE_MAT_PATH)['affine_mat'])

    work_arm = "rgt_arm"

    yumi_robot.gen_meshmodel(toggle_tcpcs=True).attach_to(base)

    pos, rot = yumi_robot.get_gl_tcp(work_arm)
    h1, w1, h2, w2, _ = extract_pixel_around_hand(robot_tcp_pos=pos,
                                               robot_tcp_rotmat=rot,
                                               affine_mat=affine_mat,
                                               img_shape=img.shape,
                                               pcd=pcd,
                                               toggle_debug=True, )

    import cv2

    depth_img_labeled = cv2.rectangle(depth2gray_map(depth_img), (w1, h1), (w2, h2),
                                      color=(255, 0, 0), thickness=3)
    gray_img_labeled = cv2.rectangle(img_to_n_channel(enhance_gray_img(img)), (w1, h1), (w2, h2),
                                     color=(255, 0, 0), thickness=3)
    cv2.imshow("depth_img_labeled", depth_img_labeled)
    cv2.imshow("gray_img_labeled", gray_img_labeled)
    cv2.waitKey(0)

    # pcd_r = rm.homomat_transform_points(affine_mat, points=pcd)

    # depth_img_render, _, _, = spawn_window(base)
    # gray_img_render, _, _, = spawn_window(base)
    # depth_img_tx = gen_img_texture_on_render(depth_img_render)
    # set_img_texture(depth2color_map(depth_img), depth_img_tx)
    # gray_img_tx = gen_img_texture_on_render(gray_img_render)
    # set_img_texture(img_to_n_channel(enhance_gray_img(img)), gray_img_tx)
    # gm.gen_pointcloud(pcd_r, [[0, 0, 0, .3]]).attach_to(base)

    base.run()
