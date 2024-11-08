"""
An example to show the point cloud data capturing from depth sensor
Author: Chen Hao
Email: chen960216@gmail.com
"""
import time

from huri.core.common_import import wd, ym, np, fs, gm, rm
from huri.vision.phoxi_capture import SensorMarkerHandler, vision_pipeline, vision_read_data, enhance_gray_img, \
    depth2gray_map
from huri.components.utils.panda3d_utils import img_to_n_channel
from huri.core.constants import SENSOR_INFO


def plot_pcd(base, pcd, color_texture=None):
    # load affine mat
    affine_mat = np.asarray(
        fs.load_json(SENSOR_INFO.PNT_CLD_CALIBR_MAT_PATH)['affine_mat'])
    # Transform and Plot Point Clouds
    pcd = rm.homomat_transform_points(affine_mat, points=pcd)
    gm.gen_pointcloud(pcd, [[0, 0, 0, .3]] if color_texture is None else color_texture).attach_to(base)


if __name__ == "__main__":
    import cv2

    is_cap = True
    is_real_rbt = False
    is_color = False
    # Init base
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    yumi_s = ym.Yumi(enable_cc=True)
    if is_real_rbt:
        from huri.components.yumi_control.yumi_con import YumiController

        yumix = YumiController()
        yumi_s.fk(component_name="rgt_arm",
                  jnt_values=yumix.get_jnt_values(component_name="rgt_arm"))
        yumi_s.fk(component_name="lft_arm",
                  jnt_values=yumix.get_jnt_values(component_name="lft_arm"))
        yumi_s.jaw_to("rgt_hnd", yumix.get_gripper_width("rgt_arm"))
        pos_x, rot_x = yumix.get_pose(component_name="rgt_arm")
        # yumi_s.fk("rgt_arm",yumi_s.ik(component_name="rgt_arm", tgt_pos=pos_x, tgt_rotmat=rot_x,seed_jnt_values=yumix.get_jnt_values(component_name="rgt_arm"),))
        yumi_s.gen_meshmodel().attach_to(base)
        print(pos_x, rot_x)
        gm.gen_frame(pos_x, rot_x).attach_to(base)
        yumix.stop()
    # pcd, img, depth_img = vision_read_data(filename=fs.workdir.joinpath("data","vision_exp","20220309-142914.pkl"))
    if is_cap:
        sensor_streamer = SensorMarkerHandler(SENSOR_INFO.IP_ADR_DEPTH_SENSOR_DEBUG)
        pcd, img, depth_img, rgb_texture, _ = vision_pipeline(sensor_streamer,
                                                              dump_path=fs.workdir.joinpath("data", "vision_exp",
                                                                                            f"{time.strftime('%Y%m%d-%H%M%S')}.pkl"),
                                                              rgb_texture=is_color)
        # color_texture = sensor_streamer.get_color_texture()
        if is_color:
            texture_color = sensor_streamer.get_color_texture()[:, :, ::-1] / 255
        else:
            texture_color = cv2.cvtColor(enhance_gray_img(img), cv2.COLOR_GRAY2BGR) / 255
        texture_color_flatten = texture_color.reshape(-1, 3)
        # Generate robot model
        # yumi_s.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
        print(yumi_s.get_gl_tcp("rgt_arm"))
    else:
        file_path = fs.workdir_data.joinpath("vision_exp", "20220422-222405.pkl")
        # file_path = fs.workdir.joinpath("examples", "data_collection", "data", "bluecap", "20220323-171932.pkl")
        pcd, img, depth_img, rgb_texture, rgb_img_raw = vision_read_data(file_path)
        if rgb_texture is None:
            texture_color = img_to_n_channel(enhance_gray_img(img))
        else:
            texture_color = rgb_texture[:, :, ::-1] / 255
        texture_color_flatten = texture_color.reshape(-1, 3)
        cv2.imshow("img", rgb_img_raw)
        # cv2.imshow("depth_img", depth2gray_map(depth_img))
        cv2.waitKey(0)

    # if is_color:
    plot_pcd(base, pcd,
             color_texture=np.hstack((texture_color_flatten, np.ones((len(texture_color_flatten), 1)))).tolist())
    # else:
    #     plot_pcd(base, pcd, )
    base.run()
