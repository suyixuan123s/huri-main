"""
An example to show the point cloud data capturing from depth sensor
Author: Chen Hao
Email: chen960216@gmail.com
Date 2022.06.22
"""

if __name__ == "__main__":
    is_real_rbt = False
    import time
    from huri.core.common_import import wd, ym, np, fs, gm, rm
    from huri.vision.phoxi_capture import SensorMarkerHandler, vision_pipeline, vision_read_data, enhance_gray_img, \
        depth2gray_map
    from huri.components.utils.panda3d_utils import img_to_n_channel
    from huri.core.constants import SENSOR_INFO
    from huri.components.yumi_control.yumi_con import YumiController

    # Init base
    # base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    yumi_s = ym.Yumi(enable_cc=True)

    yumix = YumiController()
    rgt_jnts = yumix.get_jnt_values(component_name="rgt_arm")
    lft_jnts = yumix.get_jnt_values(component_name="lft_arm")
    yumi_s.fk(component_name="rgt_arm",
              jnt_values=rgt_jnts)
    yumi_s.fk(component_name="lft_arm",
              jnt_values=lft_jnts)
    yumi_s.jaw_to("rgt_hnd", yumix.get_gripper_width("rgt_arm"))
    yumix.stop()
    sensor_streamer = SensorMarkerHandler(SENSOR_INFO.IP_ADR_DEPTH_SENSOR_DEBUG)
    pcd, texture, depth_img, rgb_texture, extcam_img = vision_pipeline(sensor_streamer, rgb_texture=True)

    # color_texture = sensor_streamer.get_color_texture()
    texture_color = sensor_streamer.get_color_texture()[:, :, ::-1] / 255
    texture_color_flatten = texture_color.reshape(-1, 3)
    print(yumi_s.get_gl_tcp("rgt_arm"))

    fs.dump_pickle({
        "rbt_rgt_jnts":rgt_jnts,
        "rbt_lft_jnts":lft_jnts,
        "data": [pcd, texture, depth_img, rgb_texture, extcam_img]
    }, f"data/tcp")
    # base.run()
    print("saved successfully")