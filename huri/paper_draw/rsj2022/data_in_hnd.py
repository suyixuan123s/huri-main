import numpy as np

from huri.core.common_import import fs, wd, ym, gm, rm
from huri.core.constants import SENSOR_INFO
from huri.core.base_boost import zoombase, boost_base

data_dict: dict = fs.load_pickle("data/blue_cap_in_hand_tcp")
# data_dict: dict = fs.load_pickle("data/tcp")

base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
yumi_s = ym.Yumi(enable_cc=True)

yumi_s.fk("rgt_arm", data_dict["rbt_rgt_jnts"])

yumi_s.jaw_to("rgt_hnd", 5.3 * 2 / 1000)
# yumi_s.gen_meshmodel().attach_to(base)

import cv2
from huri.vision.phoxi_capture import enhance_gray_img

collected_im = enhance_gray_img(data_dict["data"][1])
collected_im = cv2.cvtColor(collected_im, cv2.COLOR_GRAY2BGR)
color_c3 = collected_im.copy().reshape(-1, 3)
color_c4 = np.ones((len(color_c3), 4), dtype=float)
color_c4[..., :3] = color_c3 / 255

affine_mat = np.asarray(
    fs.load_json(SENSOR_INFO.PNT_CLD_CALIBR_MAT_PATH)['affine_mat'])
pcd = rm.homomat_transform_points(affine_mat, data_dict["data"][0])
pcd_ind = (pcd[:, 2] > .1) & (pcd[:, 2] < .2)
pcd = pcd[pcd_ind]
color_c4 = color_c4[pcd_ind]
print(color_c4)
gm.gen_pointcloud(points=pcd, rgbas=color_c4.tolist()).attach_to(base)

highlight_in_hnd = True
if highlight_in_hnd:
    pos, rot = yumi_s.get_gl_tcp("rgt_arm")
    # gm.gen_frame(pos-rot[:,2]*.06,rot).attach_to(base)
    pcd_inhnd = pcd[(pcd[:, 2] > pos[2] + .012) &
                    (pcd[:, 1] > (pos - rot[:, 2] * .051)[1]) &
                    (pcd[:, 0] > pos[0] - .012)]

    # gm.gen_pointcloud(points=pcd_inhnd).attach_to(base)
    import basis.trimesh as tm
    import vision.depth_camera.util_functions as dcuf

    pcd_trimesh = tm.Trimesh(vertices=dcuf.remove_outlier(src_nparray=pcd_inhnd.copy(),
                                                          downsampling_voxelsize=0.001,
                                                          radius=0.003))
    obb_gm = gm.GeometricModel(initor=pcd_trimesh.bounding_box_oriented)
    # bbox_pos = np.average(pcd_trimesh.bounding_box_oriented.vertices,axis=0)
    # obb_gm.set_scale(scale=[1.2, 1.2, 1.2])
    # obb_gm.objpdnp.set_scale(1.2)
    # obb_gm.set_pos(bbox_pos)
    obb_gm.set_pos(np.array([0,0,.001]))
    obb_gm.set_rgba([0, 1, 0, .3])
    obb_gm.attach_to(base)

base = boost_base(base)
base.boost.add_task(zoombase, args=[base, np.array([1, 1, 1]) * .8], timestep=0.2)
# base.boost.add_task(zoombase, args=[base, np.array([1, 1, 1])*1], timestep=0.2)
base.boost.add_task(lambda task: base.boost.screen_shot("pcd"), timestep=0.4)
base.boost.add_task(lambda task: exit(), timestep=0.6)

base.run()
