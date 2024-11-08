import numpy as np
from numpy import array, float32

from huri.core.common_import import fs, wd, ym, gm, rm
from huri.core.constants import SENSOR_INFO
from huri.core.base_boost import zoombase, boost_base
import cv2
from huri.vision.phoxi_capture import enhance_gray_img
from huri.vision.phoxi_capture import vision_read_data
import basis.trimesh as tm
import trimesh as tm2
import vision.depth_camera.util_functions as dcuf
import scipy

base = wd.World(cam_pos=[2.2, 0, 1.2], lookat_pos=[0, 0, .1], w=2000, h=2000)
yumi_s = ym.Yumi(enable_cc=True)

yumi_s.jaw_to("rgt_hnd", 5.3 * 2 / 1000)

base = boost_base(base)
affine_mat = np.asarray(
    fs.load_json(SENSOR_INFO.PNT_CLD_CALIBR_MAT_PATH)['affine_mat'])


def highlight_mask(img, mask, color=(0, 255, 0), show_bbox=False):
    # color to fill
    color = np.array(color, dtype='uint8')

    # equal color where mask, else image
    # this would paint your object silhouette entirely with `color`
    masked_img = np.where(mask[..., None], color, img)

    # use `addWeighted` to blend the two images
    # the object will be tinted toward `color`
    out = cv2.addWeighted(img, 0.8, masked_img, 0.2, 0)

    if show_bbox:
        idx_in_pixel = np.where(mask)
        h1, h2 = idx_in_pixel[0].min(), idx_in_pixel[0].max()
        w1, w2 = idx_in_pixel[1].min(), idx_in_pixel[1].max()
        out = cv2.rectangle(out, (w1, h1), (w2, h2),
                            color=color.tolist(), thickness=2)
    # merge the channels back together
    return out



def map_pcd_img2d_bbox(extracted_pcd_idx, img_sz):
    h, w = img_sz[0], img_sz[1]
    idx_in_pixel = np.unravel_index(extracted_pcd_idx, (h, w))
    h1, h2 = idx_in_pixel[0].min(), idx_in_pixel[0].max()
    w1, w2 = idx_in_pixel[1].min(), idx_in_pixel[1].max()
    return h1, w1, h2, w2


def map_pcd_img2d_mask(extracted_pcd_idx, img_sz, conv_area=False):
    h, w = img_sz[0], img_sz[1]
    idx_in_pixel = np.unravel_index(extracted_pcd_idx, (h, w))
    vertices_candidate = np.vstack((idx_in_pixel[1], idx_in_pixel[0])).T
    if conv_area:
        vertices = vertices_candidate[scipy.spatial.ConvexHull(vertices_candidate).vertices]
        return cv2.fillPoly(np.zeros(img_sz[:2]), pts=[vertices], color=(1))
    else:
        return cv2.fillPoly(np.zeros(img_sz[:2]), pts=[vertices_candidate], color=(1))


bg_pcd, bg_img, _, _, _ = vision_read_data("bg.pkl")
rack_pcd, rack_img, _, _, _ = vision_read_data("rack2.pkl")

is_bg_pcd_capture = True

if is_bg_pcd_capture:
    img = bg_img
else:
    img = rack_img
collected_im = enhance_gray_img(img)
collected_im = cv2.cvtColor(collected_im, cv2.COLOR_GRAY2BGR)
color_c3 = collected_im.copy().reshape(-1, 3)
color_c4 = np.ones((len(color_c3), 4), dtype=float)
color_c4[..., :3] = color_c3 / 255

if is_bg_pcd_capture:
    bg_pcd_align = rm.homomat_transform_points(affine_mat, bg_pcd)
    gm.gen_pointcloud(bg_pcd_align, color_c4).attach_to(base)
    # yumi_s.gen_meshmodel().attach_to(base)
    # base.boost.add_task(zoombase, args=[base, np.array([1, 0, 1]) * .8], timestep=0.2)
    base.boost.add_task(lambda task: base.boost.screen_shot("bg_pcd"), timestep=0.4)
    base.boost.add_task(lambda task: exit(), timestep=0.6)
    base.run()
else:
    rack_pcd_align = rm.homomat_transform_points(affine_mat, rack_pcd)
    gm.gen_pointcloud(rack_pcd_align, color_c4).attach_to(base)

    rack_pcd_rm = rack_pcd_align[(np.linalg.norm(rack_pcd - bg_pcd, axis=1) > 0.003) & (rack_pcd_align[:, 0] > .15) & (
            rack_pcd_align[:, 2] < .3)]

    rm_outlier = dcuf.remove_outlier(src_nparray=rack_pcd_rm,
                                     downsampling_voxelsize=0.005,
                                     nb_points=40, radius=0.005 * 10)
    # print(rm_outlier)

    # gm.gen_pointcloud(bg_pcd).attach_to(base)
    # gm.gen_pointcloud(, rgbas=[(1, 0, 0, 1)],).attach_to(base)
    mask: tm.Trimesh = tm.Trimesh(rm_outlier).bounding_box
    # gm.gen_pointcloud(rack_pcd_align[mask.contains(rack_pcd_align)], rgbas=[(1, 0, 0, 1)], pntsize=7).attach_to(base)
    obb_gm = gm.GeometricModel(initor=mask)
    # obb_gm.
    obb_gm.set_pos(np.array([0, 0, .001]))
    random_color = (1, 0, 0, 1)
    obb_gm.set_rgba([random_color[0], random_color[1], random_color[2], .3])
    # obb_gm.attach_to(base)
    base.boost.add_task(zoombase, args=[base, np.array([1, 0, 1]) * .8], timestep=0.2)
    base.boost.add_task(lambda task: base.boost.screen_shot("rack_pcd"), timestep=0.4)
    base.boost.add_task(lambda task: exit(), timestep=0.6)
    base.run()
    iid = np.where((np.linalg.norm(rack_pcd - bg_pcd, axis=1) > 0.003) & (rack_pcd_align[:, 0] > .15) & (
            rack_pcd_align[:, 2] < .3))[0]
    iiid = np.where(mask.contains(rack_pcd_align[iid]))[0]
    mask = map_pcd_img2d_mask(iid[iiid], img_sz=img.shape, conv_area=True)
    cv2.imwrite("rack.jpg", collected_im, )
    img = highlight_mask(collected_im, mask, color=[0, 0, 255], show_bbox=True)
    cv2.imwrite("hightlight_mask.jpg", img, )


# base.run()
