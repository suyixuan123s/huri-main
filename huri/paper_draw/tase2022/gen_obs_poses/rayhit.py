"""
An example to show the point cloud data capturing from depth sensor
Author: Chen Hao
Email: chen960216@gmail.com
Date 2022.06.22
"""
import numpy as np
import basis.trimesh as tm
import scipy
from huri.core.common_import import rm
import time
import modeling.collision_model as cm


def rayhit_check_2(rbt_model, phoxi_origin, toggle_debug=False):
    rbt_cm_list: List[cm.CollisionModel] = rbt_model.cm_list
    vertics = np.array([])
    for md in rbt_cm_list[24:27]:  # 23:27 hand
        if toggle_debug:
            md.attach_to(base)
        vertics = np.append(vertics, rm.homomat_transform_points(md.get_homomat(), md.objtrm.vertices))
    vertics = vertics.reshape(-1, 3)
    ttt = tm.Trimesh(vertices=vertics)
    ttt_convex_hull = ttt.convex_hull
    # ttt_super_faces, ttt_super_face_normals = ttt_convex_hull.faces, ttt_convex_hull.face_normals
    # rm.homomat_transform_points(md.get_homomat(), ttt_convex_hull.face_normals)
    ttt_super_faces, ttt_super_face_normals, _ = ttt_convex_hull.facets_over()
    a = time.time()
    vertics = np.array([])
    for i in ttt_super_faces[np.dot(ttt_super_face_normals, np.array([0, 0, -1])) < -.5]:
        vertics = np.append(vertics, ttt_convex_hull.vertices[ttt_convex_hull.faces[i]])

    vertics = vertics.reshape(-1, 3)
    ttt = tm.Trimesh(vertices=vertics)
    ttt_convex_hull = ttt.convex_hull
    hull = scipy.spatial.ConvexHull(ttt_convex_hull.vertices[:, :2])
    if toggle_debug:
        hnd_view = cm.CollisionModel(ttt_convex_hull)
        hnd_view.set_pos(hnd_view.get_pos() + np.array([0, 0, 0.001]))
        hnd_view.set_rgba([1, 1, 0, .5])
        print("?")
        hnd_view.attach_to(base)

    is_collided = False
    for i in ttt_convex_hull.vertices[hull.vertices]:
        for md in rbt_cm_list[18:21]:
            if toggle_debug:
                md.attach_to(base)
            contact_points, contact_normals = md.ray_hit(i, phoxi_origin)
            if len(contact_points) > 0:
                is_collided = True
                # for jj in contact_points:
                #     gm.gen_sphere(jj,radius=.005).attach_to(base)
    return is_collided


def rayhit_check_2_lft(rbt_model, phoxi_origin, toggle_debug=False):
    rbt_cm_list: List[cm.CollisionModel] = rbt_model.cm_list
    vertics = np.array([])
    for md in rbt_cm_list[15:17]:  # 23:27 hand
        if toggle_debug:
            md.attach_to(base)
        vertics = np.append(vertics, rm.homomat_transform_points(md.get_homomat(), md.objtrm.vertices))
    vertics = vertics.reshape(-1, 3)
    ttt = tm.Trimesh(vertices=vertics)
    ttt_convex_hull = ttt.convex_hull
    # ttt_super_faces, ttt_super_face_normals = ttt_convex_hull.faces, ttt_convex_hull.face_normals
    # rm.homomat_transform_points(md.get_homomat(), ttt_convex_hull.face_normals)
    ttt_super_faces, ttt_super_face_normals, _ = ttt_convex_hull.facets_over()
    vertics = np.array([])
    for i in ttt_super_faces[np.dot(ttt_super_face_normals, np.array([0, 0, -1])) < -.5]:
        vertics = np.append(vertics, ttt_convex_hull.vertices[ttt_convex_hull.faces[i]])

    vertics = vertics.reshape(-1, 3)
    ttt = tm.Trimesh(vertices=vertics)
    ttt_convex_hull = ttt.convex_hull
    hull = scipy.spatial.ConvexHull(ttt_convex_hull.vertices[:, :2])
    if toggle_debug:
        hnd_view = cm.CollisionModel(ttt_convex_hull)
        hnd_view.set_pos(hnd_view.get_pos() + np.array([0, 0, 0.001]))
        hnd_view.set_rgba([1, 1, 0, .5])
        print("?")
        hnd_view.attach_to(base)
    is_collided = False
    for i in ttt_convex_hull.vertices[hull.vertices]:
        for md in rbt_cm_list[9:12]:
            if toggle_debug:
                md.attach_to(base)
            contact_points, contact_normals = md.ray_hit(i, phoxi_origin)
            if len(contact_points) > 0:
                is_collided = True
                # for jj in contact_points:
                #     gm.gen_sphere(jj,radius=.005).attach_to(base)
    return is_collided


def rayhit_check(rbt_model, phoxi, see_hnd=False):
    rbt_cm_list: List[cm.CollisionModel] = rbt_model.cm_list
    vertics = np.array([])
    # for md in rbt_cm_list[17:21]:
    #     md.attach_to(base)
    for md in rbt_cm_list[23:27]:
        # md.attach_to(base)
        vertics = np.append(vertics, rm.homomat_transform_points(md.get_homomat(), md.objtrm.vertices))
    vertics = vertics.reshape(-1, 3)
    ttt = tm.Trimesh(vertices=vertics)
    ttt_convex_hull = ttt.convex_hull
    ttt_super_faces, ttt_super_face_normals, _ = ttt_convex_hull.facets_over()

    vertics = np.array([])

    # ttt_convex_hull.vertices[ttt_convex_hull.faces[i]].reshape(-1, 3)

    for i in ttt_super_faces[np.dot(ttt_super_face_normals, np.array([0, 0, -1])) < -.5]:
        vertics = np.append(vertics, ttt_convex_hull.vertices[ttt_convex_hull.faces[i]])

    vertics = vertics.reshape(-1, 3)
    ttt = tm.Trimesh(vertices=vertics)
    ttt_convex_hull = ttt.convex_hull
    phoxi_origin = phoxi.get_origin()
    hull = scipy.spatial.ConvexHull(ttt_convex_hull.vertices[:, :2])

    if see_hnd:
        hnd_view = cm.CollisionModel(ttt_convex_hull)
        hnd_view.set_pos(hnd_view.get_pos() + np.array([0, 0, 0.001]))
        hnd_view.set_rgba([1, 1, 0, .5])
        return hnd_view

    is_collided = False
    for i in ttt_convex_hull.vertices[hull.vertices]:
        for md in rbt_cm_list[17:21]:
            contact_points, contact_normals = md.ray_hit(i, phoxi_origin)
            if len(contact_points) > 0:
                is_collided = True
                # for jj in contact_points:
                #     gm.gen_sphere(jj,radius=.005).attach_to(base)
    print(is_collided)
    cone = cm.CollisionModel(tm.Trimesh(
        vertices=np.append(ttt_convex_hull.vertices[hull.vertices], phoxi_origin).reshape(-1, 3)).convex_hull)
    if not is_collided:
        cone.set_rgba(np.array([0, 1, 0, .4]))
    else:
        cone.set_rgba(np.array([1, 0, 0, .4]))
    return cone
    # cone.attach_to(base)


if __name__ == "__main__":
    from typing import List
    from huri.core.common_import import wd, ym, np, fs, gm, rm, cm
    from huri.core.base_boost import zoombase, boost_base
    from huri.paper_draw.utils import phoxi_nocone
    from huri.components.data_annotaion._constants import *

    # Init base
    base = wd.World(cam_pos=[3.4, 0, .6], lookat_pos=[.3, 0, .5])
    yumi_s = ym.Yumi(enable_cc=True)
    # phoxi
    phoxi_nocone.attach_to(base)

    wp_list = []
    wp_cache_path = SEL_PARAM_PATH.joinpath("wp_rgt.cache")
    wp_list.extend(fs.load_pickle(wp_cache_path))
    yumi_s.fk("rgt_arm", wp_list[0])
    yumi_s.jaw_to("rgt_hnd", .035)
    rbt_model_1 = yumi_s.gen_meshmodel()
    cone_1 = rayhit_check(rbt_model_1, phoxi_nocone)

    yumi_s.fk("rgt_arm", wp_list[11])
    rbt_model_2 = yumi_s.gen_meshmodel()
    cone_2 = rayhit_check(rbt_model_2, phoxi_nocone)

    base = boost_base(base)

    img_name = "collision_check"
    base.boost.add_task(zoombase, args=[base, np.array([1, 1, 1]) * .8], timestep=0.2)
    base.boost.add_task(lambda task: rbt_model_1.attach_to(base), timestep=0.2)
    base.boost.add_task(lambda task: cone_1.attach_to(base), timestep=0.2)
    base.boost.add_task(lambda task: base.boost.screen_shot(f"{img_name}_1"), timestep=0.4)
    base.boost.add_task(lambda task: rbt_model_1.detach(), timestep=0.5)
    base.boost.add_task(lambda task: cone_1.detach(), timestep=0.5)
    base.boost.add_task(lambda task: rbt_model_2.attach_to(base), timestep=0.6)
    base.boost.add_task(lambda task: cone_2.attach_to(base), timestep=0.6)
    base.boost.add_task(lambda task: base.boost.screen_shot(f"{img_name}_2"), timestep=0.8)
    base.boost.add_task(lambda task: exit(), timestep=1)
    base.run()
