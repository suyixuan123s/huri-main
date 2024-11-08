"""
An example to show the point cloud data capturing from depth sensor
Author: Chen Hao
Email: chen960216@gmail.com
Date 2022.06.22
"""

if __name__ == "__main__":
    from typing import List
    from huri.core.common_import import wd, ym, np, fs, gm, rm, cm
    import basis.trimesh as tm
    import scipy
    from huri.core.base_boost import zoombase, boost_base

    # Init base
    base = wd.World(cam_pos=[3.4, 0, .6], lookat_pos=[.3, 0, .5])
    yumi_s = ym.Yumi(enable_cc=True)

    wp_list = []
    from huri.components.data_annotaion._constants import *

    wp_cache_path = SEL_PARAM_PATH.joinpath("wp_rgt.cache")
    wp_list.extend(fs.load_pickle(wp_cache_path))
    yumi_s.fk("rgt_arm", wp_list[101])
    yumi_s.fk("rgt_arm", wp_list[0])
    yumi_s.jaw_to("rgt_hnd",.035)
    rbt_model = yumi_s.gen_meshmodel()
    # rbt_model.attach_to(base)
    rbt_cm_list: List[cm.CollisionModel] = rbt_model.cm_list
    vertics = np.array([])
    # for md in rbt_cm_list[17:21]:
    #     md.attach_to(base)
    for md in rbt_cm_list[23:27]:
        md.attach_to(base)
        vertics = np.append(vertics, rm.homomat_transform_points(md.get_homomat(), md.objtrm.vertices))
    vertics = vertics.reshape(-1, 3)
    ttt = tm.Trimesh(vertices=vertics)
    ttt_convex_hull = ttt.convex_hull
    ttt_super_faces, ttt_super_face_normals,_  = ttt_convex_hull.facets_over()

    vertics = np.array([])

    # ttt_convex_hull.vertices[ttt_convex_hull.faces[i]].reshape(-1, 3)

    for i in ttt_super_faces[np.dot(ttt_super_face_normals, np.array([0, 0, -1])) < -.5]:
        vertics = np.append(vertics, ttt_convex_hull.vertices[ttt_convex_hull.faces[i]])

    vertics = vertics.reshape(-1, 3)
    ttt = tm.Trimesh(vertices=vertics)
    ttt_convex_hull = ttt.convex_hull

    from utils import phoxi_nocone as phoxi

    # phoxi.set_pos(np.array([0.27, 0, 1.05]))
    phoxi.set_pos(np.array([0.3, -.05, 0.5]))
    phoxi.set_rpy(np.radians(90), 0, np.radians(90))
    phoxi.attach_to(base)

    phoxi_origin = phoxi.get_origin()
    hull = scipy.spatial.ConvexHull(ttt_convex_hull.vertices[:,:2])


    hhh = cm.CollisionModel(ttt_convex_hull)
    hhh.set_pos(hhh.get_pos()+np.array([0,0,0.001]))
    hhh.set_rgba([1,1,0,.5])
    hhh.attach_to(base)

    is_collided = False
    for i in ttt_convex_hull.vertices[hull.vertices]:
        for md in rbt_cm_list[17:21]:
            contact_points, contact_normals = md.ray_hit(i, phoxi_origin)
            if len(contact_points) >0:
                is_collided = True
                # for jj in contact_points:
                #     gm.gen_sphere(jj,radius=.005).attach_to(base)
    print(is_collided)
    cone = cm.CollisionModel(tm.Trimesh(vertices=np.append(ttt_convex_hull.vertices[hull.vertices], phoxi_origin).reshape(-1,3)).convex_hull)
    if not is_collided:
        cone.set_rgba(np.array([0, 1, 0, .4]))
    else:
        cone.set_rgba(np.array([1, 0, 0, .4]))
    # cone.attach_to(base)

    base = boost_base(base)

    img_name = "seehand"
    # base.boost.add_task(zoombase, args=[base, np.array([1, 1, 1])*.8], timestep=0.2)
    # base.boost.add_task(lambda task: base.boost.screen_shot(img_name), timestep=0.4)
    # base.boost.add_task(lambda task: exit(), timestep=0.6)
    base.run()
