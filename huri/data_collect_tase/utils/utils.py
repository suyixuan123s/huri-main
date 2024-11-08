"""
Functions to check if the hand is collided by other parts
Author: Chen Hao
Email: chen960216@gmail.com
Date 2022.06.22
"""

from typing import List

import scipy
import numpy as np

import basis.robot_math as rm
import basis.trimesh as tm
import modeling.collision_model as cm
import modeling.model_collection as mc


def is_pose_occluded(rbt_hand_cm_list: List[cm.CollisionModel],
                     rbt_other_cm_list: List[cm.CollisionModel],
                     cam_origin: np.ndarray,
                     toggle_debug: bool = False) -> bool:
    """
    Check if the models of robot hand is occluded by the models of robot others
    :param rbt_hand_cm_list: List contains models for the robot hand
    :param rbt_other_cm_list: List contains models for the robot others
    :param cam_origin: The origin of the camera
    :param toggle_debug: is debug
    :return: True for occlusion and False vice versa
    """
    vertics = np.array([])
    for md in rbt_hand_cm_list:
        if toggle_debug:
            md.attach_to(base)
        vertics = np.append(vertics, rm.homomat_transform_points(md.get_homomat(), md.objtrm.vertices))
    vertics = vertics.reshape(-1, 3)
    ttt = tm.Trimesh(vertices=vertics)
    ttt_convex_hull = ttt.convex_hull
    ttt_super_faces, ttt_super_face_normals, _ = ttt_convex_hull.facets_over()
    vertics = np.array([])
    for i in ttt_super_faces[np.dot(ttt_super_face_normals, np.array([0, 0, -1])) < -.5]:
        vertics = np.append(vertics, ttt_convex_hull.vertices[ttt_convex_hull.faces[i]])

    vertics = vertics.reshape(-1, 3)
    ttt = tm.Trimesh(vertices=vertics)
    ttt_convex_hull = ttt.convex_hull
    hull = scipy.spatial.ConvexHull(ttt_convex_hull.vertices[:, :2])
    is_occluded = False
    for i in ttt_convex_hull.vertices[hull.vertices]:
        for md in rbt_other_cm_list:
            if toggle_debug:
                md.attach_to(base)
            contact_points, contact_normals = md.ray_hit(i, cam_origin)
            if len(contact_points) > 0:
                is_occluded = True
                # for jj in contact_points:
                #     gm.gen_sphere(jj,radius=.005).attach_to(base)
    if toggle_debug:
        hnd_view = cm.CollisionModel(ttt_convex_hull)
        hnd_view.set_pos(hnd_view.get_pos() + np.array([0, 0, 0.001]))
        hnd_view.set_rgba([1, 1, 0, .5])
        hnd_view.attach_to(base)
        cone = cm.CollisionModel(tm.Trimesh(
            vertices=np.append(ttt_convex_hull.vertices[hull.vertices], cam_origin).reshape(-1, 3)).convex_hull)
        if not is_occluded:
            cone.set_rgba(np.array([0, 1, 0, .4]))
        else:
            cone.set_rgba(np.array([1, 0, 0, .4]))
        cone.attach_to(base)
    return is_occluded


def is_yumi_rgt_arm_pose_occluded(rbt_cm_list: List[cm.CollisionModel],
                                  cam_origin: np.ndarray,
                                  toggle_debug: bool = False) -> bool:
    """This function is only defined for the YUMI robot"""
    if isinstance(rbt_cm_list, mc.ModelCollection):
        rbt_cm_list = rbt_cm_list.cm_list
    rbt_hand_cm_list = rbt_cm_list[24:27]
    rbt_other_cm_list = rbt_cm_list[18:21]
    return is_pose_occluded(rbt_hand_cm_list=rbt_hand_cm_list,
                            rbt_other_cm_list=rbt_other_cm_list,
                            cam_origin=cam_origin,
                            toggle_debug=toggle_debug)


def is_yumi_lft_arm_pose_occluded(rbt_cm_list: List[cm.CollisionModel],
                                  cam_origin: np.ndarray,
                                  toggle_debug: bool = False) -> bool:
    """This function is only defined for the YUMI robot"""
    if isinstance(rbt_cm_list, mc.ModelCollection):
        rbt_cm_list = rbt_cm_list.cm_list
    rbt_hand_cm_list = rbt_cm_list[15:17]
    rbt_other_cm_list = rbt_cm_list[9:12]
    return is_pose_occluded(rbt_hand_cm_list=rbt_hand_cm_list,
                            rbt_other_cm_list=rbt_other_cm_list,
                            cam_origin=cam_origin,
                            toggle_debug=toggle_debug)


if __name__ == "__main__":
    from huri.core.common_import import wd, ym, np, fs, gm, rm, cm

    base = wd.World(cam_pos=[3.4, 0, .6], lookat_pos=[.3, 0, .5])
    yumi_s = ym.Yumi(enable_cc=True)
    phoxi_origin = np.array([0.3073, -0.17272, 1.02258996])

    # check for the right arm
    rgt_arm_jnts = np.array([0.99295332, -1.5254463, -1.5292628, 0.31219917, -0.33950902,
                             1.15320438, -0.85455832])
    yumi_s.fk('rgt_arm', rgt_arm_jnts)
    rbt_cm_list = yumi_s.gen_meshmodel().cm_list
    rbt_hand_cm_list = rbt_cm_list[24:27]
    rbt_other_cm_list = rbt_cm_list[18:21]
    is_rgt_occluded = is_pose_occluded(rbt_hand_cm_list=rbt_hand_cm_list, rbt_other_cm_list=rbt_other_cm_list,
                                       cam_origin=phoxi_origin, toggle_debug=True)
    print(f'Is rgt arm pose occluded:{is_rgt_occluded}')

    # check for the left arm
    lft_arm_jnts = np.array([-0.62437167, -2.21166331, 2.13093481, 0.65720134, 0.4071338,
                             0.81011842, 1.42997275])
    yumi_s.fk('lft_arm', lft_arm_jnts)
    rbt_cm_list = yumi_s.gen_meshmodel().cm_list
    rbt_hand_cm_list = rbt_cm_list[15:17]
    rbt_other_cm_list = rbt_cm_list[9:12]
    is_lft_occluded = is_pose_occluded(rbt_hand_cm_list=rbt_hand_cm_list, rbt_other_cm_list=rbt_other_cm_list,
                                       cam_origin=phoxi_origin, toggle_debug=True)
    print(f'Is lft arm pose occluded:{is_lft_occluded}')

    base.run()
