import numpy as np

from huri.core.common_import import *
from huri.vision.phoxi_capture import SensorMarkerHandler, vision_pipeline

REAL_ROBOT = False
AFFINE_MAT_PATH = fs.workdir / "data" / "calibration" / "qaqqq.json"
pcd_trimesh_boundingbox = [None]


def test(yumi_s, pcd, base):
    # load affine mat
    affine_mat = np.asarray(
        fs.load_json(AFFINE_MAT_PATH)['affine_mat'])
    # Transform and Plot Point Clouds
    pcd_r = rm.homomat_transform_points(affine_mat, points=pcd)

    affine_mat_node = [affine_mat]
    pcd_node = [gm.gen_pointcloud(pcd_r, [[0, 1, 0, .3]])]
    pcd_node[0].attach_to(base)
    angle_resolution = .01
    rgba = [1,0,0,.3]
    def adjust_pcd(affine_mat_node, pcd_node_node, pcd, task):
        if base.inputmgr.keymap["a"]:
            affine_mat_node[0][:3, 3] = affine_mat_node[0][:3, 3] + np.array([0, -.0005, 0])
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.homomat_transform_points(affine_mat_node[0], points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, [rgba])
            pcd_node_node[0].attach_to(base)
            print(repr(affine_mat_node[0]))
        elif base.inputmgr.keymap["d"]:
            affine_mat_node[0][:3, 3] = affine_mat_node[0][:3, 3] + np.array([0, .0005, 0])
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.homomat_transform_points(affine_mat_node[0], points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, [rgba])
            pcd_node_node[0].attach_to(base)
            print(repr(affine_mat_node[0]))
        elif base.inputmgr.keymap["w"]:
            affine_mat_node[0][:3, 3] = affine_mat_node[0][:3, 3] + np.array([-.0005, 0, 0])
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.homomat_transform_points(affine_mat_node[0], points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, [rgba])
            pcd_node_node[0].attach_to(base)
            print(repr(affine_mat_node[0]))
        elif base.inputmgr.keymap["s"]:
            affine_mat_node[0][:3, 3] = affine_mat_node[0][:3, 3] + np.array([.0005, 0, 0])
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.homomat_transform_points(affine_mat_node[0], points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, [rgba])
            pcd_node_node[0].attach_to(base)
            print(repr(affine_mat_node[0]))
        elif base.inputmgr.keymap["q"]:
            affine_mat_node[0][:3, 3] = affine_mat_node[0][:3, 3] + np.array([0, 0, .0005])
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.homomat_transform_points(affine_mat_node[0], points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, [rgba])
            pcd_node_node[0].attach_to(base)
            print(repr(affine_mat_node[0]))
        elif base.inputmgr.keymap["e"]:
            affine_mat_node[0][:3, 3] = affine_mat_node[0][:3, 3] + np.array([0, 0, -.0005])
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.homomat_transform_points(affine_mat_node[0], points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, [rgba])
            pcd_node_node[0].attach_to(base)
            print(repr(affine_mat_node[0]))
        elif base.inputmgr.keymap["z"]:
            affine_mat_node[0][:3, :3] = np.dot(rm.rotmat_from_axangle(np.array([1, 0, 0]), -np.radians(angle_resolution)),
                                                affine_mat_node[0][:3, :3])
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.homomat_transform_points(affine_mat_node[0], points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, [rgba])
            pcd_node_node[0].attach_to(base)
            print(repr(affine_mat_node[0]))
        elif base.inputmgr.keymap["x"]:
            affine_mat_node[0][:3, :3] = np.dot(rm.rotmat_from_axangle(np.array([1, 0, 0]), np.radians(angle_resolution)),
                                                affine_mat_node[0][:3, :3])
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.homomat_transform_points(affine_mat_node[0], points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, [rgba])
            pcd_node_node[0].attach_to(base)
            print(repr(affine_mat_node[0]))
        elif base.inputmgr.keymap["c"]:
            affine_mat_node[0][:3, :3] = np.dot(rm.rotmat_from_axangle(np.array([0, 1, 0]), -np.radians(angle_resolution)),
                                                affine_mat_node[0][:3, :3])
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.homomat_transform_points(affine_mat_node[0], points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, [rgba])
            pcd_node_node[0].attach_to(base)
            print(repr(affine_mat_node[0]))
        elif base.inputmgr.keymap["v"]:
            affine_mat_node[0][:3, :3] = np.dot(rm.rotmat_from_axangle(np.array([0, 1, 0]), np.radians(angle_resolution)),
                                                affine_mat_node[0][:3, :3])
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.homomat_transform_points(affine_mat_node[0], points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, [rgba])
            pcd_node_node[0].attach_to(base)
            print(repr(affine_mat_node[0]))
        elif base.inputmgr.keymap["b"]:
            affine_mat_node[0][:3, :3] = np.dot(rm.rotmat_from_axangle(np.array([0, 0, 1]), -np.radians(angle_resolution)),
                                                affine_mat_node[0][:3, :3])
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.homomat_transform_points(affine_mat_node[0], points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, [rgba])
            pcd_node_node[0].attach_to(base)
            print(repr(affine_mat_node[0]))
        elif base.inputmgr.keymap["n"]:
            affine_mat_node[0][:3, :3] = np.dot(rm.rotmat_from_axangle(np.array([0, 0, 1]), np.radians(angle_resolution)),
                                                affine_mat_node[0][:3, :3])
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.homomat_transform_points(affine_mat_node[0], points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, [rgba])
            pcd_node_node[0].attach_to(base)
            print(repr(affine_mat_node[0]))

        return task.again

    base.boost.add_task(adjust_pcd, args=[affine_mat_node, pcd_node, pcd])

    base.run()


if __name__ == "__main__":
    from huri.core.base_boost import zoombase, boost_base
    from huri.core.constants import SENSOR_INFO

    # Init base
    base = wd.World(cam_pos=[0, 0, 1.5], lookat_pos=[0, 0, 0] , lens_type="perspective")  # , lens_type="orthographic"
    # base = wd.World(cam_pos=[0, 0, 1.5], lookat_pos=[0, 0, 0])
    base = boost_base(base)
    gm.gen_arrow(spos=np.array([.1, 0, 0]), epos=np.array([.8, 0, 0])).attach_to(base)

    gm.gen_frame(pos=np.array([.1, -.3, 0])).attach_to(base)
    gm.gen_frame(pos=np.array([.1, .3, 0])).attach_to(base)

    yumi_s = ym.Yumi(enable_cc=True)
    if REAL_ROBOT:
        from huri.components.yumi_control.yumi_con import YumiController

        yumix = YumiController()
        yumi_s.fk(component_name="rgt_arm",
                  jnt_values=yumix.get_jnt_values(component_name="rgt_arm"))
        yumi_s.fk(component_name="lft_arm",
                  jnt_values=yumix.get_jnt_values(component_name="lft_arm"))
        yumi_s.jaw_to("rgt_hnd", yumix.get_gripper_width("rgt_arm"))
        pose = yumix.rgt_arm_hnd.get_pose()
        gm.gen_frame(pose.translation, pose.rotation).attach_to(base)

        pose_rgt_raw = yumix.rgt_arm_hnd.get_pose()
        pose_r_rgt = rm.homomat_from_posrot(pose_rgt_raw.translation, pose_rgt_raw.rotation)
        pose_s_rgt = rm.homomat_from_posrot(*yumi_s.get_gl_tcp("rgt_arm"))
        print(f"the rgt_arm pose in simulation is:{pose_s_rgt}")
        print(f"the rgt_arm pose in robot is:{pose_r_rgt}")
        print(
            f"the rgt arm pose different between real and sim robot is: {np.linalg.norm(pose_s_rgt[:3, 3] - pose_r_rgt[:3, 3]) * 1000} mm")
        yumix.stop()
    # Generate robot model
    rbt_mesh = yumi_s.gen_meshmodel(toggle_tcpcs=True)
    for cm in rbt_mesh.cm_list:
        rgba = cm.get_rgba()
        cm.set_rgba([rgba[0],rgba[1],rgba[2],.8])
        cm.attach_to(base)
    pcd, img, _, _, _ = vision_pipeline(SensorMarkerHandler(SENSOR_INFO.IP_ADR_DEPTH_SENSOR_DEBUG), dump_path=None,
                                        rgb_texture=False)
    base.boost.add_task(zoombase, args=[base, np.array([1, 0, 0])])
    test(yumi_s, pcd, base)
