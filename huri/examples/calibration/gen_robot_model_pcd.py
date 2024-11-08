import numpy as np

from huri.core.common_import import *
from huri.core.file_sys import Path, load_pickle, workdir_vision
import huri.vision.pnt_utils as pntu
from huri.vision.phoxi_capture import SensorMarkerHandler, vision_pipeline
import huri.vision.yolo.detect as yyd
import huri.core.base_boost as bb

# bad performance:
# 30 32 35 38 4 42 20(11)

pcd_trimesh_boundingbox = [None]


def input_func_factory(is_data_from_server=False, file_path=None, yolo=False):
    if not is_data_from_server:
        assert file_path is not None
        files = list(file_path.glob("*"))

        def input_func(counter):
            counter = min(max(0, counter), len(files) - 1)
            filename = files[counter]
            print(f"Running {filename}")
            _, enhanced_img, img, result, pcd = load_pickle(filename)
            if yolo:
                img, result = yyd.detect(np.stack((enhanced_img,) * 3, axis=-1))
            return img, result, pcd
    else:
        streamer = SensorMarkerHandler()

        def input_func(counter):
            # in this case, counter is useless
            pcd, texture_img = streamer.get_pcd_and_texuture()
            enhanced_image = cv2.equalizeHist(texture_img)
            img, result = yyd.detect(np.stack((enhanced_image,) * 3, axis=-1))
            return img, result, pcd
    return input_func


def concatenate_list(_list, axis=0):
    result = _list[0]
    for val in _list[1:]:
        result = np.concatenate((result, val), axis=axis)
    return result


def sample_arm(yumi, arm_name, radius=0.003):
    arm = yumi.rgt_arm
    if "lft" in arm_name:
        arm = yumi.lft_arm
    arm_mesh = arm.gen_meshmodel(tcp_loc_pos=None,
                                 tcp_loc_rotmat=None,
                                 toggle_tcpcs=False)
    arm_sample_list = []
    for i in range(len(arm_mesh.cm_list)):
        arm_sample_list.append(arm_mesh.cm_list[i].sample_surface(radius=radius)[0])
    arm_sample_points = concatenate_list(arm_sample_list)
    return arm_sample_points


def sample_hnd(yumi, hnd_name, contain_fingers=True, radius=0.003):
    hnd = yumi.rgt_hnd
    if "lft" in hnd_name:
        hnd = yumi.lft_hnd
    hnd_mesh = hnd.gen_meshmodel(tcp_loc_pos=None,
                                 tcp_loc_rotmat=None,
                                 toggle_tcpcs=False)
    hnd_sample_list = []
    if contain_fingers:
        bound = len(hnd_mesh.cm_list)
    else:
        bound = 1
    for i in range(bound):
        hnd_sample_list.append(hnd_mesh.cm_list[i].sample_surface(radius=radius)[0])
    hnd_sample_points = concatenate_list(hnd_sample_list)
    return hnd_sample_points


def sample_body(yumi, radius=0.003):
    body_mesh = yumi.lft_body.gen_meshmodel(tcp_loc_pos=None,
                                            tcp_loc_rotmat=None,
                                            toggle_tcpcs=False)
    body_sample_list = []
    body_sample_list.append(body_mesh.cm_list[1].sample_surface(radius=radius)[0])
    sample_points = concatenate_list(body_sample_list)
    return sample_points


def sample_yumi_pnts(yumi, pcd):
    # init base
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

    yumi.gen_meshmodel(rgba=[0, 0, 1, 0.3]).attach_to(base)

    # rgt_arm_points = sample_arm(yumi, "rgt_arm", radius=0.003)
    #

    # arm_points_c = np.concatenate((rgt_arm_points, rgt_hnd_points))
    # body_points = fs.load_pickle(fs.workdir/"data"/"calibration"/"body_cloud.pkl")
    # rgt_hnd_points = sample_hnd(yumi, "rgt_hnd", radius=0.002, contain_fingers=False)
    # lft_hnd_points = sample_hnd(yumi, "lft_hnd", radius=0.002, contain_fingers=False)
    body_points = sample_body(yumi,radius=.0037)
    # body_points = fs.load_json(fs.workdir / "data" / "calibration" / "body_points.json")
    # sampled_points = concatenate_list((rgt_hnd_points, lft_hnd_points, body_points))
    fs.dump_json(body_points,fs.workdir / "data" / "calibration" / "body_points.json")
    # gm.gen_pointcloud(points=sampled_points).attach_to(base)
    gm.gen_frame().attach_to(base)

    base.run()
    exit(0)
    affine_mat = np.asarray(
        fs.load_json(fs.workdir / "data" / "calibration" / "affine_mat_20210727-162433_rrr.json")['affine_mat'])
    counter = [0]
    pcd_node = [None, None]

    pcd = rm.homomat_transform_points(affine_mat, points=pcd)
    pcd_org = pcd.copy()
    pcd = pcd[np.where((pcd[:, 2] > .07) & (pcd[:, 1] < 0))]
    transform = pntu.icp(src=pcd, tgt=arm_points_c, maximum_distance=0.002)
    pcd_trans = rm.homomat_transform_points(transform, points=pcd)
    pcd_org_trans = rm.homomat_transform_points(transform, points=pcd_org)
    print(transform)
    if pcd_node[0] is not None:
        pcd_node[0].remove()
    if pcd_node[1] is not None:
        pcd_node[1].remove()
    pcd_node[0] = gm.gen_pointcloud(pcd, [[0, 1, 0, .3]])
    pcd_node[0].attach_to(base)
    pcd_node[1] = gm.gen_pointcloud(pcd_trans, [[0, 1, 1, .3]])
    pcd_node[1].attach_to(base)
    gm.gen_pointcloud(pcd_org_trans, [[.6, .6, .3, .3]]).attach_to(base)
    fs.dump_json({'affine_mat': np.dot(transform, affine_mat).tolist()},
                 path=fs.workdir / "data" / "calibration" / "affine_mat_20210727-162433_rrrr.json")

    base.run()


if __name__ == "__main__":
    from huri.components.yumi_control.yumi_con import YumiController
    import time

    REAL_ROBOT = False
    DEBUG = True
    yumi_s = ym.Yumi(enable_cc=True)
    if REAL_ROBOT:
        from huri.components.yumi_control.yumi_con import YumiController

        yumix = YumiController()
        yumi_s.fk(component_name="rgt_arm",
                  jnt_values=yumix.get_jnt_values(component_name="rgt_arm"))
        yumi_s.fk(component_name="lft_arm",
                  jnt_values=yumix.get_jnt_values(component_name="lft_arm"))
        yumi_s.jaw_to("rgt_hnd", 0.0)
    if DEBUG:
        filename = fs.workdir / "data" / "vision_exp" / "20210920-164042.pkl"
        pcd, img = fs.load_pickle(filename)
    else:
        pcd, img = vision_pipeline(SensorMarkerHandler(),
                                   dump_path=fs.workdir / "data" / "vision_exp" / f"{time.strftime('%Y%m%d-%H%M%S')}.pkl")

    sample_yumi_pnts(yumi_s, pcd)
