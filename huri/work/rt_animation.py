from huri.core.common_import import wd, np, fs, gm, rm
import copy
from direct.stdpy import threading
import time
from direct.gui.OnscreenText import OnscreenText
from huri.core.constants import SENSOR_INFO
import cv2
from huri.vision.phoxi_capture import enhance_gray_img
from robot_sim.robots.xarm_lite6_wrs import XArmLite6WRSGripper

base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
# mb_cnt, mb_data = fs.load_pickle("D:\chen\huri_shared\huri\components\exe\\version\\animation.pkl")
# ANIMATION_PATH = fs.Path(" ")
# PCD_PATH = fs.Path("")
DEBUG_DATA_PATH = fs.Path("pcd_debug_data")
DEBUG_DATA_LOCK_PATH = fs.Path(".pcd_debug_data")

_mb_data = [None]
_mb_cnt = [-1]
_data = [None, None, None]
plot_node = [None]

sim_rbt = XArmLite6WRSGripper()


# add on screen image
# font = loader.loadFont('arial.ttf')
# textObject = OnscreenText(text='my text string',
#                           pos=(-1.1, .8),
#                           scale=0.1,
#                           bg=(.3, .3, .3, 1),
#                           fg=(203 / 255, 185 / 255, 148 / 255, 1),
#                           font=font)


# TODO

# def load_animation_files():
#     while True:
#         if ANIMATION_PATH.exists():
#             try:
#                 mb_cnt, mb_data = fs.load_pickle(ANIMATION_PATH)
#                 if _mb_cnt[0] == mb_cnt:
#                     pass
#                 else:
#                     _mb_cnt[0] = mb_cnt
#                     _mb_data[0] = mb_data
#             except:
#                 pass
#         time.sleep(.5)


def load_pcd_files():
    while True:
        if not DEBUG_DATA_LOCK_PATH.exists():
            pcd_w, im, rack, rbt_jnt, _rack_tf, detector, rack_tf = fs.load_pickle(DEBUG_DATA_PATH)
            _data[0] = pcd_w
            _data[1] = rack
            _data[2] = rbt_jnt
            time.sleep(.5)


# thread1 = threading.Thread(target=load_animation_files)
# thread1.start()

thread2 = threading.Thread(target=load_pcd_files)
thread2.start()

affine_matrix = np.asarray(fs.load_json(SENSOR_INFO.PNT_CLD_CALIBR_MAT_PATH)["affine_mat"])


def update(
        pcd_data,
        task):
    if plot_node[0] is not None:
        plot_node[0].remove()
    if pcd_data[0] is not None:
        pcd = _data[0]
        rack = _data[1]
        rbt_jnt = _data[2]
        plot_node[0] = gm.gen_pointcloud(pcd)
        rack.gen_mesh_model(gen_tube=True).attach_to(plot_node[0])
        sim_rbt.fk("arm", rbt_jnt)
        sim_rbt.gen_meshmodel().attach_to(plot_node[0])
        plot_node[0].attach_to(base)
    return task.again


taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[
                          _data],
                      appendTask=True)
base.run()
