import time
from typing import List

from basis.robot_math import homomat_transform_points
from huri.components.yumi_control.yumi_con import YumiController
from huri.core.common_import import *
from direct.stdpy import threading
from huri.components.gui.tk_gui.base import GuiFrame
from huri.components.gui.tk_gui.widgets.scale import Scale


class Jogging_Animation(threading.Thread):
    def __init__(self, yumi_s: ym.Yumi,
                 yumi_con: YumiController,
                 yumi_con_lock: threading.Lock,
                 base: wd.World,
                 scalewidgets_rgt: List[Scale],
                 scalewidgets_lft: List[Scale],
                 ):
        super().__init__()
        self.yumi_con = yumi_con
        self.yumi_con_lock = yumi_con_lock
        self.yumi_s = yumi_s
        self.yumi_s_cm = self.yumi_s.gen_meshmodel()
        self.scalewidgets_rgt = scalewidgets_rgt
        self.scalewidgets_lft = scalewidgets_lft
        self.base = base
        self._trystop = False
        self._sync_value = False

    def run(self):
        while not self._trystop:
            with self.yumi_con_lock:
                try:
                    if yumi_con is not None:
                        rgt_arm_jnts = yumi_con.get_jnt_values(component_name="rgt_arm")
                        lft_arm_jnts = yumi_con.get_jnt_values(component_name="lft_arm")
                    else:
                        rgt_arm_jnts = yumi_s.get_jnt_values(component_name="rgt_arm")
                        lft_arm_jnts = yumi_s.get_jnt_values(component_name="lft_arm")
                    value_scale_widge_rgt = np.zeros(7)
                    value_scale_widge_lft = np.zeros(7)
                    if not self._sync_value:
                        for idx, scale_widget in enumerate(self.scalewidgets_rgt):
                            value_scale_widge_rgt[idx] = scale_widget.get()
                        for idx, scale_widget in enumerate(self.scalewidgets_lft):
                            value_scale_widge_lft[idx] = scale_widget.get()
                        if np.linalg.norm(lft_arm_jnts - value_scale_widge_lft) > .05:
                            for idx, scale_widget in enumerate(self.scalewidgets_lft):
                                scale_widget.set(lft_arm_jnts[idx])
                        if np.linalg.norm(rgt_arm_jnts - value_scale_widge_rgt) > .05:
                            for idx, scale_widget in enumerate(self.scalewidgets_rgt):
                                scale_widget.set(rgt_arm_jnts[idx])
                        self._sync_value = True
                    self.yumi_s.fk(component_name="rgt_arm", jnt_values=rgt_arm_jnts)
                    self.yumi_s.fk(component_name="lft_arm", jnt_values=lft_arm_jnts)
                    self.yumi_s.jaw_to(hnd_name="rgt_hnd",jaw_width=0.05)
                    self.yumi_s_cm.remove()
                    self.yumi_s_cm = yumi_s.gen_meshmodel()
                    self.yumi_s_cm.attach_to(self.base)
                except Exception as e:
                    print(f"Error:{e}")
                time.sleep(0.3)
                self._lasttime = time.time()

    def close(self):
        self._trystop = True
        self.join()


class PointCloud_Acquire(threading.Thread):
    def __init__(self, sensor_handle, affine_matrix: np.ndarray):
        super().__init__()
        self.sensor_handle = sensor_handle
        self.affine_matrix = affine_matrix
        self.pcd_gm = None
        self._trystop = False

    def run(self):
        while not self._trystop:
            pcd = self.sensor_handle.get_pcd()
            if self.pcd_gm is not None:
                self.pcd_gm.remove()
            self.pcd_gm = gm.gen_pointcloud(homomat_transform_points(affine_matrix,pcd), [[0, 0, 1, .3]])
            self.pcd_gm.attach_to(base)
            time.sleep(20)

    def close(self):
        self._trystop = True
        self.join()


class Jogging_Controller:
    def __init__(self, yumi_s: ym.Yumi, yumi_con: YumiController, base: wd.World):
        self.yumi_s = yumi_s
        self.yumi_con = yumi_con
        self.yumi_con_lock = threading.Lock()
        self.jnt_limits_lft = yumi_s.get_jnt_ranges(component_name="lft_arm")
        self.jnt_limits_rgt = yumi_s.get_jnt_ranges(component_name="lft_arm")
        self._panel = GuiFrame(base.tkRoot)
        self._panel.add_title("Control for the rgt arm", pos=(0, 0))
        self._scale_widgets_rgt = []
        self._scale_widgets_lft = []
        self.timestamp = time.time()
        for i in range(len(self.jnt_limits_rgt)):
            self._scale_widgets_rgt.append(
                self._panel.add_scale(f"rgt joint {i}", command=self.update_rgt_jnts, val_range=self.jnt_limits_rgt[i],
                                      pos=(1, i)))
        self._panel.add_title("Control for the lft arm", pos=(2, 0))
        for i in range(len(self.jnt_limits_lft)):
            self._scale_widgets_lft.append(
                self._panel.add_scale(f"lft joint {i}", command=self.update_lft_jnts, val_range=self.jnt_limits_lft[i],
                                      pos=(3, i)))

        self.jogging_animation_sync = Jogging_Animation(yumi_s=yumi_s,
                                                        yumi_con=yumi_con,
                                                        yumi_con_lock=self.yumi_con_lock,
                                                        base=base,
                                                        scalewidgets_lft=self._scale_widgets_lft,
                                                        scalewidgets_rgt=self._scale_widgets_rgt)
        self.jogging_animation_sync.start()

    def get_scale_widgets_rgt(self):
        value_scale_widge_rgt = np.zeros(7)
        for idx, scale_widget in enumerate(self._scale_widgets_rgt):
            value_scale_widge_rgt[idx] = scale_widget.get()
        return value_scale_widge_rgt

    def get_scale_widgets_lft(self):
        value_scale_widge_lft = np.zeros(7)
        for idx, scale_widget in enumerate(self._scale_widgets_lft):
            value_scale_widge_lft[idx] = scale_widget.get()
        return value_scale_widge_lft

    def update_rgt_jnts(self, val):
        jnts_rgt = self.get_scale_widgets_rgt()
        self.yumi_s.fk(component_name="rgt_arm", jnt_values=jnts_rgt)
        if self.yumi_s.is_collided():
            self.jogging_animation_sync._sync_value = False
        else:
            with self.yumi_con_lock:
                self.yumi_con.move_jnts(component_name="rgt_arm", jnt_vals=jnts_rgt)

    def update_lft_jnts(self, val):
        jnts_lft = self.get_scale_widgets_lft()
        self.yumi_s.fk(component_name="lft_arm", jnt_values=jnts_lft)
        if self.yumi_s.is_collided():
            self.jogging_animation_sync._sync_value = False
        else:
            with self.yumi_con_lock:
                self.yumi_con.move_jnts(component_name="lft_arm", jnt_vals=jnts_lft)

    def __del__(self):
        self.jogging_animation_sync.close()


if __name__ == "__main__":
    from huri.core.file_sys import load_json, workdir

    base = wd.World(cam_pos=[3, 1, 1.5], lookat_pos=[0, 0, 0.7])
    yumi_con = YumiController()
    # yumi_con = None
    yumi_s = ym.Yumi()
    rgt_arm_jnt_init = yumi_s.get_jnt_values(component_name="rgt_arm")
    lft_arm_jnt_init = yumi_s.get_jnt_values(component_name="lft_arm")
    # if yumi_con is not None:
    #     yumi_con.move_jnts(component_name="rgt_arm", jnt_vals=rgt_arm_jnt_init)
    #     yumi_con.move_jnts(component_name="lft_arm", jnt_vals=lft_arm_jnt_init)
    affine_matrix = np.asarray(load_json(workdir / "data/calibration/affine_mat_20210727-162433_r.json")['affine_mat'])
    # pcd = PointCloud_Acquire(sensor_handle=SensorMarkerHandler(), affine_matrix=affine_matrix)
    # pcd.start()
    test = Jogging_Controller(yumi_s=yumi_s, yumi_con=yumi_con, base=base)
    base.startTk()
    base.tkRoot.withdraw()
    base.run()
    exit(0)
# base.run()
# rbt_x.agv_move(agv_linear_speed=-.1, agv_angular_speed=.1, time_interval=5)
# agv_linear_speed = .2
# agv_angular_speed = .5
# arm_linear_speed = .03
# arm_angular_speed = .1
# while True:
#     pressed_keys = {'w': keyboard.is_pressed('w'),
#                     'a': keyboard.is_pressed('a'),
#                     's': keyboard.is_pressed('s'),
#                     'd': keyboard.is_pressed('d'),
#                     'r': keyboard.is_pressed('r'),  # x+ global
#                     't': keyboard.is_pressed('t'),  # x- global
#                     'f': keyboard.is_pressed('f'),  # y+ global
#                     'g': keyboard.is_pressed('g'),  # y- global
#                     'v': keyboard.is_pressed('v'),  # z+ global
#                     'b': keyboard.is_pressed('b'),  # z- gglobal
#                     'y': keyboard.is_pressed('y'),  # r+ global
#                     'u': keyboard.is_pressed('u'),  # r- global
#                     'h': keyboard.is_pressed('h'),  # p+ global
#                     'j': keyboard.is_pressed('j'),  # p- global
#                     'n': keyboard.is_pressed('n'),  # yaw+ global
#                     'm': keyboard.is_pressed('m'),  # yaw- global
#                     'o': keyboard.is_pressed('o'),  # gripper open
#                     'p': keyboard.is_pressed('p')}  # gripper close
#     values_list = list(pressed_keys.values())
#     if pressed_keys["w"] and pressed_keys["a"]:
#         rbt_x.agv_move(linear_speed=agv_linear_speed, angular_speed=agv_angular_speed, time_interval=.5)
#     elif pressed_keys["w"] and pressed_keys["d"]:
#         rbt_x.agv_move(linear_speed=agv_linear_speed, angular_speed=-agv_angular_speed, time_interval=.5)
#     elif pressed_keys["s"] and pressed_keys["a"]:
#         rbt_x.agv_move(linear_speed=-agv_linear_speed, angular_speed=-agv_angular_speed, time_interval=.5)
#     elif pressed_keys["s"] and pressed_keys["d"]:
#         rbt_x.agv_move(linear_speed=-agv_linear_speed, angular_speed=agv_angular_speed, time_interval=.5)
#     elif pressed_keys["w"] and sum(values_list) == 1:  # if key 'q' is pressed
#         rbt_x.agv_move(linear_speed=agv_linear_speed, angular_speed=0, time_interval=.5)
#     elif pressed_keys["s"] and sum(values_list) == 1:  # if key 'q' is pressed
#         rbt_x.agv_move(linear_speed=-agv_linear_speed, angular_speed=0, time_interval=.5)
#     elif pressed_keys["a"] and sum(values_list) == 1:  # if key 'q' is pressed
#         rbt_x.agv_move(linear_speed=0, angular_speed=agv_angular_speed, time_interval=.5)
#     elif pressed_keys["d"] and sum(values_list) == 1:  # if key 'q' is pressed
#         rbt_x.agv_move(linear_speed=0, angular_speed=-agv_angular_speed, time_interval=.5)
#     elif pressed_keys["o"] and sum(values_list) == 1:  # if key 'q' is pressed
#         rbt_x.arm_jaw_to(jawwidth=100)
#     elif pressed_keys["p"] and sum(values_list) == 1:  # if key 'q' is pressed
#         rbt_x.arm_jaw_to(jawwidth=0)
#     elif any(pressed_keys[item] for item in ['r', 't', 'f', 'g', 'v', 'b', 'y', 'u', 'h', 'j', 'n', 'm']) and \
#             sum(values_list) == 1:  # global
#         tic = time.time()
#         current_jnt_values = rbt_s.get_jnt_values()
#         current_arm_tcp_pos, current_arm_tcp_rotmat = rbt_s.get_gl_tcp()
#         rel_pos = np.zeros(3)
#         rel_rotmat = np.eye(3)
#         if pressed_keys['r']:
#             rel_pos = np.array([arm_linear_speed * .5, 0, 0])
#         elif pressed_keys['t']:
#             rel_pos = np.array([-arm_linear_speed * .5, 0, 0])
#         elif pressed_keys['f']:
#             rel_pos = np.array([0, arm_linear_speed * .5, 0])
#         elif pressed_keys['g']:
#             rel_pos = np.array([0, -arm_linear_speed * .5, 0])
#         elif pressed_keys['v']:
#             rel_pos = np.array([0, 0, arm_linear_speed * .5])
#         elif pressed_keys['b']:
#             rel_pos = np.array([0, 0, -arm_linear_speed * .5])
#         elif pressed_keys['y']:
#             rel_rotmat = rm.rotmat_from_euler(arm_angular_speed * .5, 0, 0)
#         elif pressed_keys['u']:
#             rel_rotmat = rm.rotmat_from_euler(-arm_angular_speed * .5, 0, 0)
#         elif pressed_keys['h']:
#             rel_rotmat = rm.rotmat_from_euler(0, arm_angular_speed * .5, 0)
#         elif pressed_keys['j']:
#             rel_rotmat = rm.rotmat_from_euler(0, -arm_angular_speed * .5, 0)
#         elif pressed_keys['n']:
#             rel_rotmat = rm.rotmat_from_euler(0, 0, arm_angular_speed * .5)
#         elif pressed_keys['m']:
#             rel_rotmat = rm.rotmat_from_euler(0, 0, -arm_angular_speed * .5)
#         new_arm_tcp_pos = current_arm_tcp_pos + rel_pos
#         new_arm_tcp_rotmat = rel_rotmat.dot(current_arm_tcp_rotmat)
#         last_jnt_values = rbt_s.get_jnt_values()
#         new_jnt_values = rbt_s.ik(tgt_pos=new_arm_tcp_pos, tgt_rotmat=new_arm_tcp_rotmat,
#                                   seed_jnt_values=current_jnt_values)
#         if new_jnt_values is None:
#             continue
#         rbt_s.fk(jnt_values=new_jnt_values)
#         toc = time.time()
#         start_frame_id = math.ceil((toc - tic) / .01)
#         rbt_x.arm_move_jspace_path([last_jnt_values, new_jnt_values], time_interval=.1, start_frame_id=start_frame_id)
