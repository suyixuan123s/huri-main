"""
Panda3d Extension
Author: Hao Chen
"""
import functools
from typing import Callable

import numpy as np
import cv2
from panda3d.core import (Texture,
                          NodePath,
                          WindowProperties,
                          Vec3,
                          Point3,
                          PerspectiveLens,
                          OrthographicLens,
                          PGTop,
                          GeomNode,
                          PGMouseWatcherBackground,
                          CollisionRay,
                          CollisionHandler,
                          CollisionHandlerQueue,
                          CollisionTraverser,
                          CollisionNode,
                          BitMask32,
                          TransparencyAttrib,
                          TextNode)
from direct.gui.OnscreenImage import OnscreenImage
from direct.gui.OnscreenText import OnscreenText

import basis.trimesh.primitives
import visualization.panda.filter as flt
import visualization.panda.inputmanager as im
import visualization.panda.world as wd
from basis.data_adapter import pdmat4_to_npmat4
from basis.robot_math import homomat_transform_points
from direct.showbase.DirectObject import DirectObject
import modeling.collision_model as cm
import basis.robot_math as rm
import math


def img_to_n_channel(img, channel=3):
    """
    Repeat a channel n times and stack
    :param img:
    :param channel:
    :return:
    """
    return np.stack((img,) * channel, axis=-1)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def gen_cm_frame_xyz(pos=np.array([0, 0, 0]),
                     rotmat=np.eye(3),
                     length=.1,
                     thickness=.005,
                     rgbmatrix=None,
                     alpha=None,
                     plotname="frame"):
    """
    gen an axis for attaching
    :param _pos:
    :param _rotmat:
    :param length:
    :param thickness:
    :param rgbmatrix: each column indicates the color of each base
    :param plotname:
    :return:
    author: weiwei
    date: 20161212tsukuba, 20191228osaka
    """
    mat = np.eye(4)
    mat[:3, 3] = pos
    mat[:3, :3] = rotmat
    _pos = np.array([0, 0, 0])
    _rotmat = np.eye(3)
    endx = _pos + _rotmat[:, 0] * length
    endy = _pos + _rotmat[:, 1] * length
    endz = _pos + _rotmat[:, 2] * length
    if rgbmatrix is None:
        rgbx = np.array([1, 0, 0])
        rgby = np.array([0, 1, 0])
        rgbz = np.array([0, 0, 1])
    else:
        rgbx = rgbmatrix[:, 0]
        rgby = rgbmatrix[:, 1]
        rgbz = rgbmatrix[:, 2]
    if alpha is None:
        alphax = alphay = alphaz = 1
    elif isinstance(alpha, np.ndarray):
        alphax = alpha[0]
        alphay = alpha[1]
        alphaz = alpha[2]
    else:
        alphax = alphay = alphaz = alpha
    # TODO 20201202 change it to StaticGeometricModelCollection
    arrowx_trm = cm.gm.trihelper.gen_arrow(spos=_pos, epos=endx, thickness=thickness)
    arrowx_nodepath = cm.da.trimesh_to_nodepath(arrowx_trm)
    arrowx_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowx_nodepath.setColor(rgbx[0], rgbx[1], rgbx[2], alphax)
    arrowy_trm = cm.gm.trihelper.gen_arrow(spos=_pos, epos=endy, thickness=thickness)
    arrowy_nodepath = cm.da.trimesh_to_nodepath(arrowy_trm)
    arrowy_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowy_nodepath.setColor(rgby[0], rgby[1], rgby[2], alphay)
    arrowz_trm = cm.gm.trihelper.gen_arrow(spos=_pos, epos=endz, thickness=thickness)
    arrowz_nodepath = cm.da.trimesh_to_nodepath(arrowz_trm)
    arrowz_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowz_nodepath.setColor(rgbz[0], rgbz[1], rgbz[2], alphaz)
    arrowx_cm = cm.CollisionModel(arrowx_nodepath)
    arrowy_cm = cm.CollisionModel(arrowy_nodepath)
    arrowz_cm = cm.CollisionModel(arrowz_nodepath)
    arrowx_cm.set_homomat(mat)
    arrowy_cm.set_homomat(mat)
    arrowz_cm.set_homomat(mat)
    return arrowx_cm, arrowy_cm, arrowz_cm


def gen_cm_torque_xyz(pos=np.array([0, 0, 0]),
                      rotmat=np.eye(3),
                      length=.1,
                      thickness=.005,
                      rgbmatrix=None,
                      alpha=None,
                      plotname="frame"):
    """
    gen an axis for attaching
    :param _pos:
    :param _rotmat:
    :param length:
    :param thickness:
    :param rgbmatrix: each column indicates the color of each base
    :param plotname:
    :return:
    author: weiwei
    date: 20161212tsukuba, 20191228osaka
    """
    mat = np.eye(4)
    mat[:3, 3] = pos
    mat[:3, :3] = rotmat
    _pos = np.array([0, 0, 0])
    _rotmat = np.eye(3)
    endx = _pos + _rotmat[:, 0] * length
    endy = _pos + _rotmat[:, 1] * length
    endz = _pos + _rotmat[:, 2] * length
    if rgbmatrix is None:
        rgbx = np.array([1, 0, 0])
        rgby = np.array([0, 1, 0])
        rgbz = np.array([0, 0, 1])
    else:
        rgbx = rgbmatrix[:, 0]
        rgby = rgbmatrix[:, 1]
        rgbz = rgbmatrix[:, 2]
    if alpha is None:
        alphax = alphay = alphaz = 1
    elif isinstance(alpha, np.ndarray):
        alphax = alpha[0]
        alphay = alpha[1]
        alphaz = alpha[2]
    else:
        alphax = alphay = alphaz = alpha
    # TODO 20201202 change it to StaticGeometricModelCollection
    arrowx_trm = cm.gm.trihelper.gen_arrow(spos=_pos, epos=endx, thickness=thickness)
    arrowx_nodepath = cm.da.trimesh_to_nodepath(arrowx_trm)
    arrowx_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowx_nodepath.setColor(rgbx[0], rgbx[1], rgbx[2], alphax)
    arrowy_trm = cm.gm.trihelper.gen_arrow(spos=_pos, epos=endy, thickness=thickness)
    arrowy_nodepath = cm.da.trimesh_to_nodepath(arrowy_trm)
    arrowy_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowy_nodepath.setColor(rgby[0], rgby[1], rgby[2], alphay)
    arrowz_trm = cm.gm.trihelper.gen_arrow(spos=_pos, epos=endz, thickness=thickness)
    arrowz_nodepath = cm.da.trimesh_to_nodepath(arrowz_trm)
    arrowz_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowz_nodepath.setColor(rgbz[0], rgbz[1], rgbz[2], alphaz)
    arrowx_cm = cm.CollisionModel(arrowx_nodepath)
    arrowy_cm = cm.CollisionModel(arrowy_nodepath)
    arrowz_cm = cm.CollisionModel(arrowz_nodepath)
    arrowx_cm.set_homomat(mat)
    arrowy_cm.set_homomat(mat)
    arrowz_cm.set_homomat(mat)
    return arrowx_cm, arrowy_cm, arrowz_cm


def gen_cm_cirframe_xyz(pos=np.array([0, 0, 0]),
                        rotmat=np.eye(3),
                        radius=.1,
                        thickness=.005,
                        rgbmatrix=None,
                        alpha=None,
                        plotname="frame"):
    mat = np.eye(4)
    mat[:3, 3] = pos
    mat[:3, :3] = rotmat
    _pos = np.array([0, 0, 0])
    _rotmat = np.eye(3)
    if rgbmatrix is None:
        rgbx = np.array([1, 0, 0])
        rgby = np.array([0, 1, 0])
        rgbz = np.array([0, 0, 1])
    else:
        rgbx = rgbmatrix[:, 0]
        rgby = rgbmatrix[:, 1]
        rgbz = rgbmatrix[:, 2]
    if alpha is None:
        alphax = alphay = alphaz = 1
    elif isinstance(alpha, np.ndarray):
        alphax = alpha[0]
        alphay = alpha[1]
        alphaz = alpha[2]
    else:
        alphax = alphay = alphaz = alpha
    x_ciraxis = cm.CollisionModel(
        gm.gen_circarrow(axis=_rotmat[:3, 0], center=_pos, thickness=thickness, radius=radius, portion=1))
    y_ciraxis = cm.CollisionModel(
        gm.gen_circarrow(axis=_rotmat[:3, 1], center=_pos, thickness=thickness, radius=radius, portion=1))
    z_ciraxis = cm.CollisionModel(
        gm.gen_circarrow(axis=_rotmat[:3, 2], center=_pos, thickness=thickness, radius=radius, portion=1))
    x_ciraxis_cm = cm.CollisionModel(x_ciraxis)
    y_ciraxis_cm = cm.CollisionModel(y_ciraxis)
    z_ciraxis_cm = cm.CollisionModel(z_ciraxis)
    x_ciraxis_cm.set_rgba([rgbx[0], rgbx[1], rgbx[2], alphax])
    y_ciraxis_cm.set_rgba([rgby[0], rgby[1], rgby[2], alphay])
    z_ciraxis_cm.set_rgba([rgbz[0], rgbz[1], rgbz[2], alphaz])

    x_ciraxis_cm.set_homomat(mat)
    y_ciraxis_cm.set_homomat(mat)
    z_ciraxis_cm.set_homomat(mat)
    return x_ciraxis_cm, y_ciraxis_cm, z_ciraxis_cm


class SelectHandler(object):
    def __init__(self, base, toggle_debug=False):
        super(SelectHandler, self).__init__()
        self.node = None
        self.init_mouse_pos = None
        self.coll_entry = None
        self._base = base
        self.cam = base.cam

        self.listen_obj_func_table = {}
        self.listen_obj_cm_table = {}

        self._keymap = {'control-mouse1': False,
                        'mouse1-up': False}

        self._base.accept('control-mouse1', self._setkeys, ['control-mouse1', True])
        self._base.accept('mouse1-up', self._setkeys, ['control-mouse1', False])

        self.cd_candidates = NodePath("cd_candidate")
        self.cd_candidates.reparent_to(self._base.render)
        self._pick_cnt = 1

        # for drag operation
        self.lastm1pos = None

        # Create collision nodes
        self.ctrav = CollisionTraverser()
        # self.collTrav.showCollisions( render )
        self.chan = CollisionHandlerQueue()
        self.picker_ray = CollisionRay()

        # Create collision ray
        picker_cdnp = CollisionNode("SelectHandler")
        picker_cdnp.addSolid(self.picker_ray)
        picker_cdnp.setIntoCollideMask(BitMask32.allOff())
        picker_np = self.cam.attachNewNode(picker_cdnp)
        self.ctrav.addCollider(picker_np, self.chan)

        # Create collision mask for the ray if one is specified
        # if self.from_collided_mask is not None:
        picker_cdnp.setFromCollideMask(GeomNode.getDefaultCollideMask())

        self.toggle_debug = toggle_debug
        # Bind mouse button events
        # event_names = ['mouse1', 'control-mouse1', 'mouse1-up']
        # self.event_names = ['control-mouse1', 'mouse1-up']
        # for event_name in self.event_names:
        #     self._base.accept(event_name,
        #                       lambda x: self.click_event(x), [event_name])
        #     if toggle_debug:
        #         self._base.accept(event_name,
        #                           lambda x: print(
        #                               f"{self.node}, {self.coll_entry}") if self.node is not None else print(
        #                               "no selection"), [event_name])
        base.taskMgr.add(self.update, "select_handler")
        base.taskMgr.add(self.mouse_drag, "drag_handler")

    def _setkeys(self, key, value):
        self._keymap[key] = value
        return

    def click_event(self, event_name, init_mouse_pos, cur_mouse_pos):
        [_(event_name, init_mouse_pos, cur_mouse_pos, self.get_seleced()) for _ in self.get_seleced_func()]

    def listen_obj(self, obj: 'CollisionModel', callback_func: Callable = None):
        obj.objpdnp.reparent_to(self.cd_candidates)
        obj.objpdnp.set_tag("listened", str(self._pick_cnt))
        self._pick_cnt += 1
        obj_hash = hash(obj.objpdnp)
        if callback_func is not None:
            if obj_hash in self.listen_obj_func_table:
                self.listen_obj_func_table[obj_hash].append(callback_func)
            else:
                self.listen_obj_func_table[obj_hash] = [callback_func]
        if obj_hash not in self.listen_obj_cm_table:
            self.listen_obj_cm_table[obj_hash] = obj

    def get_seleced(self):
        if self.node is None:
            return None
        else:
            obj_hash = hash(self.node.findNetTag("listened"))
            return self.listen_obj_cm_table.get(obj_hash, None)

    def get_seleced_func(self):
        if self.node is None:
            return []
        else:
            obj_hash = hash(self.node.findNetTag("listened"))
            return self.listen_obj_func_table.get(obj_hash, [])

    def mouse_drag(self, task):
        if self._base.mouseWatcherNode.hasMouse():
            if self._keymap["control-mouse1"]:
                if self.coll_entry is None or self.node is None:
                    if self.lastm1pos is not None:
                        self.lastm1pos = None
                    return task.cont
                curm1pos = wd.p3dh.pdv3_to_npv3(self.coll_entry.getSurfacePoint(self.cd_candidates))
                if curm1pos is None:
                    if self.lastm1pos is not None:
                        self.lastm1pos = None
                    return task.cont
                if self.lastm1pos is None:
                    # first time click
                    self.lastm1pos = curm1pos
                    # gm.gen_sphere(pos=np.array([curm1pos[0], curm1pos[1], curm1pos[2]])).attach_to(base)
                    return task.cont
                self.click_event("control-mouse1", init_mouse_pos=self.lastm1pos, cur_mouse_pos=curm1pos)
                self.lastm1pos = curm1pos
            else:
                self.lastm1pos = None
        else:
            self.lastm1pos = None
        return task.cont

    def update(self, task):
        # Update the ray's position
        x, y = None, None
        if self._base.mouseWatcherNode.hasMouse():
            mp = self._base.mouseWatcherNode.getMouse()
            x, y = mp.getX(), mp.getY()
        if x is None or y is None:
            return task.cont
        self.picker_ray.setFromLens(self.cam.node(), x, y)
        # Traverse the hierarchy and find collisions
        self.ctrav.traverse(self.cd_candidates)
        if self.chan.getNumEntries() > 0:
            # If we have hit something, sort the hits so that the closest is first
            self.chan.sortEntries()
            coll_entry = self.chan.getEntry(0)
            node = coll_entry.getIntoNodePath()
            # # If this node is different to the last node, send a mouse leave
            # # event to the last node, and a mouse enter to the new node
            if node != self.node:
                if self.node is not None:
                    self.node.hide_bounds()
            self.coll_entry = coll_entry
            self.node = node
            # self.node = self.node.findNetTag('listened')
            # if self.node.isEmpty():
            #     return task.cont
            self.node.show_tight_bounds()

        elif self.node is not None:
            if not self.node.isEmpty():
                self.node.hide_bounds()
            self.node = None
            if self.toggle_debug:
                print("Leave the last object")
        return task.cont


class InteractiveHandler(object):
    def __init__(self, obj: 'CollisionModel', select_hdl: SelectHandler, toggle_info=True):
        self._obj = obj
        self._select_hdl = select_hdl
        self._gen_ctrl_op()

        self.shift_speed_gain = 1.5
        self.rotate_speed_gain = 5

        self._select_hdl.listen_obj(self.ctrl_op[0], callback_func=self.shift_func_factory(0))
        self._select_hdl.listen_obj(self.ctrl_op[1], callback_func=self.shift_func_factory(1))
        self._select_hdl.listen_obj(self.ctrl_op[2], callback_func=self.shift_func_factory(2))
        # self._select_hdl.listen_obj(self.ctrl_op[3], callback_func=self.rotate_func_factory(0))
        # self._select_hdl.listen_obj(self.ctrl_op[4], callback_func=self.rotate_func_factory(1))
        # self._select_hdl.listen_obj(self.ctrl_op[5], callback_func=self.rotate_func_factory(2))
        self._toggle_info = toggle_info
        if self._toggle_info:
            self._onscreen_txt_xyz = OnscreenText(text='', pos=(-2 + .3, 1 - .1, 0), scale=.1,
                                                  fg=(0., 0., 0., 1),
                                                  align=TextNode.ALeft, mayChange=1)

    def _onscreen_text_xyz(self, txt_list):
        return f"POS:{[_ for _ in txt_list]}"

    def _onscreen_text_rpy(self, txt):
        return f'XYZ: [ ]'

    def _gen_ctrl_op(self):
        obj_pos, obj_rot = self._obj.get_pos(), self._obj.get_rotmat()
        self.ctrl_op = [*gen_cm_frame_xyz(obj_pos, obj_rot, thickness=.007, length=.12),
                        *gen_cm_cirframe_xyz(obj_pos, obj_rot, thickness=.007, radius=.2)]

    def shift_func_factory(self, dir_id):
        def shift_func(event_name, init_mouse_pos, cur_mouse_pos, obj):
            if event_name == "control-mouse1":
                obj_pos, obj_rot = self._obj.get_pos(), self._obj.get_rotmat()
                weight = (cur_mouse_pos - init_mouse_pos).dot(obj_rot[:3, dir_id]) * self.shift_speed_gain
                obj_pos_n = obj_pos
                obj_pos_n += weight * obj_rot[:3, dir_id]
                self._obj.set_pos(obj_pos_n)
                for _ in self.ctrl_op:
                    _.set_pos(obj_pos_n)
                self._onscreen_txt_xyz['text'] = self._onscreen_text_xyz(obj_pos_n)

        return shift_func

    def rotate_func_factory(self, dir_id):
        def rotate_func(event_name, init_mouse_pos, cur_mouse_pos, obj):
            if event_name == "control-mouse1":
                obj_pos, obj_rot = self._obj.get_pos(), self._obj.get_rotmat()
                weight_raw = cur_mouse_pos - init_mouse_pos
                if dir_id == 0:
                    dir_id_proj = 1
                elif dir_id == 1:
                    dir_id_proj = 2
                elif dir_id == 2:
                    dir_id_proj = 0
                weight = np.sign((init_mouse_pos - cur_mouse_pos).dot(obj_rot[:3, dir_id_proj])) * np.linalg.norm(
                    weight_raw) * self.rotate_speed_gain
                obj_rot_n = rm.rotmat_from_axangle(obj_rot[:3, dir_id], weight).dot(obj_rot)
                self._obj.set_rotmat(obj_rot_n)
                for _ in self.ctrl_op:
                    _.set_rotmat(obj_rot_n)

        return rotate_func


class RobotInteractiveHandler(object):
    def __init__(self, rbt_sim, select_hdl: SelectHandler, callback=None):
        self._rbt_sim = rbt_sim
        tcp_pos, tcp_rot = rbt_sim.get_gl_tcp("arm")
        self.ctrl_op = [*gen_cm_frame_xyz(tcp_pos, tcp_rot),
                        *gen_cm_cirframe_xyz(tcp_pos, tcp_rot)]

        # for select handler
        self._select_hdl = select_hdl
        self.plot_node = self._rbt_sim.gen_meshmodel()
        self.callback = callback

        select_hdl.listen_obj(self.ctrl_op[0], callback_func=self.shift_func_factory("arm", 0))
        select_hdl.listen_obj(self.ctrl_op[1], callback_func=self.shift_func_factory("arm", 1))
        select_hdl.listen_obj(self.ctrl_op[2], callback_func=self.shift_func_factory("arm", 2))
        # select_hdl.listen_obj(self.ctrl_op[3], callback_func=self.rotate_func_factory("arm", 0))
        # select_hdl.listen_obj(self.ctrl_op[4], callback_func=self.rotate_func_factory("arm", 1))
        # select_hdl.listen_obj(self.ctrl_op[5], callback_func=self.rotate_func_factory("arm", 2))

        # if callback is not None:
        # select_hdl.listen_obj(self.ctrl_op[0], callback_func=callback)
        # select_hdl.listen_obj(self.ctrl_op[1], callback_func=callback)
        # select_hdl.listen_obj(self.ctrl_op[2], callback_func=callback)
        # select_hdl.listen_obj(self.ctrl_op[3], callback_func=callback)
        # select_hdl.listen_obj(self.ctrl_op[4], callback_func=callback)
        # select_hdl.listen_obj(self.ctrl_op[5], callback_func=callback)

    def _gen_ctrl_op(self):
        tcp_pos, tcp_rot = self._rbt_sim.get_gl_tcp("arm")
        self.ctrl_op = [*gen_cm_frame_xyz(tcp_pos, tcp_rot),
                        *gen_cm_cirframe_xyz(tcp_pos, tcp_rot)]

    def shift_func_factory(self, component_name, dir_id):
        def shift_func(event_name, init_mouse_pos, cur_mouse_pos, obj):
            if event_name == "control-mouse1":
                if self.plot_node is not None:
                    self.plot_node.remove()
                tcp_pos, tcp_rot = self._rbt_sim.get_gl_tcp("arm")
                for _ in self.ctrl_op:
                    _.set_pos(tcp_pos)
                    _.set_rotmat(tcp_rot)
                self.plot_node = self._rbt_sim.gen_meshmodel()
                self.plot_node.attach_to(base)
                weight = max(min((cur_mouse_pos - init_mouse_pos).dot(tcp_rot[:3, dir_id]), .0015), -.0015)
                tcp_pos_n = tcp_pos
                tcp_pos_n += weight * tcp_rot[:3, dir_id]
                jnt = self._rbt_sim.ik(component_name, tgt_pos=tcp_pos_n, tgt_rotmat=tcp_rot,
                                       seed_jnt_values=self._rbt_sim.get_jnt_values(component_name))
                if jnt is not None:
                    self._rbt_sim.fk(component_name, jnt)
                self.callback(weight)

        return shift_func

    def rotate_func_factory(self, component_name, dir_id):
        def rotate_func(event_name, init_mouse_pos, cur_mouse_pos, obj):
            if event_name == "control-mouse1":
                if self.plot_node is not None:
                    self.plot_node.remove()
                tcp_pos, tcp_rot = self._rbt_sim.get_gl_tcp("arm")
                for _ in self.ctrl_op:
                    _.set_pos(tcp_pos)
                    _.set_rotmat(tcp_rot)
                self.plot_node = self._rbt_sim.gen_meshmodel()
                self.plot_node.attach_to(base)
                weight = max(min((cur_mouse_pos - init_mouse_pos).dot(tcp_rot[:3, dir_id]), .0015), -.0015)
                tcp_pos_n = tcp_pos
                tcp_pos_n += weight * tcp_rot[:3, 0]
                jnt = self._rbt_sim.ik(component_name, tgt_pos=tcp_pos_n, tgt_rotmat=tcp_rot,
                                       seed_jnt_values=self._rbt_sim.get_jnt_values(component_name))
                if jnt is not None:
                    self._rbt_sim.fk(component_name, jnt)

        return rotate_func


class ImgOnscreen():
    """
    Add a on screen image in the render 2d scene of Showbase
    """

    def __init__(self,
                 size,
                 pos=(0, 0),
                 parent_np=None):
        """
        :param size: (width, height)
        :param parent_np: Should be ShowBase or ExtraWindow
        """
        self._size = size
        self.tx = Texture("video")
        self.tx.setup2dTexture(size[0], size[1], Texture.TUnsignedByte, Texture.FRgb8)
        # this makes some important setup call
        # self.tx.load(PNMImage(card_size[0], card_size[1]))
        self.onscreen_image = OnscreenImage(self.tx,
                                            pos=(pos[0], 0, pos[1]),
                                            scale=(size[0] / 3840, 1, size[1] /2160 ),
                                            parent=parent_np.render2d)

    def update_img(self, img: np.ndarray):
        """
        Update the onscreen image
        :param img:
        :return:
        """
        if img.shape[2] == 1:
            img = img_to_n_channel(img)
        resized_img = letterbox(img, new_shape=[self._size[1], self._size[0]], auto=False)[0]
        self.tx.setRamImage(resized_img.tostring())

    def remove(self):
        """
        Release the memory
        :return:
        """
        if self.onscreen_image is not None:
            self.onscreen_image.destroy()

    def __del__(self):
        self.remove()


class ExtraWindow():
    """
    Create a extra window on the scene
    :return:
    """

    def __init__(self, base: wd.World,
                 window_title: str = "WRS Robot Planning and Control System",
                 cam_pos: np.ndarray = np.array([2.0, 0, 2.0]),
                 lookat_pos: np.ndarray = np.array([0, 0, 0.25]),
                 up: np.ndarray = np.array([0, 0, 1]),
                 fov: int = 40,
                 w: int = 1920,
                 h: int = 1080,
                 lens_type: str = "perspective",
                 cartoon_render=False,
                 auto_cam_rotate=False):
        self._base = base
        # setup render scene
        self.render = NodePath("extra_win_render")
        # setup render 2d
        self.render2d = NodePath("extra_win_render2d")
        self.render2d.setDepthTest(0)
        self.render2d.setDepthWrite(0)

        self.win = base.openWindow(props=WindowProperties(base.win.getProperties()),
                                   makeCamera=False,
                                   scene=self.render,
                                   requireWindow=True, )

        # set window background to white
        base.setBackgroundColor(r=1, g=1, b=1, win=self.win)
        # set window title and window's dimension
        self.set_win_props(title=window_title,
                           size=(w, h), )
        # set len for the camera and set the camera for the new window
        lens = PerspectiveLens()
        lens.setFov(fov)
        lens.setNearFar(0.001, 5000.0)
        if lens_type == "orthographic":
            lens = OrthographicLens()
            lens.setFilmSize(1, 1)
        # make aspect ratio looks same as base window
        aspect_ratio = base.getAspectRatio()
        lens.setAspectRatio(aspect_ratio)
        self.cam = base.makeCamera(self.win, scene=self.render, )  # can also be found in base.camList
        self.cam.reparentTo(self.render)
        self.cam.setPos(Point3(cam_pos[0], cam_pos[1], cam_pos[2]))
        self.cam.lookAt(Point3(lookat_pos[0], lookat_pos[1], lookat_pos[2]), Vec3(up[0], up[1], up[2]))
        self.cam.node().setLens(lens)  # use same len as sys
        self.lookat_pose = lookat_pos
        # set up cartoon effect
        if cartoon_render:
            self._separation = 1
            self.filter = flt.Filter(self.win, self.cam)
            self.filter.setCartoonInk(separation=self._separation)

        # camera in camera 2d
        self.cam2d = base.makeCamera2d(self.win, )
        self.cam2d.reparentTo(self.render2d)
        # attach GPTop to the render2d to make sure the DirectGui can be used
        self.aspect2d = self.render2d.attachNewNode(PGTop("aspect2d"))
        # self.aspect2d.setScale(1.0 / aspect_ratio, 1.0, 1.0)

        # setup mouse for the new window
        # name of mouse watcher is to adapt to the name in the input manager
        self.mouse_thrower = base.setupMouse(self.win, fMultiWin=True)
        self.mouseWatcher = self.mouse_thrower.getParent()
        self.mouseWatcherNode = self.mouseWatcher.node()
        self.aspect2d.node().setMouseWatcher(self.mouseWatcherNode)
        # self.mouseWatcherNode.addRegion(PGMouseWatcherBackground())

        # setup input manager
        self.inputmgr = im.InputManager(self, lookatpos=lookat_pos)

        # copy attributes and functions from base
        ## change the bound function to a function, and bind to `self` to become a unbound function
        self._interaction_update = functools.partial(base._interaction_update.__func__, self)
        self.p3dh = base.p3dh

        base.taskMgr.add(self._interaction_update, "interaction_extra_window", appendTask=True)

        if auto_cam_rotate:
            taskMgr.doMethodLater(.1, self._rotatecam_update, "rotate cam")

    @property
    def size(self):
        size = self.win.getProperties().size
        return np.array([size[0], size[1]])

    def getAspectRatio(self):
        return self._base.getAspectRatio(self.win)

    def set_win_props(self,
                      title: str,
                      size: tuple):
        """
        set properties of extra window
        :param title: the title of the window
        :param size: 1x2 tuple describe width and height
        :return:
        """
        win_props = WindowProperties()
        win_props.setSize(size[0], size[1])
        win_props.setTitle(title)
        self.win.requestProperties(win_props)

    def set_origin(self, origin: np.ndarray):
        """
        :param origin: 1x2 np array describe the left top corner of the window
        """
        win_props = WindowProperties()
        win_props.setOrigin(origin[0], origin[1])
        self.win.requestProperties(win_props)

    def clear_render(self):
        nodes = [_ for _ in self.render.children]
        for _ in nodes[3:]:
            _.removeNode()

    def _rotatecam_update(self, task):
        campos = self.cam.getPos()
        camangle = math.atan2(campos[1] - self.lookat_pose[1], campos[0] - self.lookat_pose[0])
        # print camangle
        if camangle < 0:
            camangle += math.pi * 2
        if camangle >= math.pi * 2:
            camangle = 0
        else:
            camangle += math.pi / 360
        camradius = math.sqrt((campos[0] - self.lookat_pose[0]) ** 2 + (campos[1] - self.lookat_pose[1]) ** 2)
        camx = camradius * math.cos(camangle)
        camy = camradius * math.sin(camangle)
        self.cam.setPos(self.lookat_pose[0] + camx, self.lookat_pose[1] + camy, campos[2])
        self.cam.lookAt(self.lookat_pose[0], self.lookat_pose[1], self.lookat_pose[2])
        return task.cont


import modeling.geometric_model as gm
import modeling.model_collection as mc
from panda3d.core import FrameBufferProperties, GraphicsPipe, GraphicsOutput, Shader
from typing import Literal


def generate_intrinsics(img_wh, fov):
    w, h = img_wh[0], img_wh[1]
    fov_x, fov_y = fov
    f_x = w / np.tan(fov_x / 2) / 2
    f_y = h / np.tan(fov_y / 2) / 2
    return np.array([[f_x, 0, w / 2],
                     [0, f_y, h / 2],
                     [0, 0, 1]])


class Camera(object):
    def __init__(self, base: wd.World,
                 world: wd.World or ExtraWindow = None,
                 img_sz: tuple or np.ndarray = (1024, 1024),
                 cam_pos: np.ndarray = np.array([2.0, 0, 2.0]),
                 lookat_pos: np.ndarray or list = np.array([0, 0, 0.25]),
                 up: np.ndarray or list = np.array([0, 0, 1]),
                 fov: int = 40,
                 lens_type: str = "perspective", ):
        if world is None:
            world = base

        self.base = base
        self.world = world
        self.img_sz = img_sz

        # graphics buffer
        self.color_buffer = self.create_color_graphics_buffer()

        # canvas to put depth image into it
        self.render = NodePath("depth_canvas")

        # set window background to white
        base.setBackgroundColor(r=1, g=1, b=1, win=self.color_buffer)

        # set len for the camera and set the camera for the new window
        lens = PerspectiveLens()
        lens.setFov(fov)
        lens.setNearFar(0.001, 5000.0)
        if lens_type == "orthographic":
            lens = OrthographicLens()
            lens.setFilmSize(1, 1)
        # make aspect ratio looks same as base window
        aspect_ratio = base.getAspectRatio(win=self.color_buffer)
        lens.setAspectRatio(aspect_ratio)

        # camera
        self.color_cam = base.makeCamera(win=self.color_buffer,
                                         scene=world.render,
                                         lens=lens)  # can also be found in base.camList

        self.cam_node = self.color_cam.node()
        # display region of the camera
        self.dr = self.cam_node.getDisplayRegion(0)
        # important !!!!!!!!!!!!!!!!!!!
        # get display region before applying filter, since a wired bug happens:
        # after applying filter, display region in camera node is different from the original one ???

        # set up cartoon effect
        # if cartoon_filter:
        self._separation = 1
        self.filter = flt.Filter(self.color_buffer, self.color_cam)
        self.filter.setCartoonInk(separation=self._separation)

        self.set_pos(cam_pos)
        self.look_at(lookat_pos, up)

        self.plot_node = None

    def set_pos(self, pos: np.ndarray):
        self.color_cam.setPos(Point3(pos[0], pos[1], pos[2]))

    def get_pos(self):
        pos = self.color_cam.getPos()
        return np.array([pos[0], pos[1], pos[2]])

    def look_at(self, lookat_pos: np.ndarray, up: np.ndarray):
        self.color_cam.lookAt(Point3(lookat_pos[0], lookat_pos[1], lookat_pos[2]), Vec3(up[0], up[1], up[2]))

    def create_color_graphics_buffer(self, ):
        buffer = base.win.makeTextureBuffer("Color Cam Texture Buffer", self.img_sz[0], self.img_sz[1])
        buffer.setSort(-3)
        return buffer

    def get_color_img(self, requested_format=None):
        """
        Returns the camera's image, which is of type uint8 and has values
        between 0 and 255.
        The 'requested_format' argument should specify in which order the
        components of the image must be. For example, valid format strings are
        "RGBA" and "BGRA". By default, Panda's internal format "BGRA" is used,
        in which case no data is copied over.
        """

        self.base.graphicsEngine.renderFrame()
        tex = self.dr.getScreenshot()
        if requested_format is None:
            data = tex.get_ram_image()
        else:
            data = tex.get_ram_image_as(requested_format)
        image = np.frombuffer(data, np.uint8)
        image.shape = (tex.getYSize(), tex.getXSize(), tex.getNumComponents())
        image = np.flipud(image)
        return image

    def show(self):
        if self.plot_node is not None and isinstance(self.plot_node, gm.GeometricModel):
            self.plot_node.remove()
        self.plot_node = gm.GeometricModel(basis.trimesh.primitives.Box(box_extents=np.array([.005, .005, .01])))
        self.plot_node.attach_to(base)


class DepthCamera(object):
    def __init__(self, base: wd.World,
                 world: wd.World or ExtraWindow = None,
                 img_sz: tuple or np.ndarray = (1024, 1024),
                 depth_fov=(np.pi / 6, np.pi / 6),
                 cam_pos: np.ndarray = np.array([2.0, 0, 2.0]),
                 lookat_pos: np.ndarray = np.array([0, 0, 0.25]),
                 up: np.ndarray = np.array([0, 0, 1]),
                 fov: int = 40,
                 lens_type: str = "perspective", ):
        if world is None:
            world = base

        self.base = base
        self.world = world
        self.img_sz = img_sz

        # graphics buffer
        self.color_buffer = self.create_color_graphics_buffer()
        self.depth_buffer, self.depth_tex = self.create_depth_graphics_buffer()

        # canvas to put depth image into it
        self.render = NodePath("depth_canvas")
        frag_txt = """
        #version 150 
        in float distanceToCamera;
        out vec4 fragColor; 
        void main() {
          fragColor = vec4(distanceToCamera, 0, 0, 1);
        }
        """
        vert_txt = """
        #version 150 
        // Uniform inputs
        uniform mat4 p3d_ProjectionMatrix;
        uniform mat4 p3d_ModelViewMatrix; 
        // Vertex inputs
        in vec4 p3d_Vertex; 
        // Vertex outputs
        out float distanceToCamera; 
        void main() {
          vec4 cs_position = p3d_ModelViewMatrix * p3d_Vertex;
          distanceToCamera = length(cs_position.xyz);
          gl_Position = p3d_ProjectionMatrix * cs_position;
        }
        """
        custom_shader = Shader.make(Shader.SL_GLSL, vertex=vert_txt, fragment=frag_txt)
        self.render.set_shader(custom_shader)

        # set window background to white
        base.setBackgroundColor(r=1, g=1, b=1, win=self.color_buffer)

        # set len for the camera and set the camera for the new window
        lens = PerspectiveLens()
        lens.setFov(fov)
        lens.setNearFar(0.001, 5000.0)
        if lens_type == "orthographic":
            lens = OrthographicLens()
            lens.setFilmSize(1, 1)
        # make aspect ratio looks same as base window
        aspect_ratio = base.getAspectRatio(win=self.color_buffer)
        lens.setAspectRatio(aspect_ratio)

        # camera
        self.color_cam = base.makeCamera(win=self.color_buffer,
                                         scene=world.render,
                                         lens=lens)  # can also be found in base.camList

        self.cam_node = self.color_cam.node()
        # display region of the camera
        self.dr = self.cam_node.getDisplayRegion(0)
        # important !!!!!!!!!!!!!!!!!!!
        # get display region before applying filter, since a wired bug happens:
        # after applying filter, display region in camera node is different from the original one ???

        # set up cartoon effect
        # if cartoon_filter:
        self._separation = 1
        self.filter = flt.Filter(self.color_buffer, self.color_cam)
        self.filter.setCartoonInk(separation=self._separation)

        # depth cam
        lens = PerspectiveLens()
        lens.setFov(*np.rad2deg(depth_fov))
        lens.setNearFar(0.001, 5000.0)
        lens.setFilmSize(*img_sz[::-1])

        # make aspect ratio looks same as base window
        aspect_ratio = base.getAspectRatio(win=self.depth_buffer)
        lens.setAspectRatio(aspect_ratio)
        self.depth_cam = base.makeCamera(self.depth_buffer,
                                         lens=lens,
                                         scene=self.render)
        self.depth_cam.reparentTo(self.render)
        # lens = self.depth_cam.node().getLens()
        # projection or intrinsic  matrix
        self.depth_cam_intrinsics = generate_intrinsics(img_wh=img_sz[::-1], fov=depth_fov)

        self.set_pos(cam_pos)
        self.look_at(lookat_pos, up)

    def set_pos(self, pos: np.ndarray):
        self.color_cam.setPos(Point3(pos[0], pos[1], pos[2]))
        self.depth_cam.setPos(Point3(pos[0], pos[1], pos[2]))

    def look_at(self, lookat_pos: np.ndarray, up: np.ndarray):
        self.color_cam.lookAt(Point3(lookat_pos[0], lookat_pos[1], lookat_pos[2]), Vec3(up[0], up[1], up[2]))
        self.depth_cam.lookAt(Point3(lookat_pos[0], lookat_pos[1], lookat_pos[2]), Vec3(up[0], up[1], up[2]))

    def create_color_graphics_buffer(self, ):
        buffer = base.win.makeTextureBuffer("Color Cam Texture Buffer", self.img_sz[0], self.img_sz[1])
        buffer.setSort(-3)
        return buffer

    def create_depth_graphics_buffer(self, ):
        base = self.base
        window_props = WindowProperties(size=(self.img_sz[0], self.img_sz[1]))
        frame_buffer_props = FrameBufferProperties()
        frame_buffer_props.set_float_color(True)
        frame_buffer_props.set_rgba_bits(32, 0, 0, 0)
        buffer = base.graphicsEngine.make_output(base.pipe,
                                                 f'Buffer',
                                                 -2,
                                                 frame_buffer_props,
                                                 window_props,
                                                 GraphicsPipe.BFRefuseWindow,  # don't open a window
                                                 base.win.getGsg(),
                                                 base.win
                                                 )
        texture = Texture()
        buffer.add_render_texture(texture, GraphicsOutput.RTMCopyRam)
        buffer.set_clear_color_active(True)
        buffer.set_clear_color((10, 0, 0, 0))
        return buffer, texture

    def attach_to_depth_canvas(self, md: gm.StaticGeometricModel or mc.ModelCollection):
        if isinstance(md, gm.StaticGeometricModel):
            md.objpdnp.reparentTo(self.render)
        elif isinstance(md, mc.ModelCollection):
            for m in md.cm_list:
                m.objpdnp.reparentTo(self.render)
        else:
            raise Exception("not support format")

    def get_color_img(self, requested_format=None):
        """
        Returns the camera's image, which is of type uint8 and has values
        between 0 and 255.
        The 'requested_format' argument should specify in which order the
        components of the image must be. For example, valid format strings are
        "RGBA" and "BGRA". By default, Panda's internal format "BGRA" is used,
        in which case no data is copied over.
        """

        self.base.graphicsEngine.renderFrame()
        tex = self.dr.getScreenshot()
        if requested_format is None:
            data = tex.get_ram_image()
        else:
            data = tex.get_ram_image_as(requested_format)
        image = np.frombuffer(data, np.uint8)
        image.shape = (tex.getYSize(), tex.getXSize(), tex.getNumComponents())
        image = np.flipud(image)
        return image

    def get_depth_img(self):
        self.base.graphicsEngine.renderFrame()
        data = self.depth_tex.get_ram_image()
        depth_image = np.frombuffer(data, np.float32)
        depth_image.shape = (self.depth_tex.getYSize(), self.depth_tex.getXSize(), self.depth_tex.getNumComponents())
        depth_image = np.flipud(depth_image)
        return depth_image

    def get_point_cloud(self):
        return self.depth_2_point_cloud(self.get_depth_img())

    def depth_2_point_cloud(self, depth_img):
        u, v = np.arange(depth_img.shape[1]), np.arange(depth_img.shape[0])
        u, v = np.meshgrid(u, v)
        u = u.astype(float)
        v = v.astype(float)
        K = self.depth_cam_intrinsics
        Z = depth_img.astype(float).squeeze(2)
        Y = (u - K[0, 2]) * Z / K[0, 0]
        X = (v - K[1, 2]) * Z / K[1, 1]
        X, Y, Z = np.ravel(X), np.ravel(Y), np.ravel(Z)
        valid = (Z > 0) & (Z < 3)
        X = X[valid]
        Y = Y[valid]
        Z = Z[valid]
        XYZ = np.vstack((X, Y, Z)).T
        mat = pdmat4_to_npmat4(self.depth_cam.get_mat())
        mat[:, [1, 2]] = mat[:, [2, 1]]
        mat[:, [0, 1]] = mat[:, [1, 0]]
        mat[:, 0] = -mat[:, 0]
        # mat[:3, 3] = np.array([-mat[:, 3][2],mat[:, 3][0],mat[:, 3][1]])
        points = homomat_transform_points(mat, XYZ)
        a = pdmat4_to_npmat4(self.depth_cam.get_mat())
        # gm.gen_frame(a[:3, 3], a[:3, :3]).attach_to(base)
        # gm.gen_frame(mat[:3, 3] - np.array([0, 0, .1]), mat[:3, :3]).attach_to(base)
        # points = XYZ
        return points


if __name__ == "__main__":
    import modeling.geometric_model as gm
    from huri.core.common_import import ym


    def show_rgbd_image(depth_image, depth_offset=0.0, depth_scale=1.0):
        if depth_image.dtype != np.uint8:
            if depth_scale is None:
                depth_scale = depth_image.max() - depth_image.min()
            if depth_offset is None:
                depth_offset = depth_image.min()
            depth_image = np.clip((depth_image - depth_offset) / depth_scale, 0.0, 1.0)
            depth_image = (255.0 * depth_image).astype(np.uint8)
        depth_image = np.tile(depth_image, (1, 1, 3))
        return depth_image


    base = wd.World(cam_pos=[0, 0, .3], lookat_pos=[0, 0, .0])
    # ew.render.set_shader(custom_shader)
    # ew = ExtraWindow(base, cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    # ew2 = ExtraWindow(base, cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    # gm.gen_frame(length=.2).attach_to(base)
    b = gm.GeometricModel("bunnysim.stl", btransparency=False)
    bb = b.copy()
    bb.set_rgba([0, 0, 0, 1])
    bb.attach_to(base)

    cam = Camera(base, img_sz=(3000, 3000),
                 cam_pos=[0, 0, .3], lookat_pos=[0, 0, 0])
    img = cam.get_color_img()
    cv2.imshow("asda", img)
    cv2.waitKey(0)
    exit(0)

    # # ImgOnscreen()
    # on_screen_img = ImgOnscreen(ew.size, parent_np=ew)
    # img = cv2.imread("/drivers/rpc/extcam/1.bmp")
    # on_screen_img.update_img(img)
    # base.run()
    a = ym.Yumi()
    a.fk("rgt_arm", a.rand_conf("rgt_arm"))
    # a.gen_meshmodel().attach_to(base)

    # gm.gen_frame().attach_to(base)
    # gm.gen_frame().objpdnp.reparentTo(ew.render)

    cm = DepthCamera(base, img_sz=(1000, 1000),
                     depth_fov=(np.pi / 3, np.pi / 3),
                     cam_pos=[0, 0, .3], lookat_pos=[0, 0, 0])

    # cm.attach_to_depth_canvas(a.gen_meshmodel())
    # cm.attach_to_depth_canvas(gm.gen_sphere())
    cm.attach_to_depth_canvas(b)

    radius = .2
    step = 1
    import matplotlib.pyplot as plt

    img = cm.get_depth_img().copy()
    img[img > 5] = 0
    img = img.squeeze()
    color_img = cm.get_color_img()
    cm.get_point_cloud()
    plt.imshow(X=img)
    plt.colorbar()
    plt.show()

    points = cm.get_point_cloud()
    gm.gen_pointcloud(points, pntsize=5).attach_to(base)
    base.run()

    for t in range(1800):
        angleDegrees = t * step
        angleRadians = angleDegrees * (np.pi / 180.0)
        cm.set_pos(np.array([2 + radius * np.sin(angleRadians), 0 - radius * np.cos(angleRadians), 1.5]))
        cm1.set_pos(np.array([2 + radius * np.sin(angleRadians), 0 - radius * np.cos(angleRadians), 1.5]))
        # cm.cam.setHpr(angleDegrees, 0, 0)

        cv2.imshow("img from panda3d", show_rgbd_image(depth_image=img))
        cv2.imshow("color img from panda3d", color_img)
        cv2.waitKey(1)
    exit(0)
    # base.run()
