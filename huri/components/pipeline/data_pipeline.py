from typing import Union, Optional
from tkinter import Tk
import logging

import numpy as np

from modeling.geometric_model import GeometricModel
from modeling.collision_model import CollisionModel
from huri.components.gui.tk_gui.base import GuiFrame
from visualization.panda.world import World
from huri.definitions.utils_structure import MotionBatch, MotionElement
from huri.components.yumi_control.yumi_con import YumiController

import motion.trajectory.piecewisepoly_fullcode as pwp


class Cell:
    def __init__(self,
                 name: str,
                 node: Union[GeometricModel, CollisionModel]):
        self.name = name
        self.render_node = node

    def detach(self):
        self.render_node.detach()

    def remove(self):
        self.render_node.remove()

    def attach_to(self, scene):
        self.render_node.attach_to(scene)


class RenderController:
    def __init__(self, root: Tk, base: World):
        self.cells = []
        self.render_controlpanel = GuiFrame(root)
        self.render_controlpanel.set_title("The visualization controller")
        self.base = base

    def attach(self, node: Union[GeometricModel, CollisionModel], name="node"):
        node.attach_to(self.base)
        self.base._internal_update_obj_list.append(node)
        self.render_controlpanel.add_checkbutton(text=name,
                                                 callback=lambda check_status: self.base.attach_internal_update_obj(
                                                     node)
                                                 if check_status else self.base.detach_internal_update_obj(node),
                                                 pos=(int(len(self.cells) / 3),
                                                      len(self.cells) % 3))
        self.cells.append(Cell(name=name, node=node))


class MotionController:
    def __init__(self, root: Tk, yumi_con: YumiController = None):
        self.panel = GuiFrame(root)
        self.panel.set_title("The motion controller")
        self._options = []
        self._ref = []
        self.panel.add_button(text="execute next motion", command=self.execute_next_motion, pos=(1, 0))
        self._listbox = self.panel.add_listbox(options=self._options)
        self._counter = 0
        self._finish = False
        self.yumi_con = yumi_con

    def add_batch(self, motion_batch: MotionBatch, name="motion batch"):
        self._options += [f"{name}-{i}" for i in range(len(motion_batch))]
        self._ref += [motion_batch[i] for i in range(len(motion_batch))]
        self._listbox.updateitems(self._options)

    def execute_next_motion(self):
        if self._finish:
            print("Motion has already finished")
            return
        if self._counter >= len(self._options):
            self._listbox._clear_selected()
            self._finish = True
            return
        self._listbox.set_select(index=self._counter)
        if self.yumi_con is not None:
            self.execute_motion(self._ref[self._counter])
        self._counter = self._counter + 1

    def execute_motion(self, motion_element: MotionElement):
        if self.yumi_con is None:
            raise Exception("no real robot")
        yumi_con = self.yumi_con
        hndname = motion_element.hnd_name
        conf_list = motion_element.conf_list
        jawwidth_list = motion_element.jawwidth_list
        if np.linalg.norm(yumi_con.get_jnt_values(component_name=hndname) - conf_list[0]) > np.pi / 20:
            yumi_con.move_jnts(component_name=hndname, jnt_vals=conf_list[0])
        if jawwidth_list[0] >= 0.05:
            yumi_con.open_gripper(component_name=hndname)
        if jawwidth_list[0] < 0.05:
            yumi_con.close_gripper(component_name=hndname)
        yumi_con.move_jntspace_path(path=conf_list, component_name=hndname)


trajopt_gen = pwp.PiecewisePoly(method="quintic")


def motion_executer(yumi_s,
                    yumi_con: Optional[YumiController],
                    motion_batch: MotionBatch,
                    open_gripper_len=0.05,
                    speed_n=100,
                    logger=logging.getLogger(__name__)):
    logger_name = "motion_executer"
    if yumi_con is None:
        print("-------------- no real robot -----------")
        return None
    is_closed = [False]
    is_refined_motion = [False, None]

    def open_gripper():
        if is_closed[0]:
            yumi_con.set_gripper_width(component_name=hndname, width=open_gripper_len)
            is_closed[0] = False

    def close_gripper():
        if not is_closed[0]:
            yumi_con.close_gripper(component_name=hndname)
            is_closed[0] = True

    for motion_name, motion_element in motion_batch:
        hndname = motion_element.hnd_name
        conf_list = motion_element.conf_list
        jawwidth_list = motion_element.jawwidth_list
        if is_refined_motion[0]:
            is_refined_motion[0] = False
            conf_list = [is_refined_motion[1]] + conf_list
            is_refined_motion[1] = None
        if np.linalg.norm(yumi_con.get_jnt_values(component_name=hndname) - conf_list[0]) > np.pi / 20:
            yumi_con.move_jnts(component_name=hndname, jnt_vals=conf_list[0], speed_n=speed_n)
        if jawwidth_list[0] >= 0.05:
            open_gripper()
        if jawwidth_list[0] < 0.05:
            close_gripper()
        if motion_name == "place_approach":
            speed_n = 100
        else:
            # if motion_name == "start2pickapproach" or motion_name == "placedepart2goal" or motion_name == "placedepart":
            speed_n = -1
        if motion_name == "pick_approach" or motion_name == "place_approach":
            # if motion_name == "place_approach":
            yumi_con.move_jntspace_path(path=conf_list[:-1], component_name=hndname, speed_n=speed_n)
            yumi_con.contactL(hndname, jnt_vals=conf_list[-1])
            open_gripper()
            is_refined_motion[0] = True
            is_refined_motion[1] = yumi_con.get_jnt_values(hndname)
            print(f"the rgt arm joints are: {repr(yumi_con.get_jnt_values(hndname))}")
        else:
            # interpolated_confs, interpolated_spds, interpolated_accs, interpolated_x = \
            #     trajopt_gen.interpolate(conf_list,control_frequency=.8)
            # print(interpolated_confs)
            yumi_con.move_jntspace_path(path=conf_list, component_name=hndname, speed_n=speed_n)
            print(motion_name)
            # if motion_name == "pick_approach":
            #     yumi_s.fk(hndname, yumi_con.get_jnt_values(hndname))
            #     yumi_s.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
            #     pose_rgt_raw = yumi_con.rgt_arm_hnd.get_pose()
            #     print("real tcp pose", pose_rgt_raw.translation, pose_rgt_raw.rotation)
            #     print(yumi_s.get_gl_tcp(hndname))
            #     base.run()


def motion_executer2(yumi_con: Optional[YumiController], motion_element: MotionElement, open_gripper_len=0.05,
                     speed_n=100):
    print(yumi_con)
    if yumi_con is None:
        print("-------------- no real robot -----------")
        return None
    hndname = motion_element.hnd_name
    conf_list = motion_element.conf_list
    jawwidth_list = motion_element.jawwidth_list
    if np.linalg.norm(yumi_con.get_jnt_values(component_name=hndname) - conf_list[0]) > np.pi / 20:
        yumi_con.move_jnts(component_name=hndname, jnt_vals=conf_list[0], speed_n=speed_n)
    if jawwidth_list[0] >= 0.05:
        yumi_con.set_gripper_width(component_name=hndname, width=open_gripper_len)
    if jawwidth_list[0] < 0.05:
        yumi_con.close_gripper(component_name=hndname)
    yumi_con.move_jntspace_path(path=conf_list, component_name=hndname, speed_n=speed_n)


if __name__ == "__main__":
    from huri.definitions.utils_structure import MotionBatch, MotionElement

    motionbatch = MotionBatch()
    motioncell1 = MotionElement(None, None, None, None)
    motioncell2 = MotionElement(None, None, None, None)
    motioncell3 = MotionElement(None, None, None, None)
    motionbatch.append(motioncell1)
    motionbatch.append(motioncell2)
    motionbatch.append(motioncell3)
    root = Tk()
    child = MotionController(root)
    child.add_batch(motionbatch)
    root.mainloop()
