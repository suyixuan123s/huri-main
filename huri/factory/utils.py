""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230909osaka

"""
import numpy as np
from tqdm import tqdm

import modeling.geometric_model as gm
import basis.robot_math as rm
from huri.components.yumi_control.yumi_con import YumiController
from huri.math.units import Mm
from robot_sim.robots.yumi.yumi import Yumi


def examine_rbt_con_fk(rbt_con: YumiController):
    """
    Examine if the fk of the robot controller is consistent with the real robot. If not, raise an error.
    :param rbt_con: the robot controller instance
    :return:
    """
    pos_rgt_fk, rot_rgt_fk = rbt_con.fk('rgt_arm', rbt_con.get_jnt_values('rgt_arm'))
    pos_lft_fk, rot_lft_fk = rbt_con.fk('lft_arm', rbt_con.get_jnt_values('lft_arm'))
    pos_rgt_real, rot_rgt_real = rbt_con.get_pose('rgt_arm')
    pos_lft_real, rot_lft_real = rbt_con.get_pose('lft_arm')
    if np.linalg.norm(pos_rgt_fk - pos_rgt_real) > 1e-3 or np.linalg.norm(pos_lft_fk - pos_lft_real) > 1e-3:
        raise ValueError(
            "Please restart the Yumi controller. fk's position of the robot controller is not consistent with the real robot.")
    # check the rotation error between the fk of the robot controller and the real robot
    rot_rgt_error = np.linalg.norm(rot_rgt_fk - rot_rgt_real)
    rot_lft_error = np.linalg.norm(rot_lft_fk - rot_lft_real)
    if rot_rgt_error > 1e-1 or rot_lft_error > 1e-3:
        raise ValueError(
            "Please restart the Yumi controller. fk's rotation of the robot controller is not consistent with the real robot.")


def generate_slot_centers(rack_shape: tuple or list or np.ndarray = (5, 10),
                          center_dist_x: float or int = Mm(20),
                          center_dist_y: float or int = Mm(20),
                          offset_x: float or int = 0,
                          offset_y: float or int = 0,
                          offset_z: float or int = 0, ) -> np.ndarray:
    """
    Generate the slot centers of a 2D rack
    :param rack_shape:  the shape of the rack
    :param center_dist_x:  the distance between two adjacent slots in x direction
    :param center_dist_y:  the distance between two adjacent slots in y direction
    :param offset_x:  the offset of the rack in x direction
    :param offset_y:  the offset of the rack in y direction
    :return:  the slot centers of the rack
    """
    if isinstance(rack_shape, list) or isinstance(rack_shape, tuple):
        if len(rack_shape) != 2:
            raise ValueError('shape must be a list or tuple with length of 2')
    elif isinstance(rack_shape, np.ndarray):
        if rack_shape.ndim != 2:
            raise ValueError('shape must be a numpy array with ndim of 2')
    assert isinstance(center_dist_x, float) or isinstance(center_dist_x,
                                                          int), 'center_dist_x must be type of float or int'
    assert isinstance(center_dist_y, float) or isinstance(center_dist_y,
                                                          int), 'center_dist_y must be type of float or int'
    # assert the offset_x must be a numeric value
    assert isinstance(offset_x, float) or isinstance(offset_x, int), 'offset_x must be type of float or int'
    assert isinstance(offset_y, float) or isinstance(offset_y, int), 'offset_y must be type of float or int'
    assert isinstance(offset_z, float) or isinstance(offset_z, int), 'offset_z must be type of float or int'

    slot_centers = []

    for i in range(rack_shape[0]):
        for j in range(rack_shape[1]):
            slot_centers.append(np.array([i * center_dist_x + offset_x, j * center_dist_y + offset_y, offset_z]))
    # make slot_centers into a numpy array with rack shape
    slot_centers = np.array(slot_centers).reshape(rack_shape[0], rack_shape[1], 3)
    return slot_centers


def yumi_solve_ik(rbt_sim: Yumi,
                  rbt_con: YumiController,
                  pos: np.ndarray,
                  rot: np.ndarray,
                  component_name: str = 'rgt_arm',
                  seed_jnt_values: np.ndarray = None,
                  toggle_collision_detection: bool = True,
                  toggle_flip: bool = False,
                  toggle_visual: bool = False) -> np.ndarray or None:
    """
    Solve the ik of the end effector of the robot given the position and rotation
    :param rbt_sim: the simulated robot instance
    :param rbt_con: the real robot controller instance
    :param pos: the position of the end effector
    :param rot: the rotation of the end effector
    :param component_name: the name of the component. Must be ['rgt_arm', 'lft_arm']
    :param seed_jnt_values: the seed joint values of the robot
    :param toggle_collision_detection: toggle the collision detection
    :param toggle_flip: toggle the flip of the rotation matrix
    :param toggle_visual: toggle the visualization of the ik solution
    :return: the joint values of the robot
    """
    assert isinstance(rbt_sim, Yumi), 'rbt_sim must be a Yumi instance'
    assert isinstance(rbt_con, YumiController), 'rbt_real must be a YumiController instance'
    assert isinstance(pos, np.ndarray), 'pos must be a numpy array'
    assert isinstance(rot, np.ndarray), 'rot must be a numpy array'
    assert seed_jnt_values is None or isinstance(seed_jnt_values, np.ndarray), 'seed_jnt_values must be a numpy array'
    assert toggle_visual is True or toggle_visual is False, 'toggle_visual must be either True or False'
    assert toggle_flip is True or toggle_flip is False, 'toggle_flip must be either True or False'
    assert toggle_collision_detection is True or toggle_collision_detection is False, \
        'toggle_collision_detection must be either True or False'
    if pos.ndim != 1 or pos.shape[0] != 3:
        raise ValueError('pos must be a numpy array with shape of (3,)')
    if rot.ndim != 2 or rot.shape != (3, 3):
        raise ValueError('rot must be a numpy array with shape of (3, 3)')
    if component_name not in ['rgt_arm', 'lft_arm']:
        raise ValueError('component_name must be either "rgt_arm" or "lft_arm"')
    if seed_jnt_values is not None:
        if seed_jnt_values.ndim != 1 or seed_jnt_values.shape[0] != 7:
            raise ValueError('seed_jnt_values must be a numpy array with shape of (7,)')
    # solve the ik of the end effector
    jnt_val1 = rbt_sim.ik(component_name, pos, rot, seed_jnt_values=seed_jnt_values)
    # get robot configuration
    _, _, conf, ext_axis = rbt_con.fk(component_name, jnt_val1, return_conf=True)
    # solve accurate ik using ik function on the real robot
    accurate_jnt_val = rbt_con.ik(component_name, pos, rot, conf, ext_axis=ext_axis)
    if toggle_flip and (seed_jnt_values is not None or accurate_jnt_val is None):
        # get robot configuration
        jnt_val2 = rbt_sim.ik(component_name, pos, np.dot(np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]), rot),
                              seed_jnt_values=seed_jnt_values)
        _, _, conf, ext_axis = rbt_con.fk(component_name, jnt_val2, return_conf=True)
        # solve accurate ik using ik function on the real robot
        accurate_jnt_val2 = rbt_con.ik(component_name, pos, rot, conf, ext_axis=ext_axis)
        # compare the distance between the seed_jnt_values and the ik solution
        if accurate_jnt_val2 is not None:
            if accurate_jnt_val is None:
                accurate_jnt_val = accurate_jnt_val2
            elif seed_jnt_values is not None:
                if np.linalg.norm(accurate_jnt_val2 - seed_jnt_values) < np.linalg.norm(
                        accurate_jnt_val - seed_jnt_values):
                    accurate_jnt_val = accurate_jnt_val2  # choose the ik solution with smaller distance
    if accurate_jnt_val is None:  # if the ik solution is not found, return None
        print("Accurate not found")
        return None
    if toggle_visual:
        rbt_sim.fk(component_name, accurate_jnt_val)  # update the pose of the simulated robot
        rbt_sim_mdl = rbt_sim.gen_meshmodel().cm_list
        for cm in rbt_sim_mdl:
            cm.set_rgba([*cm.get_rgba()[:3], .5])
            cm.attach_to(base)
    if toggle_collision_detection:
        old_jnt_val = rbt_sim.get_jnt_values(component_name)  # store the old joint values
        rbt_sim.fk(component_name, accurate_jnt_val)  # update the pose of the simulated robot
        is_collided = rbt_sim.is_collided()  # check if the robot is collided
        rbt_sim.fk(component_name, old_jnt_val)  # restore the pose of the simulated robot
        if is_collided:  # if the robot is collided, return None
            print('collision detected')

            return None
    return accurate_jnt_val


def yumi_gen_motion_slot_centers(rbt_sim: Yumi,
                                 rbt_con: YumiController,
                                 slot_centers: np.ndarray,
                                 init_joint_val: np.ndarray,
                                 component_name: str = 'rgt_arm',
                                 toggle_visual: bool = False,
                                 approach_dist=.1,
                                 departure_dist=.1) -> (np.ndarray, np.ndarray):
    """
    Solve the ik of the end effector of the robot at the slot centers given the initial joint values
    :param rbt_sim: the simulated robot instance
    :param rbt_con: the real robot controller instance
    :param slot_centers: 1x3 vector describe the slot centers of the rack
    :param init_joint_val: the initial joint values of the robot at the slot center [0, 0]  (the first slot)
    :param component_name: the name of the component. Must be ['rgt_arm', 'lft_arm']
    :param toggle_visual: toggle the visualization of the ik solution
    :return: the joint values of the robot given the 2d slot centers
    """
    assert isinstance(rbt_sim, Yumi), 'rbt_sim must be a Yumi instance'
    assert isinstance(rbt_con, YumiController), 'rbt_real must be a YumiController instance'
    assert isinstance(slot_centers, np.ndarray), 'slot_centers must be a numpy array'
    assert isinstance(init_joint_val, np.ndarray), 'init_joint_val must be a numpy array'
    assert isinstance(toggle_visual, bool), 'toggle_visual must be either True or False'
    if slot_centers.ndim != 3 or slot_centers.shape[2] != 3:
        raise ValueError('slot_centers must be a numpy array with shape of (n, m, 3)')
    if init_joint_val.ndim != 1 or init_joint_val.shape[0] != 7:
        raise ValueError('init_joint_val must be a numpy array with shape of (7,)')
    if component_name not in ['rgt_arm', 'lft_arm']:
        raise ValueError('component_name must be either "rgt_arm" or "lft_arm"')
    # get the pose of the end effector when the robot is at the init_joint_val
    pos, rot = rbt_con.fk(component_name, init_joint_val)
    pos_first = pos.copy()
    approach_ik_list = []
    departure_ik_list = []
    ik_list = []
    # progress bar for processing slots
    total_iterations = slot_centers.shape[0] * slot_centers.shape[1] - 1
    progress_bar = tqdm(total=total_iterations, desc="Processing Slots")
    last_jnt_val = init_joint_val
    for i in range(slot_centers.shape[0]):
        for j in range(slot_centers.shape[1]):
            pos = pos_first.copy()
            pos += slot_centers[i, j]
            accurate_jnt_val = yumi_solve_ik(rbt_sim, rbt_con,
                                             pos=pos,
                                             rot=rot,
                                             component_name=component_name,  # solve the ik of the end effector
                                             toggle_collision_detection=True,
                                             # seed_jnt_values=solve_iks[-1],
                                             seed_jnt_values=init_joint_val,
                                             toggle_visual=toggle_visual,
                                             toggle_flip=True)
            if accurate_jnt_val is None:
                raise ValueError(f'ik solution at slot [{i}, {j}] not found')
            # departure ik
            pos_departure = pos.copy()
            pos_departure[2] += departure_dist
            accurate_jnt_val_departure_list = [accurate_jnt_val]
            pos_list, rotmat_list = rm.interplate_pos_rotmat(pos,
                                                             rot,
                                                             pos_departure,
                                                             rot,
                                                             granularity=0.008)
            accurate_jnt_val_departure_ = accurate_jnt_val
            for pos_, rot_ in zip(pos_list[1:], rotmat_list[1:]):
                accurate_jnt_val_departure_ = yumi_solve_ik(rbt_sim, rbt_con,
                                                            pos=pos_,
                                                            rot=rot_,
                                                            component_name=component_name,
                                                            toggle_collision_detection=True,
                                                            seed_jnt_values=accurate_jnt_val_departure_,
                                                            toggle_visual=False)
                if accurate_jnt_val_departure_ is None:
                    raise ValueError(f'approaching ik solution at slot [{i}, {j}] not found')
                accurate_jnt_val_departure_list.append(accurate_jnt_val_departure_)
            # approach ik
            pos_approach = pos.copy()
            pos_approach += approach_dist * rm.unit_vector(np.array([-1, -1, 1]))
            accurate_jnt_val_approach_list = [accurate_jnt_val]
            pos_list, rotmat_list = rm.interplate_pos_rotmat(pos,
                                                             rot,
                                                             pos_approach,
                                                             rot,
                                                             granularity=0.008)
            accurate_jnt_val_approach_ = accurate_jnt_val
            for pos_, rot_ in zip(pos_list[1:], rotmat_list[1:]):
                accurate_jnt_val_approach_ = yumi_solve_ik(rbt_sim, rbt_con,
                                                           pos=pos_,
                                                           rot=rot_,
                                                           component_name=component_name,
                                                           toggle_collision_detection=True,
                                                           seed_jnt_values=accurate_jnt_val_approach_,
                                                           toggle_visual=False)
                if accurate_jnt_val_approach_ is None:
                    raise ValueError(f'approaching ik solution at slot [{i}, {j}] not found')
                accurate_jnt_val_approach_list.append(accurate_jnt_val_approach_)
            last_jnt_val = accurate_jnt_val_departure_list[-1]
            ik_list.append(accurate_jnt_val)
            departure_ik_list.append(accurate_jnt_val_departure_list)
            approach_ik_list.append(accurate_jnt_val_approach_list[::-1])
            progress_bar.update(1)  # Update the progress bar for each iteration
            # reshape solve_iks into a numpy array with shape of (n, m, 7)
    solve_iks = np.array(ik_list).reshape(slot_centers.shape[0], slot_centers.shape[1], 7)
    solve_iks_approach = np.array(approach_ik_list).reshape(slot_centers.shape[0], slot_centers.shape[1], -1, 7)
    solve_iks_departure = np.array(departure_ik_list).reshape(slot_centers.shape[0], slot_centers.shape[1], -1, 7)
    return solve_iks, solve_iks_approach, solve_iks_departure


if __name__ == '__main__':
    slot_centers = generate_slot_centers(rack_shape=(5, 10), center_dist_x=Mm(20), center_dist_y=Mm(20), offset_x=0,
                                         offset_y=0)
    print(slot_centers.shape)
    for i in range(5):
        for j in range(10):
            print('slot center [', i, ',', j, '] is', slot_centers[i, j])
