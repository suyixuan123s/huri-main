import copy

import numpy as np


def show_animation(sim_rbt, motion_batch, base):
    is_motion_exec_finished = [False]

    def update(robot_s,
               motion_batch,
               is_motion_exec_finished,
               task):
        if base.inputmgr.keymap["space"]:
            base.inputmgr.keymap["space"] = False
            try:
                motion_name, motion_element = next(motion_batch)
            except StopIteration:
                is_motion_exec_finished[0] = True
                return task.again
            except Exception as e:
                raise Exception(e)
        else:
            motion_element = motion_batch.current
        try:
            objcm, obj_pose, pose, jawwdith, hand_name, obs_list = next(motion_element)
        except StopIteration:
            return task.again
        except Exception as e:
            return Exception(e)
        # print(np.rad2deg(pose))
        robot_s.fk(hand_name, pose)
        robot_s.jaw_to(hand_name, jawwdith)
        robot_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=True)
        robot_meshmodel.attach_to(base)
        objb_copied = objcm.copy()
        objb_copied.set_homomat(obj_pose)
        objb_copied.attach_to(base)
        obs_list_copied = copy.deepcopy(obs_list)
        obs_list_copied.attach_to(base)
        motion_element.reset_robot_gm(robot_meshmodel)
        motion_element.reset_obj_gm(objb_copied)
        motion_element.reset_obs_list_gm(obs_list_copied)
        return task.again

    taskMgr.doMethodLater(0.1, update, "update",
                          extraArgs=[sim_rbt,
                                     motion_batch,
                                     is_motion_exec_finished],
                          appendTask=True)