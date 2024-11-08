import copy
import time
import math
import numpy as np
import motion.trajectory.piecewisepoly_toppra as trajp
import drivers.orin_bcap.bcapclient as bcapclient
import numpy.typing as npt
from typing import List
import basis.robot_math as rm


class CobottaX(object):

    def __init__(self, host='192.168.0.1', port=5007, timeout=2000):
        """
        :param host:
        :param port:
        :param timeout:

        author: weiwei
        date: 20210507
        """
        self.bcc = bcapclient.BCAPClient(host, port, timeout)
        self.bcc.service_start("")
        # Connect to RC8 (RC8(VRC)provider)
        self.hctrl = self.bcc.controller_connect("", "CaoProv.DENSO.VRC", ("localhost"), (""))
        self.clear_error()
        # get robot_s object hanlde
        self.hrbt = self.bcc.controller_getrobot(self.hctrl, "Arm", "")
        # print(self.bcc.robot_getvariablenames(self.hrbt))
        # self.bcc.controller_getextension(self.hctrl, "Hand", "")
        # take arm
        self.hhnd = self.bcc.robot_execute(self.hrbt, "TakeArm", [0, 0])
        # motor on
        self.bcc.robot_execute(self.hrbt, "Motor", [1, 0])
        # set ExtSpeed = [speed, acc, dec]
        self.bcc.robot_execute(self.hrbt, "ExtSpeed", [100, 100, 100])
        self.traj_gen = trajp.PiecewisePolyTOPPRA()
        # self.bcc.robot_execute(self.hrbt,"ErAlw",[True,1,0.01])
        # self.bcc.robot_execute(self.hrbt,"ErAlw",[True,2,0.01])
        # self.bcc.robot_execute(self.hrbt,"ErAlw",[True,3,0.01])
        # self.bcc.robot_execute(self.hrbt,"ErAlw",[True,4,0.01])
        # self.bcc.robot_execute(self.hrbt,"ErAlw",[True,5,0.01])
        # self.bcc.robot_execute(self.hrbt,"ErAlw",[True,6,0.01])

    @staticmethod
    def wrshomomat2cobbotapos(wrshomomat, k=-1):
        pos = wrshomomat[:3, 3]
        rpy = rm.rotmat_to_euler(wrshomomat[:3, :3])
        return np.hstack([pos, rpy, k])

    @staticmethod
    def cobottapos2wrspos(cobbota_pos):
        pos = cobbota_pos[:3]
        rpy = cobbota_pos[3:6]
        return pos, rm.rotmat_from_euler(*rpy)

    def __del__(self):
        self.clear_error()
        self.bcc.controller_getrobot(self.hrbt, "Motor", [0, 0])
        self.bcc.robot_execute(self.hrbt, "GiveArm", None)
        self.bcc.robot_release(self.hrbt)
        self.bcc.controller_disconnect(self.hctrl)
        self.bcc.service_stop()

    def clear_error(self):
        self.bcc.controller_execute(self.hctrl, "ClearError", None)

    def disconnect(self):
        self.bcc.controller_disconnect(self.hctrl)

    def moveto_named_pose(self, name):
        self.bcc.robot_move(self.hrbt, 1, name, "")

    def move_jnts_motion(self, path: List[npt.NDArray[float]], toggle_debug: bool = False):
        """
        :param path:
        :return:
        author: weiwei
        date: 20210507
        """
        time.sleep(0.1)
        self.hhnd = self.bcc.robot_execute(self.hrbt, "TakeArm", [0, 0])  # 20220317, needs further check, speedmode?
        time.sleep(0.2)
        new_path = []
        for i, pose in enumerate(path):
            if i < len(path) - 1 and not np.allclose(pose, path[i + 1]):
                new_path.append(pose)
        new_path.append(path[-1])
        path = new_path
        # max_vels = [math.pi * .6, math.pi * .4, math.pi, math.pi, math.pi, math.pi * 1.5]
        max_vels = [math.pi / 6, math.pi / 6, math.pi / 6, math.pi / 3, math.pi / 3, math.pi / 2]
        interpolated_confs = \
            self.traj_gen.interpolate_by_max_spdacc(path,
                                                    control_frequency=.008,
                                                    max_vels=max_vels,
                                                    toggle_debug=toggle_debug)
        # print(f"toppra{interpolated_confs[:,2].max()}")
        # Slave move: Change mode
        while True:
            try:
                # time.sleep(.2)
                self.bcc.robot_execute(self.hrbt, "slvChangeMode", 0x202)
                time.sleep(.5)
                # print("sleep done")
                print(self.get_jnt_values())
                print(interpolated_confs[0].tolist())
                self.bcc.robot_execute(self.hrbt, "slvMove", np.degrees(interpolated_confs[0]).tolist() + [0, 0])
                # time.sleep(.2)
                # print("try exec done")
                break
            except:
                print("exception, continue")
                self.clear_error()
                time.sleep(0.2)
                continue
        try:
            for jnt_values in interpolated_confs:
                jnt_values_degree = np.degrees(jnt_values)
                self.bcc.robot_execute(self.hrbt, "slvMove", jnt_values_degree.tolist() + [0, 0])
            # print("trajectory done")
        except:
            # print("trajectory exception, continue")
            # self.clear_error()
            time.sleep(0.2)
            return False
        self.bcc.robot_execute(self.hrbt, "slvChangeMode", 0x000)
        self.bcc.robot_execute(self.hrbt, "GiveArm", None)
        time.sleep(0.1)
        return True

    def get_jnt_values(self):
        pose = self.bcc.robot_execute(self.hrbt, "CurJnt", None)
        return np.radians(np.array(pose[:6]))

    def get_pose(self):
        """
        x,y,z,r,p,y,fig
        :return:
        author: weiwei
        date: 20220115
        """
        return self.cobottapos2wrspos(self.get_pose_values())

    def get_pose_values(self):
        """
        x,y,z,r,p,y,fig
        :return:
        author: weiwei
        date: 20220115
        """
        pose = self.bcc.robot_execute(self.hrbt, "CurPos", None)
        return_value = np.array(pose[:7])
        return_value[:3] *= .001
        return_value[3:6] = np.radians(return_value[3:6])
        return return_value

    def move_jnts(self, jnt_values: npt.NDArray[float]):
        """
        :param jnt_values:  1x6 np array
        :return:
        author: weiwei
        date: 20210507
        """
        self.hhnd = self.bcc.robot_execute(self.hrbt, "TakeArm", [0, 0])
        time.sleep(0.1)
        jnt_values_degree = np.degrees(jnt_values)
        self.bcc.robot_move(self.hrbt, 1, [jnt_values_degree.tolist(), "J", "@E"], "")
        self.bcc.robot_execute(self.hrbt, "GiveArm", None)
        time.sleep(0.1)

    def move_p(self, pos, rot, speed=100):
        pose = self.wrshomomat2cobbotapos(rm.homomat_from_posrot(pos, rot))
        self.move_pose(pose, speed)

    def move_pose(self, pose, speed=100):
        self.hhnd = self.bcc.robot_execute(self.hrbt, "TakeArm", [0, 0])
        time.sleep(0.1)
        pose = np.array(pose)
        pose_value = copy.deepcopy(pose)
        pose_value[:3] *= 1000
        pose_value[3:6] = np.degrees(pose_value[3:6])
        self.bcc.robot_move(self.hrbt, 1, [pose_value.tolist(), "P", "@E"], f"SPEED={speed}")
        self.bcc.robot_execute(self.hrbt, "GiveArm", None)
        time.sleep(0.1)

    def open_gripper(self, dist=.03, speed=100):
        """
        :param dist:
        :return:
        """
        assert 0 <= dist <= .03
        self.bcc.controller_execute(self.hctrl, "HandMoveA", [dist * 1000, speed])

    def close_gripper(self, dist=.0, speed=100):
        """
        :param dist:
        :return:
        """
        assert 0 <= dist <= .03
        self.bcc.controller_execute(self.hctrl, "HandMoveA", [dist * 1000, speed])

    def defult_gripper(self, dist=.014, speed=100):
        """
        :param dist:
        :return:
        """
        assert 0 <= dist <= .03
        self.bcc.controller_execute(self.hctrl, "HandMoveA", [dist * 1000, speed])

    def P2J(self, pose):
        pose = np.array(pose)
        pose_value = copy.deepcopy(pose)
        pose_value[:3] *= 1000
        pose_value[3:6] = np.degrees(pose_value[3:6])
        return np.radians(self.bcc.robot_execute(self.hrbt, "P2J", pose_value.tolist()))[:6]

    def ik(self, pos, rot):
        pose = self.wrshomomat2cobbotapos(rm.homomat_from_posrot(pos, rot))
        self.P2J(pose)

    def J2P(self, jnt_values):
        jnt_values = np.array(jnt_values)
        jnt_values_degree = np.degrees(jnt_values)
        pose_value = np.radians(self.bcc.robot_execute(self.hrbt, "J2P", jnt_values_degree.tolist()))
        return_value = np.array(pose_value[:7])
        return_value[:3] *= .001
        return_value[3:6] = np.radians(return_value[3:6])
        return return_value

    def null_space_search(self, current_pose):
        pose = copy.deepcopy(current_pose)
        times = 0
        for angle in range(0, 180, 5):
            for i in [-1, 1]:
                try:
                    self.P2J(pose)
                    return pose, times
                except:
                    self.clear_error()
                    times += 1
                    time.sleep(0.1)
                    pose[5] = current_pose[5] + np.radians(angle * i)
        return None, times

    def move_p_nullspace(self, tgt_pos, tgt_rot, k=-1, speed=100):
        pose = self.wrshomomat2cobbotapos(wrshomomat=rm.homomat_from_posrot(tgt_pos, tgt_rot), k=k)
        self.move_to_pose_nullspace(pose, speed)

    def move_to_pose_nullspace(self, pose, speed=100):
        pose, times = self.null_space_search(pose)
        if pose is not None:
            self.move_pose(pose, speed=speed)
            return times
        else:
            raise Exception("No solution!")


if __name__ == '__main__':
    import math
    import numpy as np
    import basis.robot_math as rm
    import robot_sim.robots.cobotta.cobotta_ripps as cbt
    import motion.probabilistic.rrt_connect as rrtc
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[1, 1, .5], lookat_pos=[0, 0, .2])
    gm.gen_frame().attach_to(base)

    robot_s = cbt.CobottaRIPPS()
    robot_x = CobottaX()
    # for i in range(5):
    #     print(i)
    #     robot_x = CobottaX()
    #     robot_x.disconnect()
    # robot_x.defult_gripper()

    eject_jnt_values1 = np.array([1.37435462, 0.98535585, 0.915062, 1.71130978, -1.23317083, -0.93993529])
    eject_jnt_values2 = np.array([1.55101555, 1.0044305, 0.92469737, 1.60398958, -1.46487782, -0.80639233])
    eject_jnt_values3 = np.array([1.41503157, 0.67831314, 1.41330569, 1.45821765, -1.41208026, -0.45279814])
    eject_jnt_values4 = np.array([1.67416974, 0.64160526, 1.4883013, 1.52974369, -1.5222467, -0.38028545])
    eject_jnt_values_list = [eject_jnt_values1, eject_jnt_values2, eject_jnt_values3, eject_jnt_values4]
    record_poae = np.array(
        [1.02445228e-01, 1.48697438e-01, 2.60223845e-01, 1.57047373e+00, -6.51089430e-04, 1.57065059e+00,
         5.00000000e+00])
    # robot_x.move_pose(record_poae)

    eject_pos_values1 = np.array([0.14, 0.3, 0.15, 2.16862219, 1.14515016, 2.21134745, 1.])
    eject_pos_values2 = np.array([0.1, 0.3, 0.15, 1.73981412, 1.15473168, 1.79367149, 1.])
    eject_pos_values3 = np.array([0.14, 0.235, 0.17, 1.53877781, 0.9840029, 1.58184061, 5.])
    eject_pos_values4 = np.array([0.08, 0.235, 0.17, 1.55544263, 0.94070481, 1.72464343, 5.])
    eject_pos_values_list = [eject_pos_values1, eject_pos_values2, eject_pos_values3, eject_pos_values4]


    def move_to_new_pose(pose, speed=100):
        pose, times = robot_x.null_space_search(pose)
        if pose is not None:
            robot_x.move_pose(pose, speed=speed)
            return times
        else:
            raise Exception("No solution!")


    def vertical_pose():
        pose = robot_x.get_pose_values()
        pose[4] = 0
        pose[3] = np.pi / 2
        robot_x.move_pose(pose)


    # robot_x.move_pose(eject_pos_values_list[0])
    for i in range(4):
        time.sleep(1)
        # robot_x.move_jnts(eject_jnt_values_list[i])
        # robot_x.move_pose(eject_pos_values_list[i])
        # current_pose = robot_x.get_pose_values()
        # print(current_pose)
        # time.sleep(0.1)
        # current_pose[2] -= 0.05
        # move_to_new_pose(current_pose)
        # robot_x.close_gripper()
        # robot_x.defult_gripper()
        # current_pose = robot_x.get_pose_values()
        # time.sleep(0.1)
        # current_pose[2] += 0.06
        # move_to_new_pose(current_pose)
        # input()

    # current_pose = robot_x.get_pose_values()
    # current_pose[2] = 0.23
    # current_pose[3] = math.pi/2
    # current_pose[4]=0
    # robot_x.move_pose(current_pose)
    #
    pose_0 = np.array([3.13938052e-01, 1.14694316e-02, 2.06249518e-01, 1.57079480e+00, 1.35045584e-05, 1.57050228e+00,
                       5.00000000e+00])
    pose_30 = np.array([3.17465267e-01, 1.24703030e-02, 2.06320865e-01, 1.57076624e+00, 1.28756352e-05, 2.09432500e+00,
                        5.00000000e+00])
    pose_60 = np.array([3.15938607e-01, 8.98071678e-03, 2.06288399e-01, 1.57083038e+00, -1.95680153e-05, 1.04722283e+00,
                        5.00000000e+00])

    # robot_x.move_pose(pose_0)
    print(robot_x.get_pose_values())
    print(robot_x.get_jnt_values())
    current_pose = robot_x.get_pose_values()
    # current_pose[0] = current_pose[0]+0.001
    # current_pose[1] = current_pose[1]+0.002
    # current_pose[5] = np.pi/2
    # robot_x.move_pose(pose_0)
    # vertical_pose()
    # robot_x.move_pose(np.array([0.23,0.25,0.23,np.pi*0.5,0,np.pi*0.5,5]))
    # vertical_pose()

    current_jnts = robot_x.get_jnt_values()
    robot_s.fk(jnt_values=current_jnts)
    robot_s.gen_meshmodel(toggle_tcpcs=True, toggle_jntscs=True).attach_to(base)
    # base.run()
    # robot_x.move_jnts(np.array([0,0,np.pi/2,0,0,0,5]))
    # pose = np.array([2.05635125e-01, 5.58309103e-03, 1.41542739e-01, 1.57103164, 6.31578171e-04, 1.04726525e+00, 5.])
    # pose = np.array([1.99862692e-01, 5.08963298e-03, 1.41543218e-01, 1.57110944e+00, -3.80156293e-02, 2.09479858, 257])
    # robot_x.move_pose(pose)

    # robot_x.open_gripper()
    # robot_x.close_gripper()
    robot_x.defult_gripper()

    # robot_x.open_gripper()
    # time.sleep(0.5)
    # robot_x.defult_gripper(speed=60)
    # start_conf = robot_x.get_jnt_values()
    # print("start_radians", start_conf)
    # tgt_pos = np.array([.25, .2, .15])
    # tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3)
    # jnt_values = robot_s.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
    # rrtc_planner = rrtc.RRTConnect(robot_s)
    # path = rrtc_planner.plan(component_name="arm",
    #                          start_conf=start_conf,
    #                          goal_conf=jnt_values,
    #                          ext_dist=.1,
    #                          max_time=300)
    # robot_x.move_jnts_motion(path)
    # robot_x.close_gripper()
    # for pose in path:
    #     robot_s.fk("arm", pose)
    #     robot_meshmodel = robot_s.gen_meshmodel()
    #     robot_meshmodel.attach_to(base)
    # base.run()

# [ 2.75611010e-01 -2.65258214e-02  1.95358273e-01  1.57197528e+00  -1.22512188e-03  1.57128610e+00  5.00000000e+00]
# [ 2.12097060e-01  1.01848105e-01  1.95275462e-01  1.57236128e+00#  -9.17989652e-04  1.57119442e+00  5.00000000e+00]
