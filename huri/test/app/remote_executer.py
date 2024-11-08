import copy
from huri.components.pipeline.data_pipeline import motion_executer2
from huri.core.common_import import *
import huri.test.app.remote_exe_pb2_grpc as re_rpc
import huri.test.app.remote_exe_pb2 as re_msg
import grpc
import pickle
from concurrent import futures
from huri.components.yumi_control.yumi_con import YumiController
from huri.math.units import Mm
from huri.definitions.utils_structure import MotionBatch
from huri.core._logging import color_logger, logging
import motion.probabilistic.rrt_connect as rrtc


def go_init_pose(yumi_s: ym.Yumi, yumi_x: YumiController, component_name="rgt_arm", method="RRT", speed_n=300,
                 logger=logging.getLogger(__name__)):
    logger.setLevel(logging.INFO)
    logger.info(f'Start to move the robot back to the initial pose. Use RRT: {method.lower() == "rrt"}')
    jnt_vals_rgt = yumi_x.get_jnt_values(component_name="rgt_arm")
    jnt_vals_lft = yumi_x.get_jnt_values(component_name="lft_arm")
    if method.lower() == "rrt":
        # initialize the module for RRT
        rrtc_planner = rrtc.RRTConnect(yumi_s)
        # the left and right arm go initial pose
        if component_name in ["rgt_arm", "rgt_hnd", "both"]:
            rrt_path_rgt = rrtc_planner.plan(component_name="rgt_arm",
                                             start_conf=np.array(jnt_vals_rgt),
                                             goal_conf=np.array(yumi_s.rgt_arm.homeconf),
                                             obstacle_list=[],
                                             ext_dist=.05,
                                             max_time=300)
            yumi_x.move_jntspace_path(component_name="rgt_arm", path=rrt_path_rgt, speed_n=speed_n)
        if component_name in ["lft_arm", "lft_hnd", "both"]:
            rrt_path_lft = rrtc_planner.plan(component_name="lft_arm",
                                             start_conf=np.array(jnt_vals_lft),
                                             goal_conf=np.array(yumi_s.lft_arm.homeconf),
                                             obstacle_list=[],
                                             ext_dist=.05,
                                             max_time=300)
            yumi_x.move_jntspace_path(component_name="lft_arm", path=rrt_path_lft, speed_n=speed_n)
    else:
        if component_name in ["rgt_arm", "rgt_hnd", "both"]:
            yumi_x.move_jnts(component_name="rgt_arm", jnt_vals=yumi_s.rgt_arm.homeconf, speed_n=speed_n)
        if component_name in ["lft_arm", "lft_hnd", "both"]:
            yumi_x.move_jnts(component_name="lft_arm", jnt_vals=yumi_s.lft_arm.homeconf, speed_n=speed_n)


class Remote_Executer(re_rpc.RemoteExecuterServicer):
    def __init__(self, base, real_rbt=False, motion_batch_init=None):
        self.rbt = ym.Yumi()
        if real_rbt:
            self.yumi_con = self.init_real_rbt()
        else:
            self.yumi_con = None
        if motion_batch_init is None:
            self.motion_batch = MotionBatch()
        else:
            self.motion_batch = motion_batch_init
        self.is_motion_exe_flag = False
        self.base = base
        self.init_task()

    def init_real_rbt(self, rbt_speed=100, open_gripper_len=0.035):
        try:
            yumi_con = YumiController()
        except:
            yumi_con = YumiController()
        go_init_pose(self.rbt, yumi_con, component_name="rgt_arm")
        yumi_con.set_gripper_width(component_name="rgt_arm", width=open_gripper_len)
        yumi_con.set_gripper_speed("rgt_arm", 10)

        self.rbt_speed = rbt_speed
        self.open_gripper_len = open_gripper_len

        return yumi_con

    def run_motion(self, request, context):
        try:
            data = request.data
            self.motion_batch.add_(pickle.loads(data))
            self.is_motion_exe_flag = False
            print("New motion arrived!!")
        except Exception as e:
            print(e)
            return re_msg.Status(value=re_msg.Status.ERROR)
        return re_msg.Status(value=re_msg.Status.DONE)

    def init_task(self):
        def update(base,
                   robot_s,
                   yumi_con,
                   task):
            if base.inputmgr.keymap["space"]:
                base.inputmgr.keymap["space"] = False
                print("is motion finished exec:", self.is_motion_exe_flag)
                if yumi_con is not None and not self.is_motion_exe_flag:
                    print("motion exec")
                    _exe_element = self.motion_batch.current
                    motion_executer2(yumi_con=yumi_con,
                                     motion_element=_exe_element,
                                     open_gripper_len=self.open_gripper_len,
                                     speed_n=self.rbt_speed)
                try:
                    motion_element = next(self.motion_batch)
                except StopIteration:
                    self.is_motion_exe_flag = True
                    return task.again
                except Exception as e:
                    raise Exception(e)
            else:
                if len(self.motion_batch) > 0:
                    motion_element = self.motion_batch.current
                else:
                    motion_element = None
            if motion_element is not None:
                try:
                    objcm, obj_pose, pose, jawwdith, hand_name, obs_list = next(motion_element)
                except StopIteration:
                    return task.again
                except Exception as e:
                    return Exception(e)
                robot_s.fk(hand_name, pose)
                robot_s.jaw_to(hand_name, jawwdith)
                robot_meshmodel = robot_s.gen_meshmodel()
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

        taskMgr.doMethodLater(0.08, update, "update",
                              extraArgs=[self.base,
                                         self.rbt,
                                         self.yumi_con],
                              appendTask=True)


def serve(host="localhost:18300", is_real_rbt=False, motion_batch_init_path=None):
    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    options = [('grpc.max_send_message_length', 100 * 1024 * 1024),
               ('grpc.max_receive_message_length', 100 * 1024 * 1024)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    if motion_batch_init_path is not None:
        motion_batch_init = fs.load_pickle(motion_batch_init_path)
    else:
        motion_batch_init = None
    rvs = Remote_Executer(base, real_rbt=is_real_rbt, motion_batch_init=motion_batch_init)
    re_rpc.add_RemoteExecuterServicer_to_server(rvs, server)
    server.add_insecure_port(host)
    server.start()
    try:
        print("The Remote Exe server is started!")
        base.run()
    except SystemExit:
        from time import strftime
        fs.dump_pickle(rvs.motion_batch,
                       fs.workdir / "test" / "app" / "run" / f"motion_{strftime('%Y%m%d-%H%M%S')}.pkl")
        print("Save pickle successfully")


if __name__ == "__main__":
    is_real_rbt = False
    serve(is_real_rbt=is_real_rbt, motion_batch_init_path=None)
