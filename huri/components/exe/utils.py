from time import strftime
from pathlib import Path
import numpy as np
import basis.trimesh as tm
from basis.robot_math import deltaw_between_rotmat
from huri.core.file_sys import workdir, load_pickle, load_json
from huri.core.common_import import cm
from huri.vision.phoxi_capture import vision_pipeline, vision_read_data, SensorMarkerHandler
from huri.components.exe.executer import MotionExeSingleArm
from huri.components.exe.exe_loggining import exe_logger
from huri.components.yumi_control.yumi_con import YumiController


def gen_camera_obs() -> cm.CollisionModel:
    """
    Generate the collision model for the phoxi camera
    :return:
    """
    # setup the collision model for phoxi camera
    camera_obs = cm.CollisionModel(tm.primitives.Box(box_extents=[1, 1, .2]))
    camera_obs.set_pos(np.array([0.1, 0, 1]))
    return camera_obs

def change_led_power(ip_adr="192.168.125.100:18300", led_power=2000):
    streamer = SensorMarkerHandler(ip_adr=ip_adr)
    streamer.set_led_power(led_power)
    del streamer

def capture_vision_info(ip_adr="192.168.125.100:18300",
                        debug_filename=workdir / "data" / "vision_exp" / "20211228-234523.pkl",
                        toggle_save=False,
                        save_path=None,
                        toggle_debug=False,
                        streamer=None):
    if not isinstance(streamer, SensorMarkerHandler):
        streamer = SensorMarkerHandler(ip_adr=ip_adr)
    if toggle_debug:
        pcd, img, depth_img, _, _ = vision_read_data(debug_filename)
    else:
        if toggle_save:
            if save_path is None:
                save_path = workdir / "data" / "vision_exp" / f"{strftime('%Y%m%d-%H%M%S')}.pkl"
        else:
            save_path = None
        pcd, img, depth_img, _, _ = vision_pipeline(streamer,
                                                    save_path,
                                                    rgb_texture=False)
    return pcd, img, depth_img


def init_real_rbt(gripper_len=.035, component_name="rgt_arm", toggle_debug=False, use_motion_exe=False):
    if toggle_debug:
        return None
    if use_motion_exe:
        yumi_con = MotionExeSingleArm(component_name=component_name,
                                      max_open_gripper_len=gripper_len,
                                      logger=exe_logger)
    else:
        yumi_con = YumiController()
    yumi_con.set_gripper_width(component_name=component_name, width=gripper_len)
    yumi_con.set_gripper_speed(component_name=component_name, speed=10)
    return yumi_con


def is_restart_planning(rack_tf1, rack_tf2, pos_thresh=1e-3, rot_thresh=np.pi / 12, toggle_debug=False):
    pos_rack_tf1 = rack_tf1[:3, 3]
    pos_rack_tf2 = rack_tf2[:3, 3]
    rot_rack_tf1 = rack_tf1[:3, :3]
    rot_rack_tf2 = rack_tf2[:3, :3]

    if toggle_debug:
        print(f"Pos diff: {np.linalg.norm(pos_rack_tf2 - pos_rack_tf1)}/{pos_thresh}, "
              f"Rot diff: {np.linalg.norm(deltaw_between_rotmat(rot_rack_tf1, rot_rack_tf2))}/{rot_thresh}")
    if np.linalg.norm(pos_rack_tf2 - pos_rack_tf1) < pos_thresh and np.linalg.norm(
            deltaw_between_rotmat(rot_rack_tf1, rot_rack_tf2)) < rot_thresh:
        return False
    else:
        return True


def create_directory(path) -> Path:
    """
    Create a directory at the specified path if it does not exist.

    :param path: Path of the directory to be created.
    """
    directory_path = Path(path)
    try:
        directory_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory created or already exists: {directory_path}")
    except OSError as error:
        print(f"Error creating directory {directory_path}: {error}")
    return directory_path
