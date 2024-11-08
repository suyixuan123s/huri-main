from time import strftime

from utils import CalibrationSettings, RgtArmInfo, LftArmInfo
from huri.core.common_import import *
from huri.core.constants import SENSOR_INFO
from huri.components.yumi_control.yumi_con import YumiController
from huri.components.pipeline.data_pipeline import RenderController
import vision.depth_camera.depth_calibrator as dcdc
import vision.depth_camera.sensor_marker_handler as dcsmh

if __name__ == "__main__":
    '''initilization'''
    base = wd.World(cam_pos=[0.3, 0, 1], lookat_pos=[0.3, 0, 0])
    std_out = RenderController(root=base.tkRoot, base=base)
    robot_s = ym.Yumi()  # simulation robot
    robot_x = YumiController()  # robot control handler
    calibrator = dcdc.DepthCaliberator(robot_x=robot_x, robot_s=robot_s)  # init calibrator

    sensor_handler = dcsmh.SensorMarkerHandler2(SENSOR_INFO.IP_ADR_DEPTH_SENSOR, aruco_target_id_list=[0, 1],
                                                toggle_cv_show=True)
    sensor_handler.sensor_client.changeledpower(CalibrationSettings.led_power)
    ArmInfo = RgtArmInfo if CalibrationSettings.work_arm == 'rgt_arm' else LftArmInfo
    robot_x.move_jnts(component_name=ArmInfo.arm,
                      jnt_vals=ArmInfo.init_jnt_val)
    robot_s.fk(ArmInfo.arm, robot_x.get_jnt_values(ArmInfo.arm))
    arm_jnts = robot_x.get_jnt_values(component_name=ArmInfo.arm)

    '''calibration
    Step 1: Find the marker pos on the hand
    Step 2: Generate the calibration matrix
    '''
    # Step 1
    marker_pos_in_hnd, inhand_rotmat = calibrator.find_board_center_in_hand(
        component_name=ArmInfo.arm,
        sensor_marker_handler=sensor_handler,
        action_center_pos=ArmInfo.action_center_pos,
        action_center_rotmat=ArmInfo.action_center_rotmat,
        action_dist=.04,
        std_out=std_out)
    print("Marker in hand", marker_pos_in_hnd)

    # # Step 2
    robot_x.move_jnts(component_name=ArmInfo.arm,
                      jnt_vals=ArmInfo.init_jnt_val)
    pos_list = []
    for x in CalibrationSettings.work_range_x:
        for y in CalibrationSettings.work_range_y:
            for z in CalibrationSettings.work_range_z:
                pos_list.append(np.array([x, y, z]))
    matrix = calibrator.calibrate(
        component_name=ArmInfo.arm,
        marker_pos_in_hnd=marker_pos_in_hnd,
        action_pos_list=pos_list,
        action_rotmat_list=[ArmInfo.action_center_rotmat] * len(pos_list),
        sensor_marker_handler=sensor_handler
    )
    print("calibrated matrix is", matrix)
    fs.dump_json({'affine_mat': matrix.tolist(), 'marker_pos_in_hnd': marker_pos_in_hnd.tolist()},
                 fs.workdir / "data" / "calibration" / f"affine_mat_{strftime('%Y%m%d-%H%M%S')}.json")
    robot_s.fk(component_name=ArmInfo.arm,
               jnt_values=robot_x.get_jnt_values(component_name=ArmInfo.arm))
    robot_s.gen_meshmodel(rgba=[0, 0, 0, .5]).attach_to(base)
    pcd = sensor_handler.get_pcd()
    pcd = rm.homomat_transform_points(matrix, points=pcd)
    gm.gen_pointcloud(pcd, [[0, 0, 1, .3]]).attach_to(base)
    base.run()
