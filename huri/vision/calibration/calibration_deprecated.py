import vision.depth_camera.refine_depth_calibrator as dcdc
import vision.depth_camera.sensor_marker_handler as dcsmh
import robot_sim.robots.yumi.yumi as ym
from huri.components.yumi_control.yumi_con import YumiController
import numpy as np
from huri.core.common_import import *
from huri.components.pipeline.data_pipeline import RenderController
from time import strftime
import basis.trimesh.primitives as btp

if __name__ == "__main__":
    ################ Calibration MATRIX NEEDED TO BE REFINED ################
    AFFINE_MAT_PATH = fs.workdir / "data" / "calibration" / "affine_mat_20210918-210002.json"
    ########################################################################

    '''initilization'''
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    std_out = RenderController(root=base.tkRoot, base=base)
    robot_s = ym.Yumi()  # simulation robot
    robot_x = YumiController()  # robot control handler
    calibrator = dcdc.DepthCaliberator(robot_x=robot_x, robot_s=robot_s,
                                       affine_mat_path=AFFINE_MAT_PATH)  # init calibrator

    rgt_arm_jnts = robot_s.get_jnt_values(component_name="rgt_arm")
    robot_x.move_jnts(component_name="rgt_arm",
                      jnt_vals=[1.6533504, -1.74690005, -1.1974704, 1.04388143, -2.28341426, 1.66836023, -0.42236968])
    '''calibration
    Step 1: Find the marker pos on the hand
    Step 2: Generate the calibration matrix
    '''

    # Step 1

    marker_pos_in_hnd, inhand_rotmat = calibrator.find_board_center_in_hand(
        component_name="rgt_arm",
        sensor_marker_handler=sensor_handler,
        action_center_pos=np.array([.32, 0.02, .25]),
        action_center_rotmat=np.dot(
            rm.rotmat_from_axangle([1, 0, 0], np.radians(0)),
            np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).T
        ),
        action_dist=.1,
        std_out=std_out)
    # marker_pos_in_hnd = np.array([0.0013179 , 0.00947674 ,0.0226476 ])
    # [5.79355314 0.34722762 1.31885981]
    # [-0.6899418   0.04762786  0.40279171]
    # [-0.00084689  0.01082143  0.04170861]
    # [-0.0017815204343458958, 0.011339426137114568, 0.04339973373974998]
    print("chen marker in hand", marker_pos_in_hnd)
    # base.run()
    # marker_pos_in_hnd_inrobot = marker_pos_in_hnd
    # marker_pos_in_hnd_inrobot[2] = -marker_pos_in_hnd_inrobot[2]
    # # marker_pos_in_hnd_inrobot[1] = - marker_pos_in_hnd_inrobot[1]
    # print(marker_pos_in_hnd_inrobot)
    #
    # board = cm.CollisionModel(btp.Box(box_extents=[0.005, 0.15, 0.1]))
    robot_s.fk("rgt_arm", robot_x.get_jnt_values("rgt_arm"))
    robot_s.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
    rp, rrot = robot_s.get_gl_tcp("rgt_arm")
    gm.gen_frame(rp, rrot, length=.2).attach_to(base)

    # # Step 2
    pos_list = []
    for x in [300 / 1000, 360 / 1000, 400 / 1000]:
        for y in [-200 / 1000, 0, 200 / 1000]:
            for z in [70 / 1000, 90 / 1000, 130 / 1000, 200 / 1000]:
                pos_list.append(np.array([x, y, z]))

    matrix = calibrator.calibrate(
        component_name="rgt_arm",
        marker_pos_in_hnd=marker_pos_in_hnd,
        action_pos_list=pos_list,
        action_rotmat_list=[np.dot(
            rm.rotmat_from_axangle([0, 1, 0], np.radians(0)),
            np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]
                     )).T] * len(pos_list),
        sensor_marker_handler=sensor_handler
    )

    print("calibrated matrix is", matrix)

    fs.dump_json({'affine_mat': matrix.tolist(), 'marker_pos_in_hnd': marker_pos_in_hnd.tolist()},
                 fs.workdir / "data" / "calibration" / f"affine_mat_{strftime('%Y%m%d-%H%M%S')}.json")
    robot_s.fk(component_name="rgt_arm",
               jnt_values=robot_x.get_jnt_values(component_name="rgt_arm"))
    robot_s.gen_meshmodel().attach_to(base)
    pcd = sensor_handler.get_pcd()
    pcd = rm.homomat_transform_points(matrix, points=pcd)
    gm.gen_pointcloud(pcd, [[0, 0, 1, .3]]).attach_to(base)
    # gm.gen_frame(pos=robot_s.get_gl_tcp("rgt_arm")[0],rotmat=inhand_rotmat).attach_to(base)
    # robot_s.fk("rgt_arm", robot_x.get_jnt_values("rgt_arm"))
    # robot_rot = robot_s.get_gl_tcp("rgt_arm")[1]
    # zaxis = robot_rot[:3,2]
    # marker_pos_in_hnd_inrobot = np.dot(robot_rot,marker_pos_in_hnd_inrobot)
    # marker_pos_in_hnd = np.dot(robot_rot,marker_pos_in_hnd)
    #
    # rot = rm.rotmat_between_vectors(zaxis, marker_pos_in_hnd_inrobot)
    # robot_s_p = robot_s.get_gl_tcp("rgt_arm")[0]
    # gm.gen_arrow(spos=robot_s_p, epos=robot_s_p + zaxis).attach_to(base)
    # gm.gen_arrow(spos=robot_s_p, epos=robot_s_p - marker_pos_in_hnd_inrobot * 2).attach_to(base)
    # gm.gen_sphere(pos=robot_s_p + marker_pos_in_hnd_inrobot, radius=.005).attach_to(base)
    # gm.gen_sphere(pos=robot_s_p + marker_pos_in_hnd, rgba=[0,0,1,1]).attach_to(base)
    # # board.set_homomat(
    # #     np.dot(rm.homomat_from_posrot(marker_pos_in_hnd_inrobot_, rot),
    # #     rm.homomat_from_posrot(*robot_s.get_gl_tcp("rgt_arm"))
    # #            ))
    # board.show_localframe()
    # board.set_homomat(
    #     np.dot(rm.homomat_from_posrot(np.array([0,0,0]), rot),
    #            rm.homomat_from_posrot(*robot_s.get_gl_tcp("rgt_arm"))
    #            ))
    # board.set_pos(
    #     robot_s_p + marker_pos_in_hnd_inrobot - board.get_rotmat()[:3,2]*0.02
    # )
    # board.attach_to(base)
    # robot_s.gen_meshmodel().attach_to(base)

    base.run()
