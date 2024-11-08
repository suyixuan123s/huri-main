"""
这是一个示例，用于：
1. 生成仿真机器人执行的线性运动
2. 使用 RRT 将物体从一个位置移动到另一个位置
（请先运行 4_define_grasp.py 文件）
"""
from huri.core.common_import import *
import motion.optimization_based.incremental_nik as inik
import motion.probabilistic.rrt_connect as rrtc

# 创建虚拟环境
base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

# 生成 Yumi 机器人
ym_rbt = ym.Yumi(enable_cc=True)
component_name = "rgt_arm"  # 设置操作的手臂为右臂

# 生成要抓取的管状物体，并定义物体的初始和目标位姿
obj_mdl = cm.CollisionModel(initor="../../models/tubebig.stl")
obj_mdl_init_pose = rm.homomat_from_posrot(np.array([0.2, -.07, 0.0]), np.eye(3))  # 初始位置
obj_mdl_tgt_pose = rm.homomat_from_posrot(np.array([0.36, 0.07, 0.2]), np.eye(3))  # 目标位置
obj_mdl.set_homomat(obj_mdl_init_pose)
obj_mdl.set_rgba(np.array([0, 0, 0, 1]))  # 设置物体颜色为黑色
obj_mdl.attach_to(base)  # 将物体添加到虚拟环境中

# 初始化线性运动生成模块
inik_svlr = inik.IncrementalNIK(robot_s=ym_rbt)

# 初始化 RRT 模块
rrtc_planner = rrtc.RRTConnect(ym_rbt)

# 加载物体的抓取姿势
grasps_list_info = fs.load_pickle("grasps.pkl")

# 保存路径的列表
path = []

# 搜索可以将物体移动到目标位姿的抓取姿势
for ind, (jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat) in enumerate(grasps_list_info):
    print(f"--------------------- 抓取姿势索引: {ind} ---------------------------")
    # 生成齐次变换矩阵（包含位移和旋转的 4x4 矩阵）
    grasp_pose = rm.homomat_from_posrot(jaw_center_pos, jaw_center_rotmat)
    print(f"抓取姿势的齐次矩阵为: {grasp_pose}")
    # 求解物体初始位姿的逆运动学解
    rbt_ee_pose_init = np.dot(obj_mdl_init_pose, grasp_pose)
    ik_sol_init = ym_rbt.ik(component_name, tgt_pos=rbt_ee_pose_init[:3, 3], tgt_rotmat=rbt_ee_pose_init[:3, :3])
    # 求解物体目标位姿的逆运动学解
    rbt_ee_pose_tgt = np.dot(obj_mdl_tgt_pose, grasp_pose)
    ik_sol_tgt = ym_rbt.ik(component_name, tgt_pos=rbt_ee_pose_tgt[:3, 3],
                           tgt_rotmat=rbt_ee_pose_tgt[:3, :3],
                           seed_jnt_values=ik_sol_init)
    ym_rbt.rgt_hnd.jaw_to(jaw_width)  # 设置右手夹爪的开口宽度
    if ik_sol_init is not None and ik_sol_tgt is not None:  # 检查逆运动学解是否可行
        ym_rbt.fk(component_name, ik_sol_init)
        is_self_collided_init = ym_rbt.is_collided()
        ym_rbt.fk(component_name, ik_sol_tgt)
        is_self_collided_tgt = ym_rbt.is_collided()
        if is_self_collided_init or is_self_collided_tgt:  # 检查是否发生自碰撞
            print(">>> 机器人发生了自碰撞")
            continue
        else:
            # 生成机器人抬起管道的运动路径
            ym_rbt.fk(component_name, ik_sol_init)
            rbt_tcp_pos, rbt_tcp_rot = ym_rbt.get_gl_tcp(component_name)
            obj_mdl_grasped = obj_mdl.copy()
            ym_rbt.hold(hnd_name="rgt_hnd", objcm=obj_mdl_grasped, jaw_width=jaw_width)
            path_up = inik_svlr.gen_linear_motion(component_name,
                                                  start_tcp_pos=rbt_tcp_pos,
                                                  start_tcp_rotmat=rbt_tcp_rot,
                                                  goal_tcp_pos=rbt_tcp_pos + np.array([0, 0, .15]),
                                                  goal_tcp_rotmat=rbt_tcp_rot,
                                                  granularity=0.01)
            if path_up is not None:
                # 生成移动管道到目标位置的运动路径
                rrt_path = rrtc_planner.plan(component_name=component_name,
                                             start_conf=np.array(path_up[-1]),
                                             goal_conf=np.array(ik_sol_tgt),
                                             obstacle_list=[],
                                             ext_dist=.05,
                                             max_time=300)
                if rrt_path is not None:
                    # 显示路径
                    path = path_up + rrt_path
                    for jnts_s in path:
                        ym_rbt.fk(component_name, jnts_s)
                        ym_rbt.gen_meshmodel().attach_to(base)
                    base.run()
                else:
                    print(">>> 无法通过 RRT 生成将物体移动到目标位姿的路径")
                    ym_rbt.release("rgt_hnd", objcm=obj_mdl_grasped)
            else:
                print(">>> 无法生成抬起管道的路径")
                ym_rbt.release("rgt_hnd", objcm=obj_mdl_grasped)
    else:
        print("初始位姿或目标位姿无逆运动学解")
exit(-1)  # 程序退出
