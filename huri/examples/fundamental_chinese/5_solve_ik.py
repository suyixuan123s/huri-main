"""
这是一个示例，用于：
1. 求解逆运动学 (IK)
2. 检查仿真机器人自身的碰撞情况
（请先运行 4_define_grasp.py 文件）
"""
import numpy as np

from huri.core.common_import import *

# 创建虚拟环境
base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

# 生成 Yumi 机器人
ym_rbt = ym.Yumi(enable_cc=True)
component_name = "rgt_arm"  # 指定操作的手臂为右臂

# 生成要抓取的管状物体
obj_mdl = cm.CollisionModel(initor="../../models/tubebig.stl")
obj_mdl.set_pos(np.array([0.36, -.07, 0.1]))  # 设置物体位置
obj_mdl.set_rgba(np.array([0, 0, 0, 1]))  # 设置物体颜色为黑色
obj_mdl.attach_to(base)  # 将物体添加到虚拟环境中

# 加载管道的抓取姿势信息
grasps_list_info = fs.load_pickle("grasps.pkl")

# 遍历每个抓取姿势
for ind, (jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat) in enumerate(grasps_list_info):
    print(f"--------------------- 抓取姿势索引: {ind} ---------------------------")
    obj_pose = obj_mdl.get_homomat()  # 获取物体的位姿矩阵
    grasp_pose = rm.homomat_from_posrot(jaw_center_pos, jaw_center_rotmat)  # 生成抓取姿势的齐次变换矩阵
    print(f"抓取姿势的齐次矩阵为: {grasp_pose}")
    rbt_ee_pose = np.dot(obj_pose, grasp_pose)  # 计算机器人末端执行器的目标位姿

    # 求解逆运动学
    ik_sol = ym_rbt.ik(component_name, tgt_pos=rbt_ee_pose[:3, 3], tgt_rotmat=rbt_ee_pose[:3, :3])
    if ik_sol is not None:  # 检查是否有 IK 解
        ym_rbt.fk(component_name, ik_sol)  # 应用求解的关节角度进行正向运动学计算
        if ym_rbt.is_collided():  # 检查是否发生自碰撞
            print("机器人发生了自碰撞")
            continue
        else:
            # 显示用于抓取管道的 IK 解
            ym_rbt.rgt_hnd.jaw_to(jaw_width)  # 设置右手夹爪的宽度
            ym_rbt.gen_meshmodel().attach_to(base)  # 将机器人的模型添加到虚拟环境中显示
            base.run()  # 运行虚拟环境
exit(-1)  # 程序退出
