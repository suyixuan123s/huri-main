"""
这是一个示例，用于手动定义抓取姿势
"""
from huri.core.common_import import *
import robot_sim.end_effectors.gripper.yumi_gripper.yumi_gripper as yg
from grasping.annotation.utils import define_grasp_with_rotation

# 创建虚拟环境
base = wd.World(cam_pos=[.5, .5, .3], lookat_pos=[0, 0, 0])
gripper_s = yg.YumiGripper(enable_cc=True)  # 创建 Yumi 机器人的夹爪

# 生成碰撞模型并添加到虚拟环境中
objpath = "../../models/tubebig.stl"
objcm = cm.CollisionModel(objpath)
objcm.attach_to(base)  # 将物体添加到虚拟环境中
objcm.show_localframe()  # 显示物体的局部坐标系



# 定义抓取姿势
grasp_info_list = define_grasp_with_rotation(gripper_s,
                                             objcm,
                                             gl_jaw_center_pos=np.array([0, 0, .09]),  # 夹爪中心的位置
                                             gl_jaw_center_z=
                                             np.dot(rm.rotmat_from_axangle(np.array([1, 0, 0]),
                                                                           np.radians(-30)),
                                                    rm.unit_vector(np.array([0, 1, 0]))),  # 夹爪中心的 Z 轴方向
                                             gl_jaw_center_y=
                                             np.dot(rm.rotmat_from_axangle(np.array([1, 0, 0]),
                                                                           np.radians(-30)),
                                                    np.array([0, 0, 1])),  # 夹爪中心的 Y 轴方向
                                             jaw_width=.018,  # 夹爪的宽度
                                             rotation_interval=np.radians(60),  # 旋转角度间隔
                                             gl_rotation_ax=np.array([0, 0, 1]),  # 抓取姿势的旋转轴
                                             toggle_debug=True)  # 开启调试模式

print("抓取姿势的数量为", len(grasp_info_list))

# 保存抓取信息列表
fs.dump_pickle(grasp_info_list, "grasps.pkl")

# 显示所有的抓取姿势
for grasp_info in grasp_info_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    gic = gripper_s.copy()  # 创建夹爪的副本
    gic.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)  # 根据姿势调整夹爪
    gic.gen_meshmodel().attach_to(base)  # 将生成的夹爪模型添加到虚拟环境中

# 运行虚拟环境
base.run()
