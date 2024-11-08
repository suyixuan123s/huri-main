"""
这是一个示例，用于：
1. 生成机器人执行“抓取-放置”运动，将试管从一个架子移动到另一个架子
"""
from huri.definitions.rack_def import Rack_Hard_Proto, TubeType
from huri.core.common_import import *
from huri.core.print_tool import text_pd
from itertools import product
import motion.optimization_based.incremental_nik as inik
import motion.probabilistic.rrt_connect as rrtc

# 创建虚拟环境
base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

# 生成 Yumi 机器人及其模型
ym_rbt = ym.Yumi(enable_cc=True)
component_name = "rgt_arm"

# 初始化生成线性运动的模块
inik_svlr = inik.IncrementalNIK(robot_s=ym_rbt)

# 初始化 RRT 模块
rrtc_planner = rrtc.RRTConnect(ym_rbt)

# 生成试管架并设置位置
rack_1 = Rack_Hard_Proto.copy()
rack_1.set_homomat(rm.homomat_from_posrot(pos=np.array([0.4, -.2, 0])))
# 向试管架插入试管
rack_1.insert_tube(slot_id=np.array([0, 3]), tube=TubeType.TUBE_TYPE_3)
rack_1.insert_tube(slot_id=np.array([4, 1]), tube=TubeType.TUBE_TYPE_3)
rack_1.insert_tube(slot_id=np.array([3, 2]), tube=TubeType.TUBE_TYPE_3)
rack_1.insert_tube(slot_id=np.array([2, 3]), tube=TubeType.TUBE_TYPE_3)
print("插入试管后的试管架状态: ")
print(text_pd(rack_1.rack_status))
print("-" * 30)

# 从原型生成新的试管架
rack_2 = Rack_Hard_Proto.copy()
rack_2.set_homomat(rm.homomat_from_posrot(pos=np.array([0.4, 0, 0])))

# 选择要移除的试管
remove_ind = np.array([3, 2])
tube_type, tube_homomat_gl = rack_1.remove_slot(slot_id=remove_ind)
tube_move = TubeType.gen_tube_by_tubetype(tube_type)
tube_move.set_homomat(tube_homomat_gl)
tube_cm_mdl = tube_move.gen_collision_model()
tube_cm_mdl.attach_to(base)
# 获取试管的对称变换矩阵
symmetry_homomat_init = tube_move.get_symmetry_homomat(discrete_factor=10)

# 选择要插入新试管架的位置
insert_ind = np.array([0, 1])
tgt_tube_pose = rack_2.get_slot_homomat(slot_id=insert_ind)
symmetry_homomat_tgt = tube_move.get_symmetry_homomat(discrete_factor=10, rot_mat=tgt_tube_pose)

# 加载试管的抓取信息
tube_move_grasps_list_info = fs.load_json(tube_move.grasps_path)
# 生成环境中的碰撞模型
obs_rack_1, obs_tube_collection = rack_1.gen_collision_model(gen_tube=True)
obs_rack_2 = rack_2.gen_collision_model()

# 保存“抓取-放置”路径的列表
path = []

# 搜索可以移动物体到目标位置的抓取姿势
for ind, (jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat) in enumerate(tube_move_grasps_list_info):
    print(f"--------------------- 抓取姿势索引: {ind} ---------------------------")
    grasp_pose = rm.homomat_from_posrot(jaw_center_pos, jaw_center_rotmat)
    print(f"抓取姿势的齐次矩阵为: {grasp_pose}")
    # 找到初始位置的逆运动学解集
    rbt_ee_poses_init = np.dot(symmetry_homomat_init.reshape((-1, 4)), grasp_pose).reshape((-1, 4, 4))
    ik_sol_set_init = []
    for _rbt_ee_pose_init in rbt_ee_poses_init:
        ik_sol_init = ym_rbt.ik(component_name, tgt_pos=_rbt_ee_pose_init[:3, 3], tgt_rotmat=_rbt_ee_pose_init[:3, :3])
        if ik_sol_init is not None:
            ik_sol_set_init.append(ik_sol_init)
    # 找到目标位置的逆运动学解集
    rbt_ee_poses_tgt = np.dot(symmetry_homomat_tgt.reshape((-1, 4)), grasp_pose).reshape((-1, 4, 4))
    ik_sol_set_tgt = []
    for _rbt_ee_pose_tgt in rbt_ee_poses_tgt:
        ik_sol_tgt = ym_rbt.ik(component_name, tgt_pos=_rbt_ee_pose_tgt[:3, 3], tgt_rotmat=_rbt_ee_pose_tgt[:3, :3])
        if ik_sol_tgt is not None:
            ik_sol_set_tgt.append(ik_sol_tgt)
    if len(ik_sol_set_init) > 0 and len(ik_sol_set_tgt) > 0:
        # 检查碰撞
        for _ik_sol_init, _ik_sol_tgt in product(ik_sol_set_init, ik_sol_set_tgt):
            ym_rbt.fk(component_name, _ik_sol_init)
            is_collided_init = ym_rbt.is_collided([obs_rack_1, obs_rack_2, *obs_tube_collection.cm_list])
            ym_rbt.fk(component_name, _ik_sol_tgt)
            is_collided_tgt = ym_rbt.is_collided([obs_rack_1, obs_rack_2, *obs_tube_collection.cm_list])
            if is_collided_init or is_collided_tgt:
                print(">>> 机器人发生了碰撞")
                continue
            else:
                # 生成抓取动作
                ym_rbt.fk(component_name, _ik_sol_init)
                rbt_tcp_pos, rbt_tcp_rot = ym_rbt.get_gl_tcp(component_name)
                obj_mdl_grasped = tube_cm_mdl.copy()
                obj_mdl_grasped.set_homomat(symmetry_homomat_init[0])
                ym_rbt.hold(hnd_name="rgt_hnd", objcm=obj_mdl_grasped, jaw_width=jaw_width)
                path_up = inik_svlr.gen_linear_motion(component_name,
                                                      start_tcp_pos=rbt_tcp_pos,
                                                      start_tcp_rotmat=rbt_tcp_rot,
                                                      goal_tcp_pos=rbt_tcp_pos + np.array([0, 0, .15]),
                                                      goal_tcp_rotmat=rbt_tcp_rot,
                                                      obstacle_list=[*obs_tube_collection.cm_list, obs_rack_2],
                                                      granularity=0.01)
                ym_rbt.release(hnd_name="rgt_hnd", objcm=obj_mdl_grasped)

                # 生成放置动作
                ym_rbt.fk(component_name, _ik_sol_tgt)
                rbt_tcp_pos, rbt_tcp_rot = ym_rbt.get_gl_tcp(component_name)
                obj_mdl_grasped.set_homomat(symmetry_homomat_tgt[0])
                ym_rbt.hold(hnd_name="rgt_hnd", objcm=obj_mdl_grasped, jaw_width=jaw_width)
                path_down_reverse = inik_svlr.gen_linear_motion(component_name,
                                                                start_tcp_pos=rbt_tcp_pos,
                                                                start_tcp_rotmat=rbt_tcp_rot,
                                                                goal_tcp_pos=rbt_tcp_pos + np.array([0, 0, .15]),
                                                                goal_tcp_rotmat=rbt_tcp_rot,
                                                                obstacle_list=[*obs_tube_collection.cm_list,
                                                                               obs_rack_1],
                                                                granularity=0.01)
                if path_up is not None and path_down_reverse is not None:
                    path_down = path_down_reverse[:: -1]
                    rrt_path = rrtc_planner.plan(component_name=component_name,
                                                 start_conf=np.array(path_up[-1]),
                                                 goal_conf=np.array(path_down[0]),
                                                 obstacle_list=[obs_rack_1, obs_rack_2, *obs_tube_collection.cm_list],
                                                 ext_dist=.05,
                                                 max_time=300)
                    if rrt_path is not None:
                        # 将试管插入目标试管架
                        rack_2.insert_tube(slot_id=insert_ind, tube=tube_move)
                        # 显示路径
                        path = path_up + rrt_path + path_down
                        for jnts_s in path:
                            rack_1.gen_mesh_model(gen_tube=True).attach_to(base)
                            rack_2.gen_mesh_model(gen_tube=True).attach_to(base)
                            ym_rbt.fk(component_name, jnts_s)
                            ym_rbt.gen_meshmodel().attach_to(base)
                            obj_mdl_draw = obj_mdl_grasped.copy()
                            obj_mdl_draw.attach_to(base)
                        base.run()
                else:
                    print(">>> 无法生成路径将物体移动到目标位置")
                    ym_rbt.release("rgt_hnd", objcm=obj_mdl_grasped)
        else:
            print(">>> 无法生成抬起试管的路径")
            ym_rbt.release("rgt_hnd", objcm=obj_mdl_grasped)
    else:
        print("初始位置或目标位置无逆运动学解")
exit(-1)  # 程序退出
