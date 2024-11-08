import time


def cal_feasible_common_grasps_in_seq(grasp_iter, chunk_init_id):
    # feasible_num = 0
    feasible_list = []
    for gid, _ in enumerate(grasp_iter):
        if _[0]:
            # feasible_num += 1
            feasible_list.append(chunk_init_id + gid)
            # save_rbt_info(arm_jnts=_[1],
            #               hand_name=hnd_name,
            #               jawwidth=_[2],
            #               rbt_instance=ym_rbt)
    grasp_iter.print_error_info()
    print(feasible_list)
    return feasible_list


def cal_feasible_common_grasps(args):
    return cal_feasible_common_grasps_in_seq(*args)


if __name__ == "__main__":
    import multiprocessing as mp
    from multiprocessing import Pool
    from huri.definitions.rack_def import Rack_Hard_Proto, TubeType
    from huri.core.common_import import *
    from huri.core.print_tool import text_pd
    import huri.components.planning.symmetric_pick_and_place_planner_c as pppc
    import huri.components.planning.symmetric_pick_and_place_planner as ppp
    from huri.components.debug.inspect import save_rbt_info, show_animation, save_error_info, IKError, RbtCollisionError
    from huri.learning.env.arrangement_planning_rack.utils import get_random_states
    from huri.learning.env.arrangement_planning_rack.env import RackState

    random_seed = 100
    np.random.seed(random_seed)

    # create the virtual environment
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

    # generate the yumi robot and its mesh model
    ym_rbt = ym.Yumi(enable_cc=True)
    ym_rbt.gen_meshmodel().attach_to(base)
    hnd_name = "rgt_hnd"

    # generate a rack from a prototype
    rack_1 = Rack_Hard_Proto.copy()
    rack_1.set_homomat(rm.homomat_from_posrot(pos=np.array([0.4, -.2, 0])))
    # insert test tube to the rack
    random_state = get_random_states(rack_size=np.array([5, 10]),
                                     goalpattern=np.array([[1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                                                           [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                                                           [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                                                           [1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
                                                           [1, 1, 1, 0, 0, 0, 1, 1, 1, 1]]), min_obj_num=20)
    slot_ids = np.vstack(np.where(random_state > 0)).T
    for slot_id in slot_ids:
        rack_1.insert_tube(slot_id=slot_id, tube=TubeType.TUBE_TYPE_3)

    rack_1_state = RackState(rack_1.rack_status)

    # move test tube
    movable_slot_ids = np.vstack(np.where(rack_1_state.movable_slots > 0)).T
    remove_ind = movable_slot_ids[[np.random.choice(np.arange(len(movable_slot_ids)))]].flatten()
    tube_type, tube_homomat_gl = rack_1.remove_slot(slot_id=remove_ind)
    tube_move = TubeType.gen_tube_by_tubetype(tube_type)
    tube_move.set_homomat(tube_homomat_gl)
    tube_cm_mdl = tube_move.gen_collision_model()
    tube_cm_mdl.set_rgba([1, 0, 0, .7])
    tube_cm_mdl.attach_to(base)

    # generate the collision model in the environment
    obs_rack_1, obs_tube_collection = rack_1.gen_collision_model(gen_tube=True)
    obs_rack_1.attach_to(base)
    obs_tube_collection.attach_to(base)
    # [_.attach_to(base) for _ in obs_tube_collection]

    print("The state after inserting the test tube rack: ")
    print(text_pd(rack_1.rack_status))
    print("-" * 30)

    fillable_slot_ids = np.vstack(np.where(rack_1_state.fillable_slots > 0)).T
    place_ind = fillable_slot_ids[[np.random.choice(np.arange(len(fillable_slot_ids)))]].flatten()
    obj_init_homomat = tube_homomat_gl
    obj_tgt_homomat = rack_1.get_slot_homomat(place_ind)
    goal_place_offset_dis = .04
    obj_tgt_homomat[:3, 3] = obj_tgt_homomat[:3, 3] + rack_1.get_homomat()[:3, 2] * goal_place_offset_dis

    # planner = PickPlacePlanner(ym_rbt)
    grasp_info_list = fs.load_json(TubeType.WHITE_CAP_TUBE.grasps_path)
    rbt_init_pos = ym_rbt.get_jnt_values(hnd_name)

    ym_rbt.fk(hnd_name, rbt_init_pos)
    st = time.time()
    common_grasp_iter = pppc.CommonGraspIteratorGenerator(robot_s=ym_rbt)

    # grasp_iter = common_grasp_iter.gen_iterator(
    #     grasp_info_list=grasp_info_list,
    #     hand_name=hnd_name,
    #     obj_init_homomat=obj_init_homomat.copy(),
    #     sym_obj_tgt_homomats=TubeType.WHITE_CAP_TUBE.get_symmetry_homomat(3, obj_tgt_homomat),
    #     pick_approach_pos_offset=-rack_1.get_homomat()[:3, 2] * .16,
    #     place_approach_pos_offset=-rack_1.get_homomat()[:3, 2] * (.16 - goal_place_offset_dis),
    #     obstacle_list=obs_tube_collection.cm_list,
    #     toggle_debug=True)
    divided_part = 1
    chunk_len = int(len(grasp_info_list)/divided_part)
    grasp_iter_chunks = [(common_grasp_iter.gen_iterator(
        grasp_info_list=grasp_info_list_chunk,
        hand_name=hnd_name,
        obj_init_homomat=obj_init_homomat.copy(),
        sym_obj_tgt_homomats=TubeType.WHITE_CAP_TUBE.get_symmetry_homomat(3, obj_tgt_homomat),
        pick_approach_pos_offset=-rack_1.get_homomat()[:3, 2] * .16,
        place_approach_pos_offset=-rack_1.get_homomat()[:3, 2] * (.16 - goal_place_offset_dis),
        obstacle_list=obs_tube_collection.cm_list,
        toggle_debug=True), _c * chunk_len) for _c, grasp_info_list_chunk in
        enumerate([grasp_info_list[x:x + chunk_len] for x in range(0, len(grasp_info_list), chunk_len)])]



    result = []
    with Pool(10) as executor:
        res = [executor.map(cal_feasible_common_grasps, grasp_iter_chunks)]
        # for _, feasible_list in
        # result += feasible_list

    # grasp_iter.print_error_info()
    print(f"Time of running is {time.time() - st}")
    print(res)
    # print(obj_init_homomat)
    #
    # if len(feasible_list) < 1:
    #     pass
    #     show_animation(info_type="cd_error")
    #     base.run()
