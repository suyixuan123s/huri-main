import logging
from multiprocessing import Process, Queue
import time
import os
import robot_sim.robots.yumi.yumi as ym
import huri.components.planning.symmetric_pick_and_place_planner_c as pppc
from huri.definitions.tube_def import TubeType
from huri.components.planning.utils import _REQ_PACKET

LOGGING_LEVEL = logging.WARNING
PROCESS_SLEEP_TIME = 0.005


class _CommonGraspSolverSeq(Process):
    def __init__(self, req, res, toggle_debug):
        Process.__init__(self)
        self._req = req
        self._res = res
        self._toggle_debug = toggle_debug

    def run(self) -> None:
        logging.getLogger().setLevel(LOGGING_LEVEL)
        # hot reloading of collision model
        common_grasp_iter = pppc.CommonGraspIteratorGenerator(ym.Yumi(enable_cc=True))

        if self._toggle_debug:
            logging.info("In DEBUG mode. Messages will NOT be sent over socket.")
        while True:
            # st = time.time()
            data_packet = self._req.get()
            # print("TIme of get is: ", time.time() - st)
            if data_packet == "stop":
                break
            res_data = self.cal_feasible_common_grasps(common_grasp_iter, *data_packet)
            self._res.put(res_data)
            time.sleep(PROCESS_SLEEP_TIME)

    def cal_feasible_common_grasps(self,
                                   common_grasp_iter,
                                   chunk_init_id,
                                   grasp_info_list,
                                   hnd_name,
                                   obj_init_homomat,
                                   sym_obj_tgt_homomats,
                                   pick_approach_pos_offset,
                                   place_approach_pos_offset,
                                   obstacle_list,
                                   feasible_ranges_rad,
                                   rack_tf):
        # st = time.time()
        grasp_iter = common_grasp_iter.gen_iterator(
            grasp_info_list=grasp_info_list,
            hand_name=hnd_name,
            obj_init_homomat=obj_init_homomat,
            sym_obj_tgt_homomats=sym_obj_tgt_homomats,
            pick_approach_pos_offset=pick_approach_pos_offset,
            place_approach_pos_offset=place_approach_pos_offset,
            obstacle_list=obstacle_list,
            feasible_ranges_rad=feasible_ranges_rad,
            rack_tf=rack_tf,
            toggle_debug=False, )
        # print("Time consuming in constructing iterator", time.time() - st)
        return_ls = []
        feasible_ls = []
        for gid, _ in enumerate(grasp_iter):
            if _[0]:
                return_ls.append(
                    [chunk_init_id + gid, _[1], _[2], _[3], _[4]]
                    # grasp id, init jnt val, init jaw width, tgt jnt val, tgt jaw width
                )
                feasible_ls.append(chunk_init_id + gid)
        grasp_iter.print_error_info()
        # print(feasible_ls)
        return return_ls


class CommonGraspSolver(object):
    def __init__(self, num_workers=5, toogle_debug=False):
        self._toogle_debug = toogle_debug
        self._num_workers = num_workers
        self._worker_req_list = []
        self._worker_res_list = []
        self._worker_process_list = []
        self.start()

    def start(self):
        self._create_seq_solver()

    def _create_seq_solver(self):
        for _ in range(self._num_workers):
            _wk_req_que = Queue()
            _wk_res_que = Queue()
            _wk_proc = _CommonGraspSolverSeq(_wk_req_que, _wk_res_que, self._toogle_debug)
            self._worker_req_list.append(_wk_req_que)
            self._worker_res_list.append(_wk_res_que)
            self._worker_process_list.append(_wk_proc)
            _wk_proc.start()
        print("All the process started")

    def send_recv_data(self, data_list):
        data_list_len = len(data_list)
        if data_list_len <= self._num_workers:
            # st = time.time()
            recv_data = []
            for i in range(data_list_len):
                _wk_req_que = self._worker_req_list[i]
                _wk_req_que.put(data_list[i])
            # print(f"Time of preparing is {time.time() - st}")
            for i in range(data_list_len):
                _wk_res_que = self._worker_res_list[i]
                recv_data += _wk_res_que.get()
            return recv_data
        else:
            raise Exception("Number of worker should be larger than the data length")

    def stop(self):
        '''Stops subprocess for ethernet communication. Allows program to exit gracefully.
        '''
        for _wk_que in self._worker_req_list:
            _wk_que.put("stop")
        for _wk_proc in self._worker_process_list:
            _wk_proc.terminate()

    def __del__(self):
        self.stop()


class CommonGraspSolverDebug(object):
    def __init__(self, num_workers=5, toogle_debug=False):
        self._toogle_debug = toogle_debug
        self._num_workers = num_workers

    def send_recv_data(self, data_list):
        feasible_grasps = []
        common_grasp_iter = pppc.CommonGraspIteratorGenerator(ym.Yumi(enable_cc=True), )
        for d in data_list:
            feasible_grasps += self.cal_feasible_common_grasps(common_grasp_iter, *d)
        print(f"number of feasible grasps len {feasible_grasps}")

        if len(feasible_grasps) > 0:
            return feasible_grasps

        from huri.components.debug.inspect import save_rbt_info, show_animation, save_error_info, IKError, \
            RbtCollisionError
        show_animation("rbt_hnd_error")

    def cal_feasible_common_grasps(self,
                                   common_grasp_iter,
                                   chunk_init_id,
                                   grasp_info_list,
                                   hnd_name,
                                   obj_init_homomat,
                                   sym_obj_tgt_homomats,
                                   pick_approach_pos_offset,
                                   place_approach_pos_offset,
                                   obstacle_list,
                                   feasible_ranges_rad,
                                   rack_tf):
        # st = time.time()
        grasp_iter = common_grasp_iter.gen_iterator(
            grasp_info_list=grasp_info_list,
            hand_name=hnd_name,
            obj_init_homomat=obj_init_homomat,
            sym_obj_tgt_homomats=sym_obj_tgt_homomats,
            pick_approach_pos_offset=pick_approach_pos_offset,
            place_approach_pos_offset=place_approach_pos_offset,
            obstacle_list=obstacle_list,
            feasible_ranges_rad=feasible_ranges_rad,
            rack_tf=rack_tf,
            toggle_debug=True)
        # print("Time consuming in constructing iterator", time.time() - st)
        return_ls = []
        feasible_ls = []
        for gid, _ in enumerate(grasp_iter):
            if _[0]:
                return_ls.append(
                    [chunk_init_id + gid, _[1], _[2], _[3], _[4]]
                    # grasp id, init jnt val, init jaw width, tgt jnt val, tgt jaw width
                )
                feasible_ls.append(chunk_init_id + gid)
        grasp_iter.print_error_info()
        # print(feasible_ls)
        return return_ls


if __name__ == "__main__":
    from huri.definitions.rack_def import Rack_Hard_Proto, TubeType
    from huri.core.common_import import *
    from huri.core.print_tool import text_pd
    from huri.learning.env.arrangement_planning_rack.utils import get_random_states
    from huri.learning.env.arrangement_planning_rack.env import RackState

    random_seed = 1000
    np.random.seed(random_seed)
    # create the virtual environment
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    number_of_worker = 20
    worker = CommonGraspSolver(num_workers=number_of_worker)
    time.sleep(20)
    # generate the yumi robot and its mesh model
    ym_rbt = ym.Yumi(enable_cc=True)
    ym_rbt.gen_meshmodel().attach_to(base)
    hnd_name = "rgt_hnd"
    num_of_trial = 100
    time_exp = []
    for _ in range(num_of_trial):
        print(f"{_} trial")
        # generate a rack from a prototype
        rack_1 = Rack_Hard_Proto.copy()
        rack_1.set_homomat(rm.homomat_from_posrot(pos=np.array([0.4, -.2, 0])))
        # insert test tube to the rack
        random_state = get_random_states(rack_size=np.array([5, 10]),
                                         goalpattern=np.array([[1, 1, 1, 0, 0, 0, 0, 3, 3, 3],
                                                               [1, 1, 1, 0, 0, 0, 0, 3, 3, 3],
                                                               [1, 1, 1, 0, 0, 0, 0, 3, 3, 3],
                                                               [1, 1, 1, 0, 0, 0, 0, 3, 3, 3],
                                                               [1, 1, 1, 0, 0, 0, 0, 3, 3, 3]]),
                                         min_obj_num=2,
                                         obj_num=25,
                                         np_random=np.random)
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
        # base.run()
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

        print("grasp len:", len(grasp_info_list))
        st = time.time()
        chunk_len = int(np.ceil(len(grasp_info_list) / number_of_worker))
        datalist = [_REQ_PACKET(chunk_init_id=_c * chunk_len,
                                grasp_info_list=grasp_info_list_chunk,
                                hnd_name=hnd_name,
                                obj_init_homomat=obj_init_homomat.copy(),
                                sym_obj_tgt_homomats=TubeType.WHITE_CAP_TUBE.get_symmetry_homomat(3, obj_tgt_homomat),
                                pick_approach_pos_offset=-rack_1.get_homomat()[:3, 2] * .16,
                                place_approach_pos_offset=-rack_1.get_homomat()[:3, 2] * (.16 - goal_place_offset_dis),
                                obstacle_list=obs_tube_collection.cm_list)
                    for _c, grasp_info_list_chunk in
                    enumerate([grasp_info_list[x:x + chunk_len] for x in range(0, len(grasp_info_list), chunk_len)])]
        grasps = worker.send_recv_data(datalist)
        ed = time.time()
        time_exp.append(ed - st)
        # print(f"Time of preparing segment is {time.time() - st}")

        # print(grasps)
        # print("FINISHED")
    print(time_exp)
    print(f"Time of running is {sum(time_exp) / len(time_exp)}")
    exit(0)
    # while True:
    #     time.sleep(1)
    # worker.stop()
