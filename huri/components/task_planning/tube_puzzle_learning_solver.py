import time
import copy
import torch
import numpy as np
import os
import huri.core.file_sys as fs
from huri.learning.network.d3qn import DuelingDQN
from huri.learning.method.DQN.eval.mcts import MCTS, MCTS_Continue
from huri.learning.env.rack_v2.env import RackArrangementEnv
import huri.learning.env.rack_v3.env as env3
from huri.learning.env.rack_v3.utils import get_satisfied_constraints
import huri.learning.method.APEX_DQN.env3_fixed_goal.dqn_model_cnn as nncnn3
from huri.learning.env.arrangement_planning_two_rack.env import TwoRackArrangementEnv
from huri.examples.task_planning.a_star import TubePuzzle
from huri.learning.A_start_teacher.A_star_teacher import rm_ras_actions
from huri.learning.method.APEX_DQN.distributed_fixed_goal.env import create_fixe_env, RackArrangementEnv, RackStatePlot, \
    RackState
from huri.learning.method.APEX_DQN.distributed_fixed_goal.network import DDQN2 as DDQN
from huri.learning.method.APEX_DQN.distributed_fixed_goal.pipeline import dqn_select_action, itertools, cv2
from huri.learning.A_start_teacher.A_star_teacher import rm_ras_actions_recur, rm_ras_actions_recur2, \
    rm_ras_actions_recur3_6, a_star_solve, nx


def action_between_states(s_current, s_next):
    move = s_next - s_current
    move_to_idx = np.where(move > 0)
    move_from_idx = np.where(move < 0)
    rack_size = s_current.shape
    pick_id_int = move_from_idx[0] * rack_size[1] + move_from_idx[1]
    place_id_int = move_to_idx[0] * rack_size[1] + move_to_idx[1]
    action_ids = place_id_int * np.prod(rack_size) + pick_id_int
    return action_ids.tolist()


def get_suspicious_action_id(path, goal_pattern):
    path_len = len(path)
    suspicious_id = []
    for state_id in range(0, path_len - 1):
        s_i, s_ii = path[state_id], path[state_id + 1]
        move_map = s_ii - s_i
        move_to_idx = np.where(move_map > 0)
        move_from_idx = np.where(move_map < 0)
        is_move_to_pattern = goal_pattern[move_to_idx] == s_ii[move_to_idx]
        is_move_in_pattern = goal_pattern[move_to_idx] == goal_pattern[move_from_idx] and goal_pattern[move_to_idx] == \
                             s_ii[move_to_idx]
        if (not is_move_to_pattern) or is_move_in_pattern:
            suspicious_id.append(state_id + 1)
    return suspicious_id


def refine_immediate_redundant_action(path, goal_pattern, infeasible_dict):
    is_continue_iter = True
    path_r = copy.deepcopy(path)
    while is_continue_iter:
        is_continue_iter = False
        check_list = get_suspicious_action_id(path_r, goal_pattern)
        if len(check_list) == 0:
            break
        while len(check_list) > 0:
            check_id = check_list[0]
            if len(path_r) < 3:
                break
            if check_id >= len(path_r) - 1:  # at the end
                if np.array_equal(path_r[check_id - 1], path_r[check_id]):
                    path_r.pop(check_id)
                check_list.pop(0)
                continue
            s_0, s_1, s_2 = path_r[check_id - 1], path_r[check_id], path_r[check_id + 1]
            move_map = s_2 - s_0
            move_ids = np.vstack(np.where(move_map != 0)).T
            if len(move_ids) <= 2:
                if len(move_ids) == 2:
                    if action_between_states(s_0, s_2)[0] in infeasible_dict.get(str(s_0), []):
                        check_list.pop(0)
                        continue
                is_continue_iter = True
                # remove the redundant move
                path_r.pop(check_id)
                # update the suspicious action id
                for _ in range(1, len(check_list)):
                    check_list[_] = check_list[_] - 1
            check_list.pop(0)
    return path_r


def refine_eventually_redundant_action(path, goal_pattern, infeasible_dict):
    path_r = path.copy()
    check_list = get_suspicious_action_id(path_r, goal_pattern)
    while len(check_list) > 0:
        check_id = check_list[0] - 1
        end_id = len(path_r) - 1
        s_i, s_i_end = path_r[check_id], path_r[end_id]
        tp = TubePuzzle(s_i.copy())
        tp.goalpattern = s_i_end.copy()
        is_finished, refined_tmp_path = tp.atarSearch(infeasible_dict=infeasible_dict, max_iter_cnt=50)
        if is_finished and len(refined_tmp_path) < end_id - check_id:
            path_r = path_r[:check_id] + [_ for _ in refined_tmp_path]
            break
        check_list.pop(0)
    return path_r


def refine_redundant_action(path, goal_pattern, infeasible_dict):
    # print("Start refine immediate redundant action")
    path_r = refine_immediate_redundant_action(path, goal_pattern, infeasible_dict)
    # print("Start refine eventually redundant action")
    if len(path_r) < 8:
        path_r = refine_eventually_redundant_action(path_r, goal_pattern, infeasible_dict)
    return path_r


model = [None]


class DQNSolver():
    def __init__(self,
                 model_path=fs.workdir_learning / "run" / f"dqn_2022_01_08_21_13_41" / "model" / "model_5076000-5078000.pth",
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 num_tube_classes=3,
                 rack_size=(5, 10),
                 seed=777):
        # set up environment
        self.rack_size = rack_size
        action_space_dim = np.prod(rack_size) ** 2
        observation_space_dim = (1, *rack_size)
        self.env = self._init_env(num_tube_classes, seed, action_space_dim, observation_space_dim)
        self.net = self._init_net(model_path, observation_space_dim, action_space_dim, device)
        self.root = None

    def _init_env(self, num_tube_classes, seed, action_space_dim, observation_space_dim):
        return RackArrangementEnv(rack_size=self.rack_size,
                                  num_classes=num_tube_classes,
                                  observation_space_dim=observation_space_dim,
                                  action_space_dim=action_space_dim,
                                  is_curriculum_lr=True,
                                  is_goalpattern_fixed=True,
                                  seed=seed, )

    def _init_net(self, model_path, observation_space_dim, action_space_dim, device):
        # load trained model
        if model[0] is None:
            net = DuelingDQN(obs_dim=observation_space_dim, action_dim=action_space_dim).to(device)
            model[0] = net
        else:
            net = model[0]
        net.load_state_dict(torch.load(model_path))
        return net

    def solve(self,
              current_state: np.ndarray,
              goal_pattern: np.ndarray,
              infeasible_dict=None,
              infeasible_local_pattern=None,
              toggle_result=False,
              iter_num=300):
        # print("--- Start Task Planning ---")
        if infeasible_dict is None:
            infeasible_dict = {}
        if infeasible_local_pattern is None:
            infeasible_local_pattern = {}
        state = self.env.reset_state_goal(current_state, goal_pattern)
        root, paths = MCTS(state=state,
                           env=self.env,
                           net=self.net,
                           iter_num=iter_num,
                           lr=.95,
                           infeasible_dict=infeasible_dict,
                           infeasible_local_pattern=infeasible_local_pattern,
                           is_debug=toggle_result)
        self.root = root
        if len(paths) > 0:
            shortest_path = paths[np.argmin([len(_) for _ in paths])][::-1]
            # refined_path = refine_path([_.state for _ in shortest_path], stride=max(int(len(shortest_path) / 5), 2), )
            shortest_path_np = [_.state for _ in shortest_path]
            # print("---- Start Refining Path ---")
            # r_shortest_path_np = refine_redundant_action(shortest_path_np, goal_pattern, infeasible_dict)
            fs.dump_pickle(shortest_path_np, "rm_state_debug", reminder=False)
            r_shortest_path_np = rm_ras_actions(shortest_path_np, h=1, infeasible_dict=infeasible_dict,
                                                action_between_states_func=self.env.action_between_states)
            r_shortest_path_np = rm_ras_actions(r_shortest_path_np, h=8, infeasible_dict=infeasible_dict,
                                                goal_pattern=goal_pattern,
                                                action_between_states_func=self.env.action_between_states)

            if toggle_result:
                print("AA")
                print(r_shortest_path_np)
                drawer = RackStatePlot(goal_pattern)
                drawer.plot_states(shortest_path_np, row=22)
            fs.dump_pickle([shortest_path_np, r_shortest_path_np, infeasible_dict], path='debug_path', reminder=False)
            return r_shortest_path_np
        else:
            # print("Cannot find out the path")
            return []

    def action_between_states(self, s_current, s_next, remove_action_group=False):
        move = s_next - s_current
        move_to_idx = np.where(move > 0)
        move_from_idx = np.where(move < 0)

        def _to_action(place_id, pick_id, rack_size, remove_action_group=False):
            if not remove_action_group:
                place_id_int = place_id[0] * rack_size[1] + place_id[1]
            else:
                place_id_int = np.arange(0, np.prod(rack_size))
            pick_id_int = pick_id[0] * rack_size[1] + pick_id[1]
            return place_id_int * np.prod(rack_size) + pick_id_int

        action_ids = _to_action(move_to_idx, move_from_idx, s_current.shape, remove_action_group)
        return action_ids.tolist()


class DQNSolverTwoRack():
    def __init__(self,
                 model_path=fs.workdir_learning / "run" / f"dqn_2022_05_08_21_54_31" / "model" / "model5.pth",
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 num_tube_classes=3,
                 rack_size_1=(5, 10),
                 rack_size_2=(5, 10),
                 seed=777):
        # set up environment
        two_rack_size = (rack_size_1[0] + rack_size_2[0] + 1,
                         max(rack_size_1[1], rack_size_2[1]))
        action_space_dim = np.prod(two_rack_size) ** 2
        observation_space_dim = (1, *two_rack_size)
        self.env = TwoRackArrangementEnv(from_rack_size=rack_size_1,
                                         to_rack_size=rack_size_2,
                                         num_classes=num_tube_classes,
                                         observation_space_dim=observation_space_dim,
                                         action_space_dim=action_space_dim,
                                         is_curriculum_lr=True,
                                         is_goalpattern_fixed=True,
                                         seed=seed,
                                         is_evl_mode=True)

        # load trained model
        self.net = DuelingDQN(obs_dim=observation_space_dim, action_dim=action_space_dim).to(device)
        self.net.load_state_dict(torch.load(model_path))
        self.root = None

    def solve(self,
              current_state: np.ndarray,
              goal_pattern: np.ndarray,
              infeasible_dict=None,
              toggle_result=False):
        # print("--- Start Task Planning ---")
        if infeasible_dict is None:
            infeasible_dict = {}
        state = self.env.reset_state_goal(current_state, goal_pattern)
        root, paths = MCTS(state=state,
                           env=self.env,
                           net=self.net,
                           iter_num=100,
                           lr=.95,
                           infeasible_dict=infeasible_dict,
                           is_debug=toggle_result)
        self.root = root
        if len(paths) > 0:
            shortest_path = paths[np.argmin([len(_) for _ in paths])][::-1]
            # refined_path = refine_path([_.state for _ in shortest_path], stride=max(int(len(shortest_path) / 5), 2), )
            shortest_path_np = [_.state for _ in shortest_path]
            # print("---- Start Refining Path ---")
            r_shortest_path_np = refine_redundant_action(shortest_path_np, goal_pattern, infeasible_dict)
            if toggle_result:
                print("AA")
                print(r_shortest_path_np)
                drawer = RackStatePlot(goal_pattern)
                drawer.plot_states(shortest_path_np, row=22)
            return r_shortest_path_np
        else:
            # print("Cannot find out the path")
            return []

    def action_between_states(self, s_current, s_next, remove_action_group=False):
        move = s_next - s_current
        move_to_idx = np.where(move > 0)
        move_from_idx = np.where(move < 0)

        def _to_action(place_id, pick_id, rack_size, remove_action_group=False):
            if not remove_action_group:
                place_id_int = place_id[0] * rack_size[1] + place_id[1]
            else:
                place_id_int = np.arange(0, np.prod(rack_size))
            pick_id_int = pick_id[0] * rack_size[1] + pick_id[1]
            return place_id_int * np.prod(rack_size) + pick_id_int

        action_ids = _to_action(move_to_idx, move_from_idx, s_current.shape, remove_action_group)
        return action_ids.tolist()


class DQNSolver2(DQNSolver):
    def _init_env(self, num_tube_classes, seed, action_space_dim, observation_space_dim):
        observation_space_dim, action_space_dim = env3.RackState.get_obs_act_dim_by_size(self.rack_size)
        return env3.RackArrangementEnv(rack_size=self.rack_size,
                                       num_classes=num_tube_classes,
                                       observation_space_dim=observation_space_dim,
                                       action_space_dim=action_space_dim,
                                       is_curriculum_lr=True,
                                       is_goalpattern_fixed=False,
                                       seed=seed, )

    def _init_net(self, model_path, observation_space_dim, action_space_dim, device):
        # load trained model
        if model[0] is None:
            observation_space_dim, action_space_dim = env3.RackState.get_obs_act_dim_by_size(self.rack_size)
            net = nncnn3.DuelingDQNCNN(obs_dim=observation_space_dim, action_dim=action_space_dim, num_classes=2).to(
                device)
            model[0] = net
        else:
            net = model[0]
        net.load_state_dict(torch.load(model_path))
        return net

    def solve(self,
              current_state: np.ndarray,
              goal_pattern: np.ndarray,
              infeasible_dict=None,
              infeasible_local_pattern=None,
              toggle_result=False,
              iter_num=100):
        # print("--- Start Task Planning ---")
        if infeasible_dict is None:
            infeasible_dict = {}
        if infeasible_local_pattern is None:
            infeasible_local_pattern = {}

        state = self.env.reset_state_goal(current_state, goal_pattern)
        root, paths = MCTS(state=state,
                           env=self.env,
                           net=self.net,
                           iter_num=iter_num,
                           lr=.95,
                           infeasible_dict=infeasible_dict,
                           infeasible_local_pattern=infeasible_local_pattern,
                           is_debug=toggle_result)
        self.root = root
        if len(paths) > 0:
            shortest_path = paths[np.argmin([len(_) for _ in paths])][::-1]
            # refined_path = refine_path([_.state for _ in shortest_path], stride=max(int(len(shortest_path) / 5), 2), )
            shortest_path_np = [_.state for _ in shortest_path]
            # print("---- Start Refining Path ---")
            # r_shortest_path_np = refine_redundant_action(shortest_path_np, goal_pattern, infeasible_dict)
            fs.dump_pickle(shortest_path_np, "rm_state_debug", reminder=False)
            r_shortest_path_np = rm_ras_actions(shortest_path_np, h=1, infeasible_dict=infeasible_dict,
                                                action_between_states_func=self.env.action_between_states)
            r_shortest_path_np = rm_ras_actions(r_shortest_path_np, h=8, infeasible_dict=infeasible_dict,
                                                goal_pattern=goal_pattern,
                                                action_between_states_func=self.env.action_between_states)

            if toggle_result:
                print("AA")
                print(r_shortest_path_np)
                drawer = env3.RackStatePlot(goal_pattern)
                fig = drawer.plot_states(shortest_path_np, row=14).get_img()
                import cv2
                cv2.imshow('test', fig)
                cv2.waitKey(0)
            fs.dump_pickle([shortest_path_np, r_shortest_path_np, infeasible_dict], path='debug_path', reminder=False)
            return r_shortest_path_np
        else:
            # print("Cannot find out the path")
            return []

    def action_between_states(self, s_current, s_next, remove_action_group=False):
        action_ids = self.env.action_between_states(s_current, s_next, toggle_strict_mode=True)
        return [action_ids]


model_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), "models")


class SpecialistModels(object):
    model = {
        0: os.path.join(model_directory, "model0.chkpt"),
        1: os.path.join(model_directory, "model1.chkpt"),
        2: os.path.join(model_directory, "model2.chkpt"),
        3: os.path.join(model_directory, "model3.chkpt"),
    }
    model_params = {
        3: os.path.join(model_directory, "params3.yaml"),
    }
    goal = {
        0: np.array([[1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ],
                     [1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ],
                     [1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ],
                     [1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ],
                     [1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ]]),
        1: np.array([[1, 1, 1, 1, 0, 0, 2, 2, 2, 2],
                     [1, 1, 1, 1, 0, 0, 2, 2, 2, 2],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [3, 3, 3, 3, 0, 0, 4, 4, 4, 4],
                     [3, 3, 3, 3, 0, 0, 4, 4, 4, 4]]),
        2: np.array([[1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
                     [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
                     [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
                     [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
                     [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ]]),
        3: np.array([[1, 0, 2, 0, 3],
                     [1, 0, 2, 0, 3],
                     [1, 0, 2, 0, 3]], ),
    }


def get_3x3_local_state(state, center_id):
    return np.pad(state, 1)[center_id[0]:center_id[0] + 3,
           center_id[1]:center_id[1] + 3].copy()


class AStarSolver(object):
    def __init__(self, goal_pattern):
        self.goal_pattern = goal_pattern

    def solve(self, init_state: np.ndarray,
              infeasible_actions=None,
              max_iter=300,
              toggle_show=False, ):
        path = a_star_solve(init_state,
                            self.goal_pattern,
                            max_iter_cnt=max_iter,
                            infeasible_dict=infeasible_actions)
        return path


class D3QNSolver():
    def __init__(self,
                 goal_pattern=None,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 seed=777,
                 toggle_init_net=True, ):
        # set up environment
        self.possible_goals = list(SpecialistModels.goal.values())

        agent_id = next((i for i, arr in enumerate(self.possible_goals) if np.array_equal(arr, goal_pattern)), None)
        if agent_id is None:
            raise Exception("The goal pattern has not been solved, please train it first")

        self.device = device
        self.goal_pattern = SpecialistModels.goal[agent_id]
        self._env = self._init_env(self.goal_pattern)
        self.seed = seed
        self._history_states = []
        if toggle_init_net:
            self._net = self._init_net(SpecialistModels.model[agent_id],
                                       SpecialistModels.model_params.get(agent_id, None))
            # warmup <- load parameters. The next solve will be faster
            print("Init Task planner")
            self.random_solve(max_iter=1)
            print("Init Task planner Finish")
        else:
            self._net = False

    def _init_env(self, goal_pattern):
        rack_size = goal_pattern.shape
        num_class = len(np.unique(goal_pattern[goal_pattern > 0]))
        env = create_fixe_env(rack_sz=rack_size,
                              goal_pattern=goal_pattern,
                              num_tube_class=num_class,
                              num_history=1,
                              seed=np.random.randint(0, 1000000),
                              toggle_curriculum=False)
        return env

    def _init_net(self, model_path: str, model_params_path=None):
        # load trained model
        assert isinstance(self._env,
                          RackArrangementEnv), f'The env type should be RackArrangementEnv instead of {type(self._env)}'
        input_shape = self._env.observation_space_dim
        num_actions = self._env.action_space_dim
        if model_params_path is None:
            net = DDQN(input_shape,
                       num_actions,
                       num_category=self._env.num_classes,
                       num_filters=48,
                       num_res_block=6,
                       num_fc_units=128,
                       num_out_cnn_layers=30
                       )
        else:
            params = fs.load_yaml(model_params_path)
            net = DDQN(input_shape,
                       num_actions,
                       num_category=self._env.num_classes,
                       num_filters=params['ddqn']['num_filters'],
                       num_res_block=params['ddqn']['num_res_block'],
                       num_fc_units=params['ddqn']['num_fc_units'],
                       num_out_cnn_layers=params['ddqn']['num_out_cnn_layers']
                       )
        net.load_state_dict(torch.load(model_path)['dqn_state_dict'])
        net.to(self.device)
        return net

    def random_solve(self,
                     delete_constraints: list = None,
                     condition_set: np.ndarray = None,
                     toggle_show: bool = False,
                     max_iter=500):
        state = self._env.reset()
        return self._solve(state,
                           delete_constraints,
                           condition_set=condition_set,
                           max_iter=max_iter, toggle_show=toggle_show)

    def _solve(self,
               state,
               delete_constraints=None,
               condition_set=None,
               max_iter=300,
               infeasible_action_dict=None,
               toggle_Astar_refiner=True,
               toggle_show=False):
        if delete_constraints is None:
            delete_constraints = []
        if infeasible_action_dict is None:
            infeasible_action_dict = {}
        state = RackState(state)
        st_list = [state]
        rew_list = []
        is_path_found = False
        a = time.time()
        for t in itertools.count(1):
            if condition_set is not None:
                feasible_action = state.get_feasible_action_condition_set(condition_set)
            else:
                feasible_action = state.feasible_action_set
            delete_constraints_tmp = delete_constraints + infeasible_action_dict.get(str(state), [])
            action, dqn_action_values = dqn_select_action(env=self._env,
                                                          # feasible_action_set=t_state.feasible_action_set,
                                                          feasible_action_set=np.setdiff1d(feasible_action,
                                                                                           delete_constraints_tmp),
                                                          state=state,
                                                          dqn=self._net,
                                                          device=self.device,
                                                          toggle_no_repeat=True,
                                                          toggle_return_action_value=True, )
            if action is None or action < 0:
                next_state, reward, done = None, -10, True
            else:
                next_state, reward, done, _ = self._env.step(action, toggle_debug=False)
            state = next_state  # state = next_state
            st_list.append(np.array(state))
            rew_list.append(reward)
            if done:
                if reward > 0:
                    # print("Solution Found")
                    is_path_found = True
                break
            if t >= max_iter:
                print("Exceeds maximum iter")
                break
        if not is_path_found:
            if toggle_show:
                rsp = RackStatePlot(self.goal_pattern)
                plot = rsp.plot_states(st_list,
                                       row=8,
                                       reward_list=rew_list).get_img()
                cv2.imshow(
                    f"Failed to find the path.",
                    plot)
                cv2.waitKey(0)
            return [], []
        raw_path = [np.array(_) for _ in st_list]
        path_len_1 = len(raw_path)
        # A* refiner
        if toggle_Astar_refiner:
            path = rm_ras_actions_recur3_6(raw_path,
                                           h=10,
                                           goal_pattern=self.goal_pattern,
                                           max_refine_num=50,
                                           condition_set=condition_set,
                                           infeasible_dict=infeasible_action_dict,
                                           infeasible_set=delete_constraints,
                                           )[0]
            path_len_2 = len(path)
            # print(f"Saved path length is {path_len_1 - path_len_2}: raw path: {path_len_1}, refined path: {path_len_2}")
        else:
            path = raw_path
        # --
        b = time.time()
        if toggle_show:
            rsp = RackStatePlot(self.goal_pattern)
            # plot = rsp.plot_states(st_list,
            #                        row=8,
            # reward_list=rew_list).get_img()
            plot = rsp.plot_states(st_list,
                                   row=6, img_scale=10, toggle_fill=True)
            plot.save_fig("bg3.png", dpi=300)
            plot = rsp.plot_states(path,
                                   row=8, img_scale=10, toggle_fill=True)
            plot.save_fig("refined_path.jpg", dpi=300)
            cv2.imshow(
                f"random solved action sequence. Time consumption is {b - a:.3f}. Saved path length is {path_len_1 - path_len_2}",
                plot.get_img())
            cv2.waitKey(0)
        return raw_path, path

    def solve(self,
              init_state: np.ndarray,
              condition_set=None,
              infeasible_action_dict=None,
              infeasible_action=None,
              max_iter=300,
              toggle_Astar_refiner=True,
              toggle_show=False,
              ):
        # print("--- Start Task Planning ---")
        # repeat actions:
        if infeasible_action_dict is None:
            infeasible_action_dict = {}
        if infeasible_action is None:
            infeasible_action = []
        repeat_acts = []
        for _ in self._history_states:
            act = self._env.action_between_states_constraint_free(init_state, _)
            if act is not None:
                repeat_acts.append(act)

        # init_state
        # state_str = str(init_state)
        state = self._env.reset_state(init_state)
        # delete_constraints = repeat_acts + infeasible_action_dict.get(state_str, [])
        delete_constraints = repeat_acts + infeasible_action
        # for state_str in infeasible_action_dict.keys():
        #     infeasible_action_dict[state_str] = np.union1d(infeasible_action_dict.get(state_str, []), repeat_acts).tolist()
        raw_solution, solution = self._solve(state,
                                             delete_constraints=delete_constraints,
                                             condition_set=condition_set,
                                             max_iter=max_iter,
                                             toggle_Astar_refiner=toggle_Astar_refiner,
                                             infeasible_action_dict=infeasible_action_dict,
                                             toggle_show=toggle_show)
        return raw_solution, solution

    def cal_constraints(self,
                        state,
                        pick_slot_id,
                        place_slot_id,
                        is_pick_feasible: bool,
                        is_place_feasible: bool,
                        condition_set,
                        infeasible_action_dict: dict):

        if not is_pick_feasible:
            pick_local_region = get_3x3_local_state(state, pick_slot_id)
            satisfied_cons_mask = get_satisfied_constraints(pick_local_region)
            condition_set[tuple(pick_slot_id)] = np.maximum(condition_set[tuple(pick_slot_id)] - satisfied_cons_mask, 0)
            return True
        if is_place_feasible is None:
            return False
        if is_place_feasible is False:
            place_local_region = get_3x3_local_state(state, place_slot_id)
            satisfied_cons_mask = get_satisfied_constraints(place_local_region)
            condition_set[tuple(place_slot_id)] = np.maximum(condition_set[tuple(place_slot_id)] - satisfied_cons_mask,
                                                             0)
            return True
        raise Exception("RRT Failed")
        state_str = str(state)
        if state_str not in infeasible_action_dict:
            infeasible_action_dict[state_str] = [self._env.to_action(self._env.rack_size, pick_slot_id, place_slot_id)]
        else:
            infeasible_action_dict[state_str].append(
                self._env.to_action(self._env.rack_size, pick_slot_id, place_slot_id))
        # 判断是pick slot不行还是place slot不行
        # local_init = get_3x3_local_state(state, pick_slot_id)
        # local_goal = get_3x3_local_state()

        # 如果是pick slot不行，确认是否有grasps，及grasps对应的constraints
        # --- 如果是有grasps，确认grasps constraint，并把当前slot对应的constraint去除
        # --- 如果没有grasps，disable多有和改slot有关的动作
        # if len(init_satisfy_constraints) == 6:
        #     rack_sz = self._env.rack_size
        #     for i in range(0, rack_sz.shape[0]):
        #         for j in range(0, rack_sz.shape[1]):
        #             if (i, j) != pick_slot_id:
        #                 self._env.to_action(rack_sz, pick_slot_id, (i, j))

        # 如果是goal slot不行， 确认是否有grasps，及grasps对应的constraint

        # --- 如果是有grasps，确认grasps constraint，并把当前slot对应的constraint去除

        # --- 如果没有grasps，disable多有和改slot有关的动作
        # if len(goal_satisfy_constraints) == 6:
        #     rack_sz = self._env.rack_size
        #     for i in range(0, rack_sz.shape[0]):
        #         for j in range(0, rack_sz.shape[1]):
        #             if (i, j) != place_slot_id:
        #                 self._env.to_action(rack_sz, (i, j), place_slot_id)

    def is_state_valid(self, state):
        return self._env.is_state_valid(state)

    def get_slot_satisfied_constraints(self, state, coord):
        pick_local_region = get_3x3_local_state(state, coord)
        satisfied_cons_mask = get_satisfied_constraints(pick_local_region)
        return satisfied_cons_mask

    def get_all_infeasible_actions_coord(self, coord):
        rack_sz = self.goal_pattern.shape

        A = np.arange(rack_sz[0])
        B = np.arange(rack_sz[1])
        # Creating meshgrid
        A_grid, B_grid = np.meshgrid(A, B, indexing='ij')

        # Reshaping to get the desired combination
        combination_2d = np.array([A_grid.ravel(), B_grid.ravel()]).T
        combination_2d = combination_2d[~ ((combination_2d[:, 0] == coord[0]) & (combination_2d[:, 1] == coord[1]))]

        for i in range(rack_sz[0]):
            for j in range(rack_sz[1]):
                if i == coord[0] and j == coord[1]:
                    continue
        return self._env.to_action(rack_sz, np.repeat(coord.reshape(1, -1), len(combination_2d), axis=0),
                                   combination_2d).tolist()

    def add_history_state(self, state):
        self._history_states.append(state)

    def action_between_states(self, s_current, s_next, remove_pick_action_group=False):
        move = s_next - s_current
        move_to_idx = np.argwhere(move > 0)
        move_from_idx = np.argwhere(move < 0)
        if remove_pick_action_group:
            move_from_idx = np.dstack(
                np.meshgrid(np.arange(self._env.rack_size[0]), np.arange(self._env.rack_size[1]))).reshape(-1, 2)
        action_ids = self._env.to_action(self._env.rack_size,
                                         np.repeat(move_to_idx, repeats=len(move_from_idx), axis=0),
                                         move_from_idx)
        return action_ids.tolist()


if __name__ == "__main__":
    from numpy import array

    # goal = np.array([[1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ],
    #                  [1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ],
    #                  [1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ],
    #                  [1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ],
    #                  [1, 1, 0, 0, 2, 2, 0, 0, 3, 3, ]])
    goal = np.array([[1, 0, 2, 0, 3],
                     [1, 0, 2, 0, 3],
                     [1, 0, 2, 0, 3]], )
    # goal = np.array([[1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
    #                  [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
    #                  [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
    #                  [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
    #                  [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ]])
    # goal = np.array([[1, 1, 1, 1, 0, 0, 2, 2, 2, 2],
    #                  [1, 1, 1, 1, 0, 0, 2, 2, 2, 2],
    #                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                  [3, 3, 3, 3, 0, 0, 4, 4, 4, 4],
    #                  [3, 3, 3, 3, 0, 0, 4, 4, 4, 4]])
    # goal = np.array([[1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
    #                  [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
    #                  [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
    #                  [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
    #                  [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ]])

    RackStatePlot.LINE_WIDTH = 3
    # RackStatePlot.COLOR["alpha"] = .2
    solver = D3QNSolver(goal)
    state = np.array([[1, 3, 2, 0, 0],
                      [2, 0, 0, 0, 3],
                      [0, 0, 2, 1, 0], ])
    infeasible_action_dict = {}
    conditions = np.ones(([*goal.shape[:2], 6]), dtype=int)

    # infeasible_action_dict = {str(state): [61]}
    #
    pick_id = [1, 0]
    pick_local_region = get_3x3_local_state(state, pick_id)
    satisfied_cons_mask = get_satisfied_constraints(pick_local_region)
    print(satisfied_cons_mask)
    conditions[pick_id[0], pick_id[1]] = np.maximum(conditions[tuple(pick_id)] -
                                                    satisfied_cons_mask, 0)

    # r_path, path = solver.random_solve(toggle_show=True)
    # print(repr(path[0]))

    r_path, path = solver.solve(state,
                                infeasible_action_dict=infeasible_action_dict,
                                condition_set=conditions,
                                toggle_show=True)

    for _ in range(1, len(path)):
        print(f"{repr(path[_ - 1])}:", solver._env.action_between_states(path[_ - 1], path[_]))

    exit(0)
    # np.ones((*goal.shape[:2], 6), dtype=int)

    condition_set = np.random.choice([0, 1], size=(*goal.shape[:2], 6))
    condition_set = np.ones((*goal.shape[:2], 6), dtype=int)
    state = array([[0, 0, 2, 0, 1, 0, 1, 2, 0, 0],
                   [0, 0, 1, 1, 3, 0, 3, 1, 2, 0],
                   [0, 2, 0, 1, 3, 3, 2, 3, 2, 0],
                   [0, 0, 3, 0, 2, 3, 3, 1, 2, 0],
                   [0, 0, 2, 1, 3, 0, 1, 0, 0, 0]])
    solver.solve(init_state=state, condition_set=condition_set, toggle_show=True)
    # solver.random_solve(toggle_show=True, condition_set=condition_set)
    print("Finished")
    exit(0)

    # goal_pattern = np.array([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
    #                          [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
    #                          [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
    #                          [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
    #                          [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]])
    # p = [array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    #      array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    #      array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 2, 0, 0, 0, 0, 0]])]
    # print(refine_eventually_redundant_action(p,
    #                                          goal_pattern,
    #                                          {}))

    # goal_pattern = np.array([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
    #                          [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
    #                          [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
    #                          [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
    #                          [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]])
    # solver = DQNSolver()
    #
    # current_state = np.array([[0, 0, 0, 0, 0, 3, 0, 3, 0, 3],
    #                           [1, 1, 0, 1, 3, 3, 1, 0, 0, 3],
    #                           [1, 0, 0, 1, 3, 1, 3, 3, 0, 1],
    #                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                           [1, 1, 1, 3, 0, 0, 0, 0, 0, 3]])
    #
    # next_state = np.array([[0, 0, 0, 0, 0, 3, 0, 3, 0, 3],
    #                        [1, 1, 0, 0, 3, 3, 1, 0, 0, 3],
    #                        [1, 0, 0, 1, 3, 1, 3, 3, 0, 1],
    #                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                        [1, 1, 1, 3, 0, 0, 0, 0, 1, 3]])
    #
    # print(solver.action_between_states(current_state, next_state, remove_action_group=False))
    # print(action_between_states(current_state, next_state))
    # exit()
    # solver.solve(current_state=current_state,
    #              goal_pattern=goal_pattern,
    #              infeasible_dict={str(current_state): [
    #                  13, 2355, 63, 1955, 1563
    #              ]}, )

    # path_0 = solver.solve(current_state=np.array([[1, 0, 0, 3, 0, 2, 0, 0, 2, 0],
    #                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                               [2, 0, 3, 0, 2, 2, 0, 0, 3, 3],
    #                                               [0, 1, 0, 0, 0, 0, 0, 2, 0, 0],
    #                                               [0, 0, 1, 2, 0, 2, 0, 0, 3, 1]]),
    #                       goal_pattern=goal_pattern,
    #                       infeasible_dict={},
    #                       toggle_result=False)
    #
    # path_1 = solver.solve(current_state=np.array([[1, 0, 0, 3, 0, 2, 0, 0, 2, 0],
    #                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                               [2, 0, 3, 0, 2, 2, 0, 0, 3, 3],
    #                                               [0, 1, 0, 0, 0, 0, 0, 2, 0, 0],
    #                                               [0, 0, 1, 2, 0, 2, 0, 0, 3, 1]]),
    #                       goal_pattern=goal_pattern,
    #                       infeasible_dict={str(path_0[0]): [
    #                           solver.action_between_states(path_0[0], path_0[1])
    #                       ]},
    #                       toggle_result=False)

    # root, path, infeasible_actions = MCTS_Continue(infeasible_state_p_1=path_0[0], infeasible_state_p_2=path_0[1],
    #                                                root=solver.root)
    # shortest_path = path[np.argmin([len(_) for _ in path])][::-1]
    # shortest_path_np = [_.state for _ in shortest_path]
    # r_shortest_path_np = refine_redundant_action(shortest_path_np, goal_pattern)
    # r_shortest_path_np = refine_redundant_action(shortest_path_np, goal_pattern)
    # from huri.learning.env.arrangement_planning_rack.env import RackArrangementEnv,

    # drawer = RackStateVisualizer(goal_pattern)
    # drawer.plot_states(path_0, row=22)
    # drawer.plot_states(path_1, row=22)
    # drawer.plot_states(shortest_path_np, row=22)
    # print(path_0)
