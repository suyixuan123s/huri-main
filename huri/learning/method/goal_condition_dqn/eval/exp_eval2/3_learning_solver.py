import numpy as np
import huri.core.file_sys as fs
# from huri.components.task_planning.tube_puzzle_learning_solver import DQNSolver

# initialize the environment
import torch
from huri.learning.env.rack_v3.env import RackArrangementEnv, RackStatePlot, RackState
from huri.learning.method.APEX_DQN.distributed.network_old import DDQN as DDQN2


def dqn_select_action(env: RackArrangementEnv,
                      feasible_action_set: np.ndarray,
                      state: RackState,
                      dqn: torch.nn.Module,
                      device: str,
                      toggle_no_repeat: bool = True, ):
    if toggle_no_repeat:
        feasible_action_set_tmp = feasible_action_set
        repeat_acts = []
        for _ in env.rack_state_history:
            act = env.action_between_states_constraint_free(state, _)
            if act is not None:
                repeat_acts.append(act)
        feasible_action_set = np.setdiff1d(feasible_action_set, repeat_acts)
        if len(feasible_action_set) == 0:
            feasible_action_set = feasible_action_set_tmp
            # self.early_stop = True
            # # print("Potential ERROR Happens")
            # print("state is", self.env.state)
            # print("goal pattern is", self.env.goal_pattern)
    if len(feasible_action_set) < 1:
        return None
    with torch.no_grad():
        feasible_action_set = torch.as_tensor(feasible_action_set, dtype=torch.int64, device=device)
        # one hot design
        dqn_action_value = dqn(
            torch.as_tensor(
                np.concatenate((state.abs_state(2), env.goal_pattern.abs_state(2)), axis=0),
                dtype=torch.float32,
                device=device).unsqueeze(0), ).detach()
        # dqn_action_value = dqn(
        #     torch.as_tensor(np.stack((state.state, env.goal_pattern.state), axis=0), dtype=torch.float32,
        #                     device=device).unsqueeze(0), ).detach()
        selected_action = feasible_action_set[dqn_action_value.squeeze()[feasible_action_set].argmax()].item()
    return selected_action


from huri.learning.env.rack_v3 import create_env

num_tube_classes = 1
rack_size = (3, 6)
env: RackArrangementEnv = create_env(rack_sz=rack_size,
                                     num_tube_class=num_tube_classes,
                                     seed=7777,
                                     toggle_curriculum=True,
                                     toggle_goal_fixed=False,
                                     scheduler='GoalRackStateScheduler3',
                                     num_history=1)
input_shape = env.observation_space_dim
num_actions = env.action_space_dim
network = DDQN2(input_shape,
                num_actions,
                num_category=2,
                num_filters=10,
                num_res_block=19,
                num_fc_units=128)
network.load_state_dict(
    torch.load(r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed\runs\run_1\data\model_last.chkpt')[
        'dqn_state_dict'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network.to(device)
dqn = network

import itertools

success_path = [0] * 18
total_len = [0] * 18
average_len = [0] * 18
delete_constraints = []
reset_num = 20
for num_of_obj in range(1, 14):
    print(f"Difficult level {num_of_obj}")
    candidate_state_goal = fs.load_pickle(f"data/test_data_{num_of_obj}.pkl")
    success_path[num_of_obj] = 0
    print(" i --- ", num_of_obj)
    for i, (i_s, i_g) in enumerate(candidate_state_goal):
        t_score = 0
        t_state = env.reset_state_goal(i_s, i_g)
        is_state_goal_feasible = True
        for t in itertools.count(1):
            t_action = dqn_select_action(env=env,
                                         # feasible_action_set=t_state.feasible_action_set,
                                         feasible_action_set=np.setdiff1d(t_state.feasible_action_set,
                                                                          delete_constraints),
                                         state=t_state,
                                         dqn=dqn,
                                         device=device,
                                         toggle_no_repeat=True)
            if t_action is None:
                if t < 2:
                    is_state_goal_feasible = False
                break
            if t_action is None or t_action < 0:
                t_next_state, t_reward, t_done = None, -10, True
            else:
                t_next_state, t_reward, t_done, _ = env.step(t_action)
            t_reward = t_reward
            t_state = t_next_state  # state = next_state
            # accumulate rewards
            t_score += t_reward  # reward
            if t_done:
                if t_reward >= 0:
                    break
            if t % reset_num == 0:
                is_state_goal_feasible = False
        if is_state_goal_feasible:
            success_path[num_of_obj] += 1
            total_len[num_of_obj] += t
            average_len[num_of_obj] = total_len[num_of_obj] / success_path[num_of_obj]

fs.dump_pickle([success_path, total_len, average_len], f"./data/learning_test.pkl", reminder=False)
print(success_path, total_len, average_len)
