import numpy as np
import huri.core.file_sys as fs
import torch
from huri.learning.network.d3qn import DuelingDQN
from huri.learning.env.arrangement_planning_rack.env import RackArrangementEnv, RackStatePlot
from huri.learning.method.DQN.eval.mcts import MCTS, MCTS_Continue
from huri.examples.task_planning.a_star import TubePuzzle


def refine_path(path, stride=2, max_iter=20):
    iter_cnt = 0
    cnt = 0
    refined_paths = []
    while iter_cnt < max_iter:
        iter_cnt += 1
        tmp_path = path.copy()
        tmp_path_a = tmp_path[:cnt]
        tmp_path_b = tmp_path[cnt:]
        tmp_refined_paths = []
        for j in range(0, len(tmp_path_b) - stride - cnt, stride):
            goal_pattern, init_state = tmp_path_b[j + stride + cnt], tmp_path_b[0]
            tp = TubePuzzle(init_state.copy())
            tp.goalpattern = goal_pattern.copy()
            is_finished, refined_tmp_path = tp.atarSearch(max_iter_cnt=50)
            if is_finished and len(refined_tmp_path) < stride + 1:
                tmp_refined_paths.append([_.grid for _ in refined_tmp_path] + tmp_path_b[j + stride + 1:])
        if len(tmp_refined_paths) > 0:
            refined_paths.append(tmp_path_a + tmp_refined_paths[-1])
        cnt += stride
    return refined_paths[np.argmin([len(_) for _ in refined_paths])]


class DQNSolver():
    def __init__(self,
                 model_path=fs.workdir_learning / "run" / f"dqn_2021_12_01_18_05_30" / "model" / "model_4130000-4140000.pth",
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 num_tube_classes=3,
                 rack_size=(5, 10),
                 seed=777):
        # set up environment
        action_space_dim = np.prod(rack_size) ** 2
        observation_space_dim = (1, *rack_size)
        self.env = RackArrangementEnv(rack_size=rack_size,
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
              infeasible_action_pair=(),
              toggle_result=False):
        state = self.env.reset_state_goal(current_state, goal_pattern)
        root, paths = MCTS(state=state,
                           env=self.env,
                           net=self.net,
                           iter_num=300,
                           lr=.95)
        self.root = root
        if len(paths) > 0:
            shortest_path = paths[np.argmin([len(_) for _ in paths])][::-1]
            # refined_path = refine_path([_.state for _ in shortest_path], stride=max(int(len(shortest_path) / 5), 2), )
            if toggle_result:
                print("AA")
                print(refined_path)
                drawer = RackStatePlot(goal_pattern)
                drawer.plot_states(refined_path, row=22)
            return [_.state for _ in shortest_path]
        else:
            print("Cannot find out the path")
            return []


if __name__ == "__main__":
    goal_pattern = np.array([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                             [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                             [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                             [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                             [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]])
    solver = DQNSolver()

    path = solver.solve(current_state=np.array([[1, 0, 0, 3, 0, 2, 0, 0, 2, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [2, 0, 3, 0, 2, 2, 0, 0, 3, 3],
                                                [0, 1, 0, 0, 0, 0, 0, 2, 0, 0],
                                                [0, 0, 1, 2, 0, 2, 0, 0, 3, 1]]),
                        goal_pattern=goal_pattern,
                        toggle_result=False)

    from huri.learning.env.arrangement_planning_rack.env import RackArrangementEnv, RackStatePlot

    drawer = RackStatePlot(goal_pattern)
    # drawer.plot_states(p, row=22)
    drawer.plot_states(path, row=22)
    drawer.plot_states(path2, row=22)
    print(path)
