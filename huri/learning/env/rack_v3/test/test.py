import unittest
from huri.learning.env.rack_v3.env import RackState, get_random_states, RackArrangementEnv, RackStatePlot
import numpy as np
import networkx as nx
import cv2

np.set_printoptions(threshold=np.inf)

class_num = 10
rack_size = (2, 2)


def random_rack_state() -> RackState:
    # Create an instance of the class to test
    while 1:
        initstate = np.random.choice(class_num, rack_size, p=[.5, *([.5 / (class_num - 1)] * (class_num - 1))])
        if np.sum(initstate) > 0:
            break
    return RackState(initstate)


class RackStateTestCase(unittest.TestCase):

    def test_action_space(self):
        rack_size = (4, 4)
        obs_dim, act_dim = RackState.get_obs_act_dim_by_size(rack_size)
        print(act_dim)

    def test_random_goal_pattern_scheduler1(self):
        rack_size = (3, 6)
        class_num = 2
        obs_dim, act_dim = RackState.get_obs_act_dim_by_size(rack_size)
        self.env = RackArrangementEnv(rack_size=rack_size, num_classes=class_num, observation_space_dim=obs_dim,
                                      action_space_dim=act_dim, is_curriculum_lr=True, is_goalpattern_fixed=False, )
        for i in range(np.prod(rack_size)):
            for i in range(10):
                a = self.env.reset()
                rp = RackStatePlot(goal_pattern=self.env.goal_pattern)
                img = rp.plot_states(rack_states=[a], row=1).get_img()
                cv2.imshow("img", img)
                cv2.moveWindow("img", 300, 300)
                cv2.waitKey(0)
            self.env.scheduler.update_training_level()

    def test_random_goal_pattern_scheduler2(self):
        scheduler_name = 'GoalRackStateScheduler2'

    def test_random_goal_pattern_scheduler3(self):
        scheduler_name = 'GoalRackStateScheduler3'
        rack_size = (3, 6)
        class_num = 2
        obs_dim, act_dim = RackState.get_obs_act_dim_by_size(rack_size)
        self.env = RackArrangementEnv(rack_size=rack_size, num_classes=class_num, observation_space_dim=obs_dim,
                                      action_space_dim=act_dim, is_curriculum_lr=True, is_goalpattern_fixed=False,
                                      scheduler=scheduler_name)
        for i in range(np.prod(rack_size)):
            for j in range(5):
                a = self.env.reset()
                rp = RackStatePlot(goal_pattern=self.env.goal_pattern)
                img = rp.plot_states(rack_states=[a], row=1).get_img()
                cv2.imshow("img", img)
                cv2.moveWindow("img", 300, 300)
                cv2.waitKey(0)
            print(self.env.scheduler.state_level, self.env.scheduler.class_level)
            print("training level:", i + 1)
            self.env.scheduler.set_training_level(i + 1)

    def test_adjacent_mat(self):
        for _ in range(1000):
            rack_state = random_rack_state()
            state = rack_state.state
            state_ft = state.flatten()
            rack_size = state.shape
            adj_mat = rack_state.to_abs_state
            possible_actions = rack_state._cal_possible_actions()
            pick_ids = np.ravel_multi_index(possible_actions[:, 0:2].T, rack_size)
            place_ids = np.ravel_multi_index(possible_actions[:, 2:4].T, rack_size)
            G = nx.Graph()
            G.add_nodes_from(range(len(state_ft)))  # Add nodes
            G.add_weighted_edges_from(np.stack((pick_ids, place_ids, state_ft[pick_ids])).T)  # Add edges
            nx_adj_mat = nx.convert_matrix.to_numpy_array(G)
            self.assertTrue(np.all(adj_mat == nx_adj_mat))

    def test_mat(self):
        state = np.array([[1, 0, 0],
                          [0, 2, 1],
                          [0, 1, 0]])
        rack_state = RackState(state)
        adj_mat = rack_state.to_abs_state
        print(adj_mat)

        # state = np.array([[0, 1, ], [0, 0]])
        # rack_state = RackState(state)
        # adj_mat = rack_state.to_abs_state()
        # print(adj_mat)

    def test_action_id(self):
        class_num = 10
        rack_size = (2, 2)
        obs_dim, act_dim = RackState.get_obs_act_dim_by_size(rack_size)
        self.env = RackArrangementEnv(rack_size=rack_size, num_classes=class_num, observation_space_dim=obs_dim,
                                      action_space_dim=act_dim)
        state1 = np.array([[1, 0, 0],
                           [0, 2, 1],
                           [0, 1, 0]])
        state2 = np.array([[1, 0, 0],
                           [0, 2, 1],
                           [1, 0, 0]])
        print(self.env.action_between_states(state1, state2))

    def test_compress_abs_state(self):
        class_num = 10
        rack_size = (10, 20)
        obs_dim, act_dim = RackState.get_obs_act_dim_by_size(rack_size)
        self.env = RackArrangementEnv(rack_size=rack_size, num_classes=class_num, observation_space_dim=obs_dim,
                                      action_space_dim=act_dim, is_goalpattern_fixed=False)
        state = self.env.reset()

        def compress_matrix(matrix):
            # Compress the matrix by extracting the lower triangle
            compressed = matrix[np.tril_indices(matrix.shape[0])]
            return compressed

        def decompress_matrix(compressed, n):
            # Decompress the matrix by constructing a symmetric matrix
            decompressed = np.zeros((n, n))
            decompressed[np.tril_indices(n)] = compressed
            decompressed = decompressed + decompressed.T - np.diag(np.diag(decompressed))
            return decompressed

        for i in range(10):
            state = self.env.reset()
            # print(state.to_abs_state)
            compress = state.compressed_abs_state
            # print(compress.shape)
            obs_dim = np.prod(rack_size)
            self.assertTrue(compress.shape[0] == int((obs_dim - 1) * obs_dim / 2) + obs_dim)
            self.assertTrue(
                np.all(state.to_abs_state == RackState.decompress_abs_state(compress, state.abs_size[0])))
            # print(np.all(state.to_abs_state == decompress_matrix(compress, state.to_abs_state.shape[0])))

    def test_synthesize_state(self):
        class_num = 2
        rack_size = (5, 10)
        obs_dim, act_dim = RackState.get_obs_act_dim_by_size(rack_size)
        self.env = RackArrangementEnv(rack_size=rack_size, num_classes=class_num, observation_space_dim=obs_dim,
                                      action_space_dim=act_dim, is_goalpattern_fixed=False)
        state = self.env.reset()
        a = state.reflections
        b = self.env.goal_pattern.reflections
        ns, _, _, _ = self.env.step(self.env.sample())
        c = ns.reflections

        from huri.learning.env.rack_v3.env import GoalRackStateScheduler2, RackStatePlot
        import numpy as np
        import cv2
        for i in range(len(a)):
            rp = RackStatePlot(goal_pattern=b[i].state)
            img = rp.plot_states(rack_states=[a[i], c[i]], row=1).get_img()
            cv2.imshow(f"img_{i}", img)
        cv2.waitKey(0)
