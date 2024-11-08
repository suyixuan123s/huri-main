import time
from typing import List, Callable
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from huri.examples.task_planning.a_star import TubePuzzle
import cv2
from huri.learning.env.rack_v3.env import RackStatePlot, RackState, RackArrangementEnv, isdone

N: Callable = np.count_nonzero


def N_C(a: np.ndarray, b: np.ndarray) -> int:
    assert a.shape == b.shape
    non_zeros_ids = np.where(b)
    return N(a[non_zeros_ids] == b[non_zeros_ids])


# it is dangerous to use str to make numpy hashable: https://stackoverflow.com/questions/16589791/most-efficient-property-to-hash-for-numpy-array
# TODO revise it to a safer way to hash numpy array
np_hashable = lambda x: str(x)

drawer = RackStatePlot(np.array([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                                 [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                                 [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                                 [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                                 [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]]))

# def action_between_states(s_current, s_next):
#     move = s_next - s_current
#     move_to_idx = np.where(move > 0)
#     move_from_idx = np.where(move < 0)
#     rack_size = s_current.shape
#     pick_id_int = move_from_idx[0] * rack_size[1] + move_from_idx[1]
#     place_id_int = move_to_idx[0] * rack_size[1] + move_to_idx[1]
#     # print(place_id_int,pick_id_int)
#     # if len(place_id_int) != len(pick_id_int):
#     #     print("!")
#     action_ids = place_id_int * np.prod(rack_size) + pick_id_int
#     return action_ids.tolist()
action_between_states = RackArrangementEnv.action_between_states
action_between_states_condition_set = RackArrangementEnv.action_between_states_condition_set


def _E(state, goal_pattern, number_class):
    # number of elements in the same class and same position
    entropy = np.zeros(len(number_class), dtype=int)
    for i, _ in enumerate(number_class):
        entropy[i] = len(np.where(state[goal_pattern == _] == _)[0])
    return entropy


def E(state, goal_pattern):
    number_class = np.unique(state)
    number_class = number_class[number_class > 0]
    return _E(state, goal_pattern, number_class)


def F(state):
    if not isinstance(state, RackState):
        rs = RackState(state)
    else:
        rs = state
    return np.sum(rs.movable_slots + rs.fillable_slots)


def rm_ras_actions(stats: List[np.ndarray], h: int = 1, goal_pattern=None, infeasible_dict: dict = None,
                   toggle_debug=False, action_between_states_func: callable = None) -> (
        List[int], List[np.ndarray]):
    if infeasible_dict is None:
        infeasible_dict = {}
    if action_between_states_func:
        action_between_states_func = action_between_states
    stats_hashable = np.array([np_hashable(i) for i in stats])
    np_hash_table = {stats_hashable[i]: stats[i] for i in range(len(stats))}
    # create an empty graph
    G = nx.DiGraph()
    # create gamma and omega in the paper
    gamm: List[(np.ndarray, np.ndarray)] = []
    omeg: List[int] = []

    number_class = np.unique(stats[0])
    number_class = number_class[number_class > 0]
    # add node
    for s in stats:
        G.add_node(np_hashable(s))
        # print(_E(s, goal_pattern, number_class))
    # add edge
    for i in np.arange(0, len(stats) - 1):
        for j in np.arange(i + 1, len(stats)):
            d = N(stats[i]) - N_C(stats[i], stats[j])
            if toggle_debug:
                fig = drawer.plot_states([stats[i]], row=1)
                p_1 = Plot(fig=fig).get_img()
                fig = drawer.plot_states([stats[j]], row=1)
                p_2 = Plot(fig=fig).get_img()
                p = np.concatenate((p_1, p_2))
                cv2.imwrite("debug.jpg", p)
                print(f"i is {i}, j is {j}, d is {d}")
            if d == 0:
                # no different between stats[i] and stats[j]
                G.add_edge(stats_hashable[i], stats_hashable[j], weight=0)
            elif d == 1:
                action = action_between_states_func(stats[i], stats[j], toggle_strict_mode=False)
                if action is None or action in infeasible_dict.get(stats_hashable[i], []):
                    continue
                G.add_edge(stats_hashable[i], stats_hashable[j], weight=1)
            elif d > 1:
                # TODO ADD CONSTRAINTS
                if d > (j - i):  # make sure
                    gamm.append([stats[i], stats[j]])
                    omeg.append(j - i)

                if goal_pattern is not None:
                    stats_i_enp = _E(stats[i], goal_pattern=goal_pattern, number_class=number_class)
                    stats_j_enp = _E(stats[j], goal_pattern=goal_pattern, number_class=number_class)
                    enp_diff_i_j = stats_j_enp - stats_i_enp
                    if np.sum(enp_diff_i_j) < j - i and np.all(enp_diff_i_j > 0):
                        # print(j - i, np.sum(enp_diff_i_j))
                        # from huri.components.utils.matlibplot_utils import Plot
                        # fig = drawer.plot_states([stats[i]])
                        # p1 = Plot(fig=fig)
                        # fig = drawer.plot_states([stats[j]])
                        # p2 = Plot(fig=fig)
                        # cv2.imshow("i", p1.get_img())
                        # cv2.imshow("j", p2.get_img())
                        # cv2.waitKey(0)
                        gamm.append([stats[0], stats[j]])
                        omeg.append(j - i)
                        # for zzz, _ in enumerate(stats_i_enp == stats_j_enp):
                        #     if _:
                        #         c = number_class[zzz]
                        #         new_state = stats[j].copy()
                        #         new_state[stats[i] == c] = stats[i][stats[i] == c]
                        #         G.add_node(np_hashable(new_state))
                        # gamm.append([stats[i], new_state])
                        # omeg.append(j - i)

            else:
                raise Exception("Unexpected errors")

                # see the nodes
    # print(gamm)
    # if [stats[0], stats[-1]] not in gamm:
    #     gamm.append([stats[0], stats[-1]])
    #     omeg.append(-1)
    # debug lot nodes
    if toggle_debug:
        pos = nx.spring_layout(G)
        nx.draw(G, pos)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.show()

        # refine edge
    for i in np.arange(len(gamm)):
        if omeg[i] <= h:
            # A start
            a = TubePuzzle(elearray=gamm[i][0], goalpattern=gamm[i][1])
            is_find, path = a.atarSearch(infeasible_dict=infeasible_dict, max_iter_cnt=100)
            if is_find:
                # path: List = []
                # print(omeg[i] - len(path))

                # if omeg[i] - len(path) < -1:
                #     from huri.components.utils.matlibplot_utils import Plot
                #     fig = drawer.plot_states(path)
                #     p = Plot(fig=fig)
                #     cv2.imshow("i", p.get_img())
                #     cv2.waitKey(0)

                for j in range(1, len(path)):
                    node = np_hashable(path[j])
                    if node not in G:
                        np_hash_table[node] = path[j]
                        G.add_node(node)
                    if not G.has_edge(np_hashable(path[j - 1]), np_hashable(node)):
                        G.add_edge(np_hashable(path[j - 1]), np_hashable(node), weight=1)

    if not G.has_edge(stats_hashable[0], stats_hashable[-1]):
        a = TubePuzzle(elearray=stats[0], goalpattern=stats[-1])
        is_find, path = a.atarSearch(infeasible_dict=infeasible_dict, max_iter_cnt=100)
        if is_find:
            for j in range(1, len(path)):
                node = np_hashable(path[j])
                if node not in G:
                    np_hash_table[node] = path[j]
                    G.add_node(node)
                if not G.has_edge(np_hashable(path[j - 1]), np_hashable(node)):
                    G.add_edge(np_hashable(path[j - 1]), np_hashable(node), weight=1)

    r_stats_hashable = np.array(
        nx.shortest_path(G, source=stats_hashable[0], target=stats_hashable[-1], weight='weight'))

    return [np_hash_table[_] for _ in r_stats_hashable]


def rm_ras_actions_recur(stats: List[np.ndarray], h: int = 1, goal_pattern=None, infeasible_dict: dict = None,
                         toggle_debug=False) -> (
        List[int], List[np.ndarray]):
    if infeasible_dict is None:
        infeasible_dict = {}
    stats_hashable = np.array([np_hashable(i) for i in stats])
    np_hash_table = {stats_hashable[i]: stats[i] for i in range(len(stats))}
    # create an empty graph
    G = nx.DiGraph()
    # create gamma and omega in the paper
    gamm: List[(np.ndarray, np.ndarray)] = []
    omeg: List[int] = []

    number_class = np.unique(stats[0])
    number_class = number_class[number_class > 0]
    # add node
    for s in stats:
        G.add_node(np_hashable(s))
        # print(_E(s, goal_pattern, number_class))
    # add edge
    for i in np.arange(0, len(stats) - 1):
        for j in np.arange(i + 1, len(stats)):
            d = N(stats[i]) - N_C(stats[i], stats[j])
            if toggle_debug:
                fig = drawer.plot_states([stats[i]], row=1)
                p_1 = Plot(fig=fig).get_img()
                fig = drawer.plot_states([stats[j]], row=1)
                p_2 = Plot(fig=fig).get_img()
                p = np.concatenate((p_1, p_2))
                cv2.imwrite("debug.jpg", p)
                print(f"i is {i}, j is {j}, d is {d}")
            if d == 0:
                # no different between stats[i] and stats[j]
                G.add_edge(stats_hashable[i], stats_hashable[j], weight=0)
            elif d == 1:
                action = action_between_states(stats[i], stats[j], toggle_strict_mode=False)
                if action is None or action in infeasible_dict.get(stats_hashable[i], []):
                    continue
                G.add_edge(stats_hashable[i], stats_hashable[j], weight=1)
            elif d > 1:
                # TODO ADD CONSTRAINTS
                if d > (j - i):  # make sure
                    gamm.append([stats[i], stats[j]])
                    omeg.append(j - i)

                if goal_pattern is not None:
                    stats_i_enp = _E(stats[i], goal_pattern=goal_pattern, number_class=number_class)
                    stats_j_enp = _E(stats[j], goal_pattern=goal_pattern, number_class=number_class)
                    enp_diff_i_j = stats_j_enp - stats_i_enp
                    if np.sum(enp_diff_i_j) < j - i and np.all(enp_diff_i_j > 0):
                        # print(j - i, np.sum(enp_diff_i_j))
                        # from huri.components.utils.matlibplot_utils import Plot
                        # fig = drawer.plot_states([stats[i]])
                        # p1 = Plot(fig=fig)
                        # fig = drawer.plot_states([stats[j]])
                        # p2 = Plot(fig=fig)
                        # cv2.imshow("i", p1.get_img())
                        # cv2.imshow("j", p2.get_img())
                        # cv2.waitKey(0)
                        gamm.append([stats[0], stats[j]])
                        omeg.append(j - i)
                        # for zzz, _ in enumerate(stats_i_enp == stats_j_enp):
                        #     if _:
                        #         c = number_class[zzz]
                        #         new_state = stats[j].copy()
                        #         new_state[stats[i] == c] = stats[i][stats[i] == c]
                        #         G.add_node(np_hashable(new_state))
                        # gamm.append([stats[i], new_state])
                        # omeg.append(j - i)

            else:
                raise Exception("Unexpected errors")

                # see the nodes
    # print(gamm)
    # if [stats[0], stats[-1]] not in gamm:
    #     gamm.append([stats[0], stats[-1]])
    #     omeg.append(-1)
    # debug lot nodes
    if toggle_debug:
        pos = nx.spring_layout(G)
        nx.draw(G, pos)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.show()

        # refine edge
    for i in np.arange(len(gamm)):
        if omeg[i] <= h:
            # A start
            a = TubePuzzle(elearray=gamm[i][0], goalpattern=gamm[i][1])
            is_find, path = a.atarSearch(infeasible_dict=infeasible_dict, max_iter_cnt=100)
            if is_find:
                # path: List = []
                # print(omeg[i] - len(path))

                # if omeg[i] - len(path) < -1:
                #     from huri.components.utils.matlibplot_utils import Plot
                #     fig = drawer.plot_states(path)
                #     p = Plot(fig=fig)
                #     cv2.imshow("i", p.get_img())
                #     cv2.waitKey(0)

                for j in range(1, len(path)):
                    node = np_hashable(path[j])
                    if node not in G:
                        np_hash_table[node] = path[j]
                        G.add_node(node)
                    if not G.has_edge(np_hashable(path[j - 1]), np_hashable(node)):
                        G.add_edge(np_hashable(path[j - 1]), np_hashable(node), weight=1)

    r_stats_hashable = []
    for ii in range(max(len(stats_hashable) - 2, 0)):
        r_stats_hashable_tmp = np.array(
            nx.shortest_path(G, source=stats_hashable[ii], target=stats_hashable[-1], weight='weight'))
        r_stats_hashable.append([np_hash_table[_] for _ in r_stats_hashable_tmp])
    return r_stats_hashable


def rm_ras_actions_recur2(stats: List[np.ndarray], h: int = 1, goal_pattern=None, infeasible_dict: dict = None,
                          toggle_debug=False) -> (
        List[int], List[np.ndarray]):
    if infeasible_dict is None:
        infeasible_dict = {}
    stats_hashable = np.array([np_hashable(i) for i in stats])
    np_hash_table = {stats_hashable[i]: stats[i] for i in range(len(stats))}
    # create an empty graph
    G = nx.DiGraph()
    # create gamma and omega in the paper
    gamm: List[(np.ndarray, np.ndarray)] = []
    omeg: List[int] = []

    number_class = np.unique(stats[0])
    number_class = number_class[number_class > 0]
    # add node
    for s in stats:
        G.add_node(np_hashable(s))
        # print(_E(s, goal_pattern, number_class))
    # add edge
    for i in np.arange(0, len(stats) - 1):
        for j in np.arange(i + 1, len(stats)):
            d = N(stats[i]) - N_C(stats[i], stats[j])
            if toggle_debug:
                fig = drawer.plot_states([stats[i]], row=1)
                p_1 = Plot(fig=fig).get_img()
                fig = drawer.plot_states([stats[j]], row=1)
                p_2 = Plot(fig=fig).get_img()
                p = np.concatenate((p_1, p_2))
                cv2.imwrite("debug.jpg", p)
                print(f"i is {i}, j is {j}, d is {d}")
            if d == 0:
                # no different between stats[i] and stats[j]
                G.add_edge(stats_hashable[i], stats_hashable[j], weight=0)
            elif d == 1:
                action = action_between_states(stats[i], stats[j], toggle_strict_mode=False)
                if action is None or action in infeasible_dict.get(stats_hashable[i], []):
                    continue
                G.add_edge(stats_hashable[i], stats_hashable[j], weight=1)
            elif d > 1:
                # TODO ADD CONSTRAINTS
                if d > (j - i):  # make sure
                    gamm.append([stats[i], stats[j]])
                    omeg.append(j - i)

                if goal_pattern is not None:
                    stats_i_enp = _E(stats[i], goal_pattern=goal_pattern, number_class=number_class)
                    stats_j_enp = _E(stats[j], goal_pattern=goal_pattern, number_class=number_class)
                    enp_diff_i_j = stats_j_enp - stats_i_enp
                    if np.sum(enp_diff_i_j) < j - i and np.all(enp_diff_i_j > 0):
                        # print(j - i, np.sum(enp_diff_i_j))
                        # from huri.components.utils.matlibplot_utils import Plot
                        # fig = drawer.plot_states([stats[i]])
                        # p1 = Plot(fig=fig)
                        # fig = drawer.plot_states([stats[j]])
                        # p2 = Plot(fig=fig)
                        # cv2.imshow("i", p1.get_img())
                        # cv2.imshow("j", p2.get_img())
                        # cv2.waitKey(0)
                        gamm.append([stats[0], stats[j]])
                        omeg.append(j - i)
                        # for zzz, _ in enumerate(stats_i_enp == stats_j_enp):
                        #     if _:
                        #         c = number_class[zzz]
                        #         new_state = stats[j].copy()
                        #         new_state[stats[i] == c] = stats[i][stats[i] == c]
                        #         G.add_node(np_hashable(new_state))
                        # gamm.append([stats[i], new_state])
                        # omeg.append(j - i)

            else:
                raise Exception("Unexpected errors")

                # see the nodes
    # print(gamm)
    # if [stats[0], stats[-1]] not in gamm:
    #     gamm.append([stats[0], stats[-1]])
    #     omeg.append(-1)
    # debug lot nodes
    if toggle_debug:
        pos = nx.spring_layout(G)
        nx.draw(G, pos)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.show()

        # refine edge
    for i in np.arange(len(gamm)):
        if omeg[i] <= h:
            # A start
            a = TubePuzzle(elearray=gamm[i][0], goalpattern=gamm[i][1])
            is_find, path = a.atarSearch(infeasible_dict=infeasible_dict, max_iter_cnt=100)
            if is_find:
                # path: List = []
                # print(omeg[i] - len(path))

                # if omeg[i] - len(path) < -1:
                #     from huri.components.utils.matlibplot_utils import Plot
                #     fig = drawer.plot_states(path)
                #     p = Plot(fig=fig)
                #     cv2.imshow("i", p.get_img())
                #     cv2.waitKey(0)

                for j in range(1, len(path)):
                    node = np_hashable(path[j])
                    if node not in G:
                        np_hash_table[node] = path[j]
                        G.add_node(node)
                    if not G.has_edge(np_hashable(path[j - 1]), np_hashable(node)):
                        G.add_edge(np_hashable(path[j - 1]), np_hashable(node), weight=1)

    r_stats_hashable = []
    if len(stats_hashable) < 2:
        return r_stats_hashable
    for ii in range(1, max(len(stats_hashable) - 1, 0)):
        r_stats_hashable_tmp = np.array(
            nx.shortest_path(G, source=stats_hashable[0], target=stats_hashable[ii], weight='weight'))
        r_stats_hashable.append([np_hash_table[_] for _ in r_stats_hashable_tmp])
    return r_stats_hashable


def rm_ras_actions_recur3_bk(stats: List[np.ndarray], h: int = 1, goal_pattern=None, infeasible_dict: dict = None,
                             toggle_debug=False) -> (
        List[int], List[np.ndarray]):
    if infeasible_dict is None:
        infeasible_dict = {}
    stats_hashable = np.array([np_hashable(i) for i in stats])
    np_hash_table = {stats_hashable[i]: stats[i] for i in range(len(stats))}
    # create an empty graph
    G = nx.DiGraph()
    # create gamma and omega in the paper
    gamm: List[(np.ndarray, np.ndarray)] = []
    omeg: List[int] = []

    number_class = np.unique(stats[0])
    number_class = number_class[number_class > 0]
    # add node
    for s in stats:
        G.add_node(np_hashable(s))
        # print(_E(s, goal_pattern, number_class))
    # add edge
    for i in np.arange(0, len(stats) - 1):
        for j in np.arange(i + 1, len(stats)):
            d = N(stats[i]) - N_C(stats[i], stats[j])
            if toggle_debug:
                fig = drawer.plot_states([stats[i]], row=1)
                p_1 = Plot(fig=fig).get_img()
                fig = drawer.plot_states([stats[j]], row=1)
                p_2 = Plot(fig=fig).get_img()
                p = np.concatenate((p_1, p_2))
                cv2.imwrite("debug.jpg", p)
                print(f"i is {i}, j is {j}, d is {d}")
            if d == 0:
                # no different between stats[i] and stats[j]
                G.add_edge(stats_hashable[i], stats_hashable[j], weight=0)
            elif d == 1:
                action = action_between_states(stats[i], stats[j], toggle_strict_mode=False)
                if action is None or action in infeasible_dict.get(stats_hashable[i], []):
                    continue
                G.add_edge(stats_hashable[i], stats_hashable[j], weight=1)
            elif d > 1:
                # TODO ADD CONSTRAINTS
                if d > (j - i):  # make sure
                    gamm.append([stats[i], stats[j]])
                    omeg.append(j - i)

                if goal_pattern is not None:
                    stats_i_enp = _E(stats[i], goal_pattern=goal_pattern, number_class=number_class)
                    stats_j_enp = _E(stats[j], goal_pattern=goal_pattern, number_class=number_class)
                    enp_diff_i_j = stats_j_enp - stats_i_enp
                    if np.sum(enp_diff_i_j) < j - i and np.all(enp_diff_i_j > 0):
                        # print(j - i, np.sum(enp_diff_i_j))
                        # from huri.components.utils.matlibplot_utils import Plot
                        # fig = drawer.plot_states([stats[i]])
                        # p1 = Plot(fig=fig)
                        # fig = drawer.plot_states([stats[j]])
                        # p2 = Plot(fig=fig)
                        # cv2.imshow("i", p1.get_img())
                        # cv2.imshow("j", p2.get_img())
                        # cv2.waitKey(0)
                        gamm.append([stats[0], stats[j]])
                        omeg.append(j - i)
                        # for zzz, _ in enumerate(stats_i_enp == stats_j_enp):
                        #     if _:
                        #         c = number_class[zzz]
                        #         new_state = stats[j].copy()
                        #         new_state[stats[i] == c] = stats[i][stats[i] == c]
                        #         G.add_node(np_hashable(new_state))
                        # gamm.append([stats[i], new_state])
                        # omeg.append(j - i)

            else:
                raise Exception("Unexpected errors")

                # see the nodes
    # print(gamm)
    # if [stats[0], stats[-1]] not in gamm:
    #     gamm.append([stats[0], stats[-1]])
    #     omeg.append(-1)
    # debug lot nodes
    if toggle_debug:
        pos = nx.spring_layout(G)
        nx.draw(G, pos)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.show()

    # refine edge
    # counter = 0
    aa = time.time()

    for i in np.arange(len(gamm)):
        if omeg[i] <= h:
            # A start
            # counter += 1
            a = TubePuzzle(elearray=gamm[i][0], goalpattern=gamm[i][1])
            is_find, path = a.atarSearch(infeasible_dict=infeasible_dict, max_iter_cnt=50)
            if is_find:
                # path: List = []
                # print(omeg[i] - len(path))

                # if omeg[i] - len(path) < -1:
                #     from huri.components.utils.matlibplot_utils import Plot
                #     fig = drawer.plot_states(path)
                #     p = Plot(fig=fig)
                #     cv2.imshow("i", p.get_img())
                #     cv2.waitKey(0)

                for j in range(1, len(path)):
                    node = np_hashable(path[j])
                    if node not in G:
                        np_hash_table[node] = path[j]
                        G.add_node(node)
                    if not G.has_edge(np_hashable(path[j - 1]), np_hashable(node)):
                        G.add_edge(np_hashable(path[j - 1]), np_hashable(node), weight=1)

    b = time.time()
    print("section A start consuming is ", b - aa)
    # print(counter)
    r_stats_hashable = []
    if len(stats_hashable) < 2:
        return r_stats_hashable
    r_stats_hashable_1 = []
    aa = time.time()
    for ii in range(max(len(stats_hashable) - 2, 0)):
        if len(stats) - ii > h:
            continue
        if not G.has_edge(stats_hashable[ii], stats_hashable[-1]):
            a = TubePuzzle(elearray=stats[ii], goalpattern=stats[-1])
            is_find, path = a.atarSearch(infeasible_dict=infeasible_dict, max_iter_cnt=50)
            if is_find:
                for j in range(1, len(path)):
                    node = np_hashable(path[j])
                    if node not in G:
                        np_hash_table[node] = path[j]
                        G.add_node(node)
                    if not G.has_edge(np_hashable(path[j - 1]), np_hashable(node)):
                        G.add_edge(np_hashable(path[j - 1]), np_hashable(node), weight=1)
    b = time.time()
    print("section B start consuming is ", b - aa)
    aa = time.time()
    for ii in range(1, max(len(stats_hashable) - 1, 0)):
        if ii >= h:
            break
        if not G.has_edge(stats_hashable[0], stats_hashable[ii]):
            a = TubePuzzle(elearray=stats[0], goalpattern=stats[ii])
            is_find, path = a.atarSearch(infeasible_dict=infeasible_dict, max_iter_cnt=50)
            if is_find:
                for j in range(1, len(path)):
                    node = np_hashable(path[j])
                    if node not in G:
                        np_hash_table[node] = path[j]
                        G.add_node(node)
                    if not G.has_edge(np_hashable(path[j - 1]), np_hashable(node)):
                        G.add_edge(np_hashable(path[j - 1]), np_hashable(node), weight=1)
    b = time.time()
    print("section C start consuming is ", b - aa)
    for ii in range(max(len(stats_hashable) - 2, 0)):
        r_stats_hashable_tmp = np.array(
            nx.shortest_path(G, source=stats_hashable[ii], target=stats_hashable[-1], weight='weight'))
        r_stats_hashable_1.append([np_hash_table[_] for _ in r_stats_hashable_tmp])

    # r_stats_hashable_tmp = np.array(
    #     nx.shortest_path(G, source=stats_hashable[ii], target=stats_hashable[-1], weight='weight'))
    # r_stats_hashable_1.append([np_hash_table[_] for _ in r_stats_hashable_tmp])

    r_stats_hashable_2 = []
    for ii in range(1, max(len(stats_hashable) - 1, 0)):
        r_stats_hashable_tmp = np.array(
            nx.shortest_path(G, source=stats_hashable[0], target=stats_hashable[ii], weight='weight'))
        r_stats_hashable_2.append([np_hash_table[_] for _ in r_stats_hashable_tmp])
    # refined, her refined
    return r_stats_hashable_1, r_stats_hashable_2


def rm_ras_actions_recur3(stats: List[np.ndarray], h: int = 1,
                          max_refine_num=100,
                          goal_pattern=None,
                          infeasible_dict: dict = None,
                          toggle_debug=False) -> (
        List[int], List[np.ndarray]):
    if infeasible_dict is None:
        infeasible_dict = {}
    stats_hashable = np.array([np_hashable(i) for i in stats])
    np_hash_table = {stats_hashable[i]: stats[i] for i in range(len(stats))}
    # create an empty graph
    G = nx.DiGraph()
    # create gamma and omega in the paper
    gamm: List[(np.ndarray, np.ndarray)] = []
    omeg: List[int] = []

    number_class = np.unique(stats[0])
    number_class = number_class[number_class > 0]
    # add node
    for s in stats:
        G.add_node(np_hashable(s))
        # print(_E(s, goal_pattern, number_class))
    # add edge
    for i in np.arange(0, len(stats) - 1):
        for j in np.arange(i + 1, len(stats)):
            d = N(stats[i]) - N_C(stats[i], stats[j])
            if toggle_debug:
                fig = drawer.plot_states([stats[i]], row=1)
                p_1 = Plot(fig=fig).get_img()
                fig = drawer.plot_states([stats[j]], row=1)
                p_2 = Plot(fig=fig).get_img()
                p = np.concatenate((p_1, p_2))
                cv2.imwrite("debug.jpg", p)
                print(f"i is {i}, j is {j}, d is {d}")
            if d == 0:
                # no different between stats[i] and stats[j]
                G.add_edge(stats_hashable[i], stats_hashable[j], weight=0)
            elif d == 1:
                action = action_between_states(stats[i], stats[j], toggle_strict_mode=False)
                if action is None or action in infeasible_dict.get(stats_hashable[i], []):
                    continue
                G.add_edge(stats_hashable[i], stats_hashable[j], weight=1)
            elif d > 1:
                # if j - i > h:  # over the horizon
                #     continue
                # TODO ADD CONSTRAINTS
                if d > (j - i):  # make sure
                    gamm.append([stats[i], stats[j]])
                    omeg.append(j - i)
                    continue
                # TODO Complete the following code
                if goal_pattern is not None:
                    stats_i_enp = _E(stats[i], goal_pattern=goal_pattern, number_class=number_class)
                    stats_j_enp = _E(stats[j], goal_pattern=goal_pattern, number_class=number_class)
                    enp_diff_i_j = stats_j_enp - stats_i_enp
                    if np.sum(enp_diff_i_j) < j - i and np.all(enp_diff_i_j > 0):
                        # print(j - i, np.sum(enp_diff_i_j))
                        # from huri.components.utils.matlibplot_utils import Plot
                        # fig = drawer.plot_states([stats[i]])
                        # p1 = Plot(fig=fig)
                        # fig = drawer.plot_states([stats[j]])
                        # p2 = Plot(fig=fig)
                        # cv2.imshow("i", p1.get_img())
                        # cv2.imshow("j", p2.get_img())
                        # cv2.waitKey(0)
                        gamm.append([stats[0], stats[j]])
                        omeg.append(j - i)
                        # for zzz, _ in enumerate(stats_i_enp == stats_j_enp):
                        #     if _:
                        #         c = number_class[zzz]
                        #         new_state = stats[j].copy()
                        #         new_state[stats[i] == c] = stats[i][stats[i] == c]
                        #         G.add_node(np_hashable(new_state))
                        # gamm.append([stats[i], new_state])
                        # omeg.append(j - i)

            else:
                raise Exception("Unexpected errors")

                # see the nodes
    # print(gamm)
    # if [stats[0], stats[-1]] not in gamm:
    #     gamm.append([stats[0], stats[-1]])
    #     omeg.append(-1)
    # debug lot nodes
    if toggle_debug:
        pos = nx.spring_layout(G)
        nx.draw(G, pos)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.show()

    # refine edge
    # counter = 0
    # aa = time.time()
    gamm_f = [gamm[i] for i, v in enumerate(omeg) if v <= h]
    selected_gamm_f_ids = np.random.choice(len(gamm_f), min(len(gamm_f), max_refine_num), replace=False)
    # print(f"selected_gamm_f_ids is {selected_gamm_f_ids}")
    for i in selected_gamm_f_ids:
        a = TubePuzzle(elearray=gamm_f[i][0], goalpattern=gamm_f[i][1])
        is_find, path = a.atarSearch(infeasible_dict=infeasible_dict, max_iter_cnt=20)
        if is_find:
            # path: List = []
            # print(omeg[i] - len(path))
            # if omeg[i] - len(path) < -1:
            #     from huri.components.utils.matlibplot_utils import Plot
            #     fig = drawer.plot_states(path)
            #     p = Plot(fig=fig)
            #     cv2.imshow("i", p.get_img())
            #     cv2.waitKey(0)
            for j in range(1, len(path)):
                node = np_hashable(path[j])
                if node not in G:
                    np_hash_table[node] = path[j]
                    G.add_node(node)
                if not G.has_edge(np_hashable(path[j - 1]), np_hashable(node)):
                    G.add_edge(np_hashable(path[j - 1]), np_hashable(node), weight=1)

    # b = time.time()
    # print("section A start consuming is ", b - aa)
    # print(counter)
    r_stats_hashable = []
    if len(stats_hashable) < 2:
        return r_stats_hashable
    r_stats_hashable_1 = []
    # aa = time.time()
    for ii in range(max(len(stats_hashable) - 2, 0)):
        # if len(stats) - ii > h:
        #     continue
        if not G.has_edge(stats_hashable[ii], stats_hashable[-1]):
            a = TubePuzzle(elearray=stats[ii], goalpattern=stats[-1])
            is_find, path = a.atarSearch(infeasible_dict=infeasible_dict, max_iter_cnt=50)
            if is_find:
                for j in range(1, len(path)):
                    node = np_hashable(path[j])
                    if node not in G:
                        np_hash_table[node] = path[j]
                        G.add_node(node)
                    if not G.has_edge(np_hashable(path[j - 1]), np_hashable(node)):
                        G.add_edge(np_hashable(path[j - 1]), np_hashable(node), weight=1)
    # b = time.time()
    # print("section B start consuming is ", b - aa)
    # aa = time.time()
    for ii in range(1, max(len(stats_hashable) - 1, 0)):
        # if ii >= h:
        #     break
        if not G.has_edge(stats_hashable[0], stats_hashable[ii]):
            a = TubePuzzle(elearray=stats[0], goalpattern=stats[ii])
            is_find, path = a.atarSearch(infeasible_dict=infeasible_dict, max_iter_cnt=50)
            if is_find:
                for j in range(1, len(path)):
                    node = np_hashable(path[j])
                    if node not in G:
                        np_hash_table[node] = path[j]
                        G.add_node(node)
                    if not G.has_edge(np_hashable(path[j - 1]), np_hashable(node)):
                        G.add_edge(np_hashable(path[j - 1]), np_hashable(node), weight=1)
    # b = time.time()
    # print("section C start consuming is ", b - aa)
    for ii in range(max(len(stats_hashable) - 2, 0)):
        # for ii in range(1):
        r_stats_hashable_tmp = np.array(
            nx.shortest_path(G, source=stats_hashable[ii], target=stats_hashable[-1], weight='weight'))
        r_stats_hashable_1.append([np_hash_table[_] for _ in r_stats_hashable_tmp])

    # r_stats_hashable_tmp = np.array(
    #     nx.shortest_path(G, source=stats_hashable[ii], target=stats_hashable[-1], weight='weight'))
    # r_stats_hashable_1.append([np_hash_table[_] for _ in r_stats_hashable_tmp])

    r_stats_hashable_2 = []
    for ii in range(1, max(len(stats_hashable) - 1, 0)):
        r_stats_hashable_tmp = np.array(
            nx.shortest_path(G, source=stats_hashable[0], target=stats_hashable[ii], weight='weight'))
        r_stats_hashable_2.append([np_hash_table[_] for _ in r_stats_hashable_tmp])
    # refined, her refined
    return r_stats_hashable_1, r_stats_hashable_2


def rm_ras_actions_recur3_5(stats: List[np.ndarray], h: int = 1,
                            max_refine_num=100,
                            goal_pattern=None,
                            infeasible_dict: dict = None,
                            condition_set: np.ndarray = None,
                            toggle_debug=False) -> (
        List[int], List[np.ndarray]):
    if infeasible_dict is None:
        infeasible_dict = {}
    stats_hashable = np.array([np_hashable(i) for i in stats])
    np_hash_table = {stats_hashable[i]: stats[i] for i in range(len(stats))}
    # create an empty graph
    G = nx.DiGraph()
    # create gamma and omega in the paper
    gamm: List[(np.ndarray, np.ndarray)] = []
    omeg: List[int] = []

    number_class = np.unique(stats[0])
    number_class = number_class[number_class > 0]
    # add node
    for s in stats:
        G.add_node(np_hashable(s))
        # print(_E(s, goal_pattern, number_class))
    # add edge
    for i in np.arange(0, len(stats) - 1):
        for j in np.arange(i + 1, len(stats)):
            d = N(stats[i]) - N_C(stats[i], stats[j])
            if toggle_debug:
                fig = drawer.plot_states([stats[i]], row=1)
                p_1 = Plot(fig=fig).get_img()
                fig = drawer.plot_states([stats[j]], row=1)
                p_2 = Plot(fig=fig).get_img()
                p = np.concatenate((p_1, p_2))
                cv2.imwrite("debug.jpg", p)
                print(f"i is {i}, j is {j}, d is {d}")
            if d == 0:
                # no different between stats[i] and stats[j]
                G.add_edge(stats_hashable[i], stats_hashable[j], weight=0)
            elif d == 1:
                action = action_between_states(stats[i], stats[j], toggle_strict_mode=False)
                if action is None or action in infeasible_dict.get(stats_hashable[i], []):
                    continue
                G.add_edge(stats_hashable[i], stats_hashable[j], weight=1)
            elif d > 1:
                # if j - i > h:  # over the horizon
                #     continue
                # TODO ADD CONSTRAINTS
                if d > (j - i):  # make sure
                    gamm.append([stats[i], stats[j]])
                    omeg.append(j - i)
                    continue
                # TODO Complete the following code
                if goal_pattern is not None:
                    stats_i_enp = _E(stats[i], goal_pattern=goal_pattern, number_class=number_class)
                    stats_j_enp = _E(stats[j], goal_pattern=goal_pattern, number_class=number_class)
                    enp_diff_i_j = stats_j_enp - stats_i_enp
                    if np.sum(enp_diff_i_j) < j - i and np.all(enp_diff_i_j > 0):
                        # print(j - i, np.sum(enp_diff_i_j))
                        # from huri.components.utils.matlibplot_utils import Plot
                        # fig = drawer.plot_states([stats[i]])
                        # p1 = Plot(fig=fig)
                        # fig = drawer.plot_states([stats[j]])
                        # p2 = Plot(fig=fig)
                        # cv2.imshow("i", p1.get_img())
                        # cv2.imshow("j", p2.get_img())
                        # cv2.waitKey(0)
                        gamm.append([stats[0], stats[j]])
                        omeg.append(j - i)
                        # for zzz, _ in enumerate(stats_i_enp == stats_j_enp):
                        #     if _:
                        #         c = number_class[zzz]
                        #         new_state = stats[j].copy()
                        #         new_state[stats[i] == c] = stats[i][stats[i] == c]
                        #         G.add_node(np_hashable(new_state))
                        # gamm.append([stats[i], new_state])
                        # omeg.append(j - i)

            else:
                raise Exception("Unexpected errors")

                # see the nodes
    # print(gamm)
    # if [stats[0], stats[-1]] not in gamm:
    #     gamm.append([stats[0], stats[-1]])
    #     omeg.append(-1)
    # debug lot nodes
    if toggle_debug:
        pos = nx.spring_layout(G)
        nx.draw(G, pos)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.show()

    # refine edge
    # counter = 0
    # aa = time.time()
    gamm_f = [gamm[i] for i, v in enumerate(omeg) if v <= h]
    selected_gamm_f_ids = np.random.choice(len(gamm_f), min(len(gamm_f), max_refine_num), replace=False)
    # print(f"selected_gamm_f_ids is {selected_gamm_f_ids}")
    for i in selected_gamm_f_ids:
        a = TubePuzzle(elearray=gamm_f[i][0], goalpattern=gamm_f[i][1])
        is_find, path = a.atarSearch(infeasible_dict=infeasible_dict, max_iter_cnt=20)
        if is_find:
            # path: List = []
            # print(omeg[i] - len(path))
            # if omeg[i] - len(path) < -1:
            #     from huri.components.utils.matlibplot_utils import Plot
            #     fig = drawer.plot_states(path)
            #     p = Plot(fig=fig)
            #     cv2.imshow("i", p.get_img())
            #     cv2.waitKey(0)
            for j in range(1, len(path)):
                node = np_hashable(path[j])
                if node not in G:
                    np_hash_table[node] = path[j]
                    G.add_node(node)
                if not G.has_edge(np_hashable(path[j - 1]), np_hashable(node)):
                    G.add_edge(np_hashable(path[j - 1]), np_hashable(node), weight=1)

    # b = time.time()
    # print("section A start consuming is ", b - aa)
    # print(counter)
    r_stats_hashable = []
    if len(stats_hashable) < 2:
        return r_stats_hashable
    r_stats_hashable_1 = []
    # aa = time.time()
    for ii in range(max(len(stats_hashable) - 2, 0)):
        if len(stats) - ii > h:
            continue
        if not G.has_edge(stats_hashable[ii], stats_hashable[-1]):
            a = TubePuzzle(elearray=stats[ii], goalpattern=stats[-1])
            is_find, path = a.atarSearch(infeasible_dict=infeasible_dict, max_iter_cnt=20)
            if is_find:
                for j in range(1, len(path)):
                    node = np_hashable(path[j])
                    if node not in G:
                        np_hash_table[node] = path[j]
                        G.add_node(node)
                    if not G.has_edge(np_hashable(path[j - 1]), np_hashable(node)):
                        G.add_edge(np_hashable(path[j - 1]), np_hashable(node), weight=1)
    # b = time.time()
    # print("section B start consuming is ", b - aa)
    # aa = time.time()
    for ii in range(1, max(len(stats_hashable) - 1, 0)):
        if ii >= h:
            break
        if not G.has_edge(stats_hashable[0], stats_hashable[ii]):
            a = TubePuzzle(elearray=stats[0], goalpattern=stats[ii])
            is_find, path = a.atarSearch(infeasible_dict=infeasible_dict, max_iter_cnt=20)
            if is_find:
                for j in range(1, len(path)):
                    node = np_hashable(path[j])
                    if node not in G:
                        np_hash_table[node] = path[j]
                        G.add_node(node)
                    if not G.has_edge(np_hashable(path[j - 1]), np_hashable(node)):
                        G.add_edge(np_hashable(path[j - 1]), np_hashable(node), weight=1)
    # b = time.time()
    # print("section C start consuming is ", b - aa)
    # for ii in range(min(max(len(stats_hashable) - 2, 0), h)):
    for ii in range(1):
        r_stats_hashable_tmp = np.array(
            nx.shortest_path(G, source=stats_hashable[ii], target=stats_hashable[-1], weight='weight'))
        path_tmp = []
        for _ in r_stats_hashable_tmp:
            path_tmp.append(np_hash_table[_])
            if isdone(np_hash_table[_], goal_pattern):
                break
        r_stats_hashable_1.append(path_tmp)
    return r_stats_hashable_1


def rm_ras_actions_recur3_6(stats: List[np.ndarray],
                            h: int = 1,
                            max_refine_num=100,
                            goal_pattern=None,
                            infeasible_dict: dict = None,
                            condition_set: np.ndarray = None,
                            infeasible_set: list = None,
                            toggle_debug=False) -> (
        List[int], List[np.ndarray]):
    if infeasible_dict is None:
        infeasible_dict = {}
    if infeasible_set is None:
        infeasible_set = []
    stats_hashable = np.array([np_hashable(i) for i in stats])
    np_hash_table = {stats_hashable[i]: stats[i] for i in range(len(stats))}
    # create an empty graph
    G = nx.DiGraph()
    # add node
    for ii, s in enumerate(stats):
        G.add_node(np_hashable(s))
        if ii > 0:
            G.add_edge(stats_hashable[ii - 1], stats_hashable[ii], weight=1)
        # print(_E(s, goal_pattern, number_class))

    # add edge
    for i in np.arange(0, len(stats) - 2):
        for j in np.arange(len(stats) - 1, i + 1, -1):
            d = N(stats[i]) - N_C(stats[i], stats[j])
            if toggle_debug:
                fig = drawer.plot_states([stats[i]], row=1)
                p_1 = Plot(fig=fig).get_img()
                fig = drawer.plot_states([stats[j]], row=1)
                p_2 = Plot(fig=fig).get_img()
                p = np.concatenate((p_1, p_2))
                cv2.imwrite("debug.jpg", p)
                print(f"i is {i}, j is {j}, d is {d}")
            if d == 0:
                # no different between stats[i] and stats[j]
                G.add_edge(stats_hashable[i], stats_hashable[j], weight=0)
                break
            elif d == 1:
                action = action_between_states_condition_set(stats[i],
                                                             stats[j],
                                                             toggle_strict_mode=False,
                                                             condition_set=condition_set)
                if action is None or action in infeasible_dict.get(stats_hashable[i], []) + infeasible_set:
                    continue
                G.add_edge(stats_hashable[i], stats_hashable[j], weight=1)
                break
            elif d > 1:
                # if j - i > h:  # over the horizon
                #     continue
                # TODO ADD CONSTRAINTS
                if d < (j - i) < h:  # make sure
                    a = TubePuzzle(elearray=stats[i],
                                   goalpattern=stats[j])
                    is_find, path = a.atarSearch(condition_set=condition_set,
                                                 infeasible_dict=infeasible_dict,
                                                 infeasible_set=infeasible_set,
                                                 max_iter_cnt=20, )
                    if is_find:
                        # path: List = []
                        # print(omeg[i] - len(path))
                        # if omeg[i] - len(path) < -1:
                        #     from huri.components.utils.matlibplot_utils import Plot
                        #     fig = drawer.plot_states(path)
                        #     p = Plot(fig=fig)
                        #     cv2.imshow("i", p.get_img())
                        #     cv2.waitKey(0)
                        for zz in range(1, len(path)):
                            node = np_hashable(path[zz])
                            if node not in G:
                                np_hash_table[node] = path[zz]
                                G.add_node(node)
                            if not G.has_edge(np_hashable(path[zz - 1]), np_hashable(node)):
                                G.add_edge(np_hashable(path[zz - 1]), np_hashable(node), weight=1)
                        break
                # TODO Complete the following code
                # if goal_pattern is not None:
                #     stats_i_enp = _E(stats[i], goal_pattern=goal_pattern, number_class=number_class)
                #     stats_j_enp = _E(stats[j], goal_pattern=goal_pattern, number_class=number_class)
                #     enp_diff_i_j = stats_j_enp - stats_i_enp
                #     if np.sum(enp_diff_i_j) < j - i and np.all(enp_diff_i_j > 0):
                #         # print(j - i, np.sum(enp_diff_i_j))
                #         # from huri.components.utils.matlibplot_utils import Plot
                #         # fig = drawer.plot_states([stats[i]])
                #         # p1 = Plot(fig=fig)
                #         # fig = drawer.plot_states([stats[j]])
                #         # p2 = Plot(fig=fig)
                #         # cv2.imshow("i", p1.get_img())
                #         # cv2.imshow("j", p2.get_img())
                #         # cv2.waitKey(0)
                #         gamm.append([stats[0], stats[j]])
                #         omeg.append(j - i)
                #         # for zzz, _ in enumerate(stats_i_enp == stats_j_enp):
                #         #     if _:
                #         #         c = number_class[zzz]
                #         #         new_state = stats[j].copy()
                #         #         new_state[stats[i] == c] = stats[i][stats[i] == c]
                #         #         G.add_node(np_hashable(new_state))
                #         # gamm.append([stats[i], new_state])
                #         # omeg.append(j - i)

            else:
                raise Exception("Unexpected errors")

                # see the nodes
    # print(gamm)
    # if [stats[0], stats[-1]] not in gamm:
    #     gamm.append([stats[0], stats[-1]])
    #     omeg.append(-1)
    # debug lot nodes
    if toggle_debug:
        pos = nx.spring_layout(G)
        nx.draw(G, pos)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.show()

    # refine edge
    # counter = 0
    # aa = time.time()
    # gamm_f = [gamm[i] for i, v in enumerate(omeg) if v <= h]
    # selected_gamm_f_ids = np.random.choice(len(gamm_f), min(len(gamm_f), max_refine_num), replace=False)
    # print(f"selected_gamm_f_ids is {selected_gamm_f_ids}")
    # for i in selected_gamm_f_ids:
    #     a = TubePuzzle(elearray=gamm_f[i][0], goalpattern=gamm_f[i][1])
    #     is_find, path = a.atarSearch(infeasible_dict=infeasible_dict, max_iter_cnt=20)
    #     if is_find:
    #         # path: List = []
    #         # print(omeg[i] - len(path))
    #         # if omeg[i] - len(path) < -1:
    #         #     from huri.components.utils.matlibplot_utils import Plot
    #         #     fig = drawer.plot_states(path)
    #         #     p = Plot(fig=fig)
    #         #     cv2.imshow("i", p.get_img())
    #         #     cv2.waitKey(0)
    #         for j in range(1, len(path)):
    #             node = np_hashable(path[j])
    #             if node not in G:
    #                 np_hash_table[node] = path[j]
    #                 G.add_node(node)
    #             if not G.has_edge(np_hashable(path[j - 1]), np_hashable(node)):
    #                 G.add_edge(np_hashable(path[j - 1]), np_hashable(node), weight=1)

    # b = time.time()
    # print("section A start consuming is ", b - aa)
    # print(counter)
    r_stats_hashable = []
    if len(stats_hashable) < 2:
        return r_stats_hashable
    r_stats_hashable_1 = []
    # aa = time.time()
    for ii in range(max(len(stats_hashable) - 2, 0)):
        if len(stats) - ii > h:
            continue
        if not G.has_edge(stats_hashable[ii], stats_hashable[-1]):
            a = TubePuzzle(elearray=stats[ii], goalpattern=stats[-1])
            is_find, path = a.atarSearch(infeasible_dict=infeasible_dict,
                                         infeasible_set=infeasible_set,
                                         condition_set=condition_set, max_iter_cnt=20)
            if is_find:
                for j in range(1, len(path)):
                    node = np_hashable(path[j])
                    if node not in G:
                        np_hash_table[node] = path[j]
                        G.add_node(node)
                    if not G.has_edge(np_hashable(path[j - 1]), np_hashable(node)):
                        G.add_edge(np_hashable(path[j - 1]), np_hashable(node), weight=1)
    # b = time.time()
    # print("section B start consuming is ", b - aa)
    # aa = time.time()
    for ii in range(1, max(len(stats_hashable) - 1, 0)):
        if ii >= h:
            break
        if not G.has_edge(stats_hashable[0], stats_hashable[ii]):
            a = TubePuzzle(elearray=stats[0], goalpattern=stats[ii])
            is_find, path = a.atarSearch(infeasible_dict=infeasible_dict,
                                         infeasible_set=infeasible_set,
                                         condition_set=condition_set,
                                         max_iter_cnt=20)
            if is_find:
                for j in range(1, len(path)):
                    node = np_hashable(path[j])
                    if node not in G:
                        np_hash_table[node] = path[j]
                        G.add_node(node)
                    if not G.has_edge(np_hashable(path[j - 1]), np_hashable(node)):
                        G.add_edge(np_hashable(path[j - 1]), np_hashable(node), weight=1)
    # b = time.time()
    # print("section C start consuming is ", b - aa)
    # for ii in range(min(max(len(stats_hashable) - 2, 0), h)):
    for ii in range(1):
        r_stats_hashable_tmp = np.array(
            nx.shortest_path(G, source=stats_hashable[ii], target=stats_hashable[-1], weight='weight'))
        path_tmp = []
        for _ in r_stats_hashable_tmp:
            path_tmp.append(np_hash_table[_])
            if isdone(np_hash_table[_], goal_pattern):
                break
        r_stats_hashable_1.append(path_tmp)
    return r_stats_hashable_1


def rm_ras_actions_fixedgoal(stats: List[np.ndarray],
                             h: int = 1,
                             goal_pattern=None,
                             infeasible_dict: dict = None,
                             toggle_debug=False) -> (List[int], List[np.ndarray]):
    if infeasible_dict is None:
        infeasible_dict = {}
    stats_hashable = np.array([np_hashable(i) for i in stats])
    np_hash_table = {stats_hashable[i]: stats[i] for i in range(len(stats))}
    # create an empty graph
    G = nx.DiGraph()
    # create gamma and omega in the paper
    gamm: List[(np.ndarray, np.ndarray)] = []
    omeg: List[int] = []

    number_class = np.unique(stats[0])
    number_class = number_class[number_class > 0]
    # add node
    for s in stats:
        G.add_node(np_hashable(s))
        # print(_E(s, goal_pattern, number_class))
    # add edge
    for i in np.arange(0, len(stats) - 1):
        for j in np.arange(i + 1, len(stats)):
            d = N(stats[i]) - N_C(stats[i], stats[j])
            if toggle_debug:
                fig = drawer.plot_states([stats[i]], row=1)
                p_1 = Plot(fig=fig).get_img()
                fig = drawer.plot_states([stats[j]], row=1)
                p_2 = Plot(fig=fig).get_img()
                p = np.concatenate((p_1, p_2))
                cv2.imwrite("debug.jpg", p)
                print(f"i is {i}, j is {j}, d is {d}")
            if d == 0:
                # no different between stats[i] and stats[j]
                G.add_edge(stats_hashable[i], stats_hashable[j], weight=0)
            elif d == 1:
                action = action_between_states(stats[i], stats[j], toggle_strict_mode=False)
                if action is None or action in infeasible_dict.get(stats_hashable[i], []):
                    continue
                G.add_edge(stats_hashable[i], stats_hashable[j], weight=1)
            elif d > 1:
                # TODO ADD CONSTRAINTS
                if d > (j - i):  # make sure
                    gamm.append([stats[i], stats[j]])
                    omeg.append(j - i)

                if goal_pattern is not None:
                    stats_i_enp = _E(stats[i], goal_pattern=goal_pattern, number_class=number_class)
                    stats_j_enp = _E(stats[j], goal_pattern=goal_pattern, number_class=number_class)
                    enp_diff_i_j = stats_j_enp - stats_i_enp
                    if np.sum(enp_diff_i_j) < j - i:
                        # if np.sum(enp_diff_i_j) < j - i and np.all(enp_diff_i_j > 0):
                        # print(j - i, np.sum(enp_diff_i_j))
                        # from huri.components.utils.matlibplot_utils import Plot
                        # fig = drawer.plot_states([stats[i]])
                        # p1 = Plot(fig=fig)
                        # fig = drawer.plot_states([stats[j]])
                        # p2 = Plot(fig=fig)
                        # cv2.imshow("i", p1.get_img())
                        # cv2.imshow("j", p2.get_img())
                        # cv2.waitKey(0)
                        gamm.append([stats[0], stats[j]])
                        omeg.append(j - i)
                        # for zzz, _ in enumerate(stats_i_enp == stats_j_enp):
                        #     if _:
                        #         c = number_class[zzz]
                        #         new_state = stats[j].copy()
                        #         new_state[stats[i] == c] = stats[i][stats[i] == c]
                        #         G.add_node(np_hashable(new_state))
                        # gamm.append([stats[i], new_state])
                        # omeg.append(j - i)

            else:
                raise Exception("Unexpected errors")

                # see the nodes
    # print(gamm)
    # if [stats[0], stats[-1]] not in gamm:
    #     gamm.append([stats[0], stats[-1]])
    #     omeg.append(-1)
    # debug lot nodes
    if toggle_debug:
        pos = nx.spring_layout(G)
        nx.draw(G, pos)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.show()

        # refine edge
    for i in np.arange(len(gamm)):
        if omeg[i] <= h:
            # A start
            a = TubePuzzle(elearray=gamm[i][0], goalpattern=gamm[i][1])
            is_find, path = a.atarSearch(infeasible_dict=infeasible_dict, max_iter_cnt=100)
            if is_find:
                for j in range(1, len(path)):
                    node = np_hashable(path[j])
                    if node not in G:
                        np_hash_table[node] = path[j]
                        G.add_node(node)
                    if not G.has_edge(np_hashable(path[j - 1]), np_hashable(node)):
                        G.add_edge(np_hashable(path[j - 1]), np_hashable(node), weight=1)

    r_stats_hashable = []
    if len(stats_hashable) < 2:
        return r_stats_hashable

    r_stats_hashable_1 = []

    # try to connect all the points to the last point
    for ii in range(max(len(stats_hashable) - 2, 0)):
        a = TubePuzzle(elearray=stats[ii], goalpattern=stats[-1])
        is_find, path = a.atarSearch(infeasible_dict=infeasible_dict, max_iter_cnt=100)
        if is_find:
            for j in range(1, len(path)):
                node = np_hashable(path[j])
                if node not in G:
                    np_hash_table[node] = path[j]
                    G.add_node(node)
                if not G.has_edge(np_hashable(path[j - 1]), np_hashable(node)):
                    G.add_edge(np_hashable(path[j - 1]), np_hashable(node), weight=1)

    r_stats_hashable_tmp = np.array(
        nx.shortest_path(G, source=stats_hashable[0], target=stats_hashable[-1], weight='weight'))
    r_stats_hashable_1.append([np_hash_table[_] for _ in r_stats_hashable_tmp])

    # try to find all the points to the goal
    iiter = range(1, max(len(stats_hashable) - 2, 1))
    for ii in np.random.choice(iiter, min(10, len(iiter)), replace=False):
        a = TubePuzzle(elearray=stats[ii], goalpattern=goal_pattern)
        is_find, path = a.atarSearch(infeasible_dict=infeasible_dict, max_iter_cnt=100)
        if is_find and not np.array_equal(path[-1], stats[-1]):
            for j in range(1, len(path)):
                node = np_hashable(path[j])
                if node not in G:
                    np_hash_table[node] = path[j]
                    G.add_node(node)
                if not G.has_edge(np_hashable(path[j - 1]), np_hashable(node)):
                    G.add_edge(np_hashable(path[j - 1]), np_hashable(node), weight=1)
            r_stats_hashable_tmp = np.array(
                nx.shortest_path(G, source=stats_hashable[ii], target=np_hashable(path[-1]), weight='weight'))
            r_stats_hashable_1.append([np_hash_table[_] for _ in r_stats_hashable_tmp])

    return r_stats_hashable_1


def rm_ras_actions_fixedgoal2(stats: List[np.ndarray],
                              h: int = 1,
                              goal_pattern=None,
                              infeasible_dict: dict = None,
                              toggle_debug=False) -> (List[int], List[np.ndarray]):
    if infeasible_dict is None:
        infeasible_dict = {}
    stats_hashable = np.array([np_hashable(i) for i in stats])
    np_hash_table = {stats_hashable[i]: stats[i] for i in range(len(stats))}
    # create an empty graph
    G = nx.DiGraph()
    # create gamma and omega in the paper
    gamm: List[(np.ndarray, np.ndarray)] = []
    omeg: List[int] = []

    number_class = np.unique(stats[0])
    number_class = number_class[number_class > 0]
    # add node
    for s in stats:
        G.add_node(np_hashable(s))
        # print(_E(s, goal_pattern, number_class))
    # add edge
    for i in np.arange(0, len(stats) - 1):
        for j in np.arange(i + 1, len(stats)):
            d = N(stats[i]) - N_C(stats[i], stats[j])
            if toggle_debug:
                fig = drawer.plot_states([stats[i]], row=1)
                p_1 = Plot(fig=fig).get_img()
                fig = drawer.plot_states([stats[j]], row=1)
                p_2 = Plot(fig=fig).get_img()
                p = np.concatenate((p_1, p_2))
                cv2.imwrite("debug.jpg", p)
                print(f"i is {i}, j is {j}, d is {d}")
            if d == 0:
                # no different between stats[i] and stats[j]
                G.add_edge(stats_hashable[i], stats_hashable[j], weight=0)
            elif d == 1:
                action = action_between_states(stats[i], stats[j], toggle_strict_mode=False)
                if action is None or action in infeasible_dict.get(stats_hashable[i], []):
                    continue
                G.add_edge(stats_hashable[i], stats_hashable[j], weight=1)
            elif d > 1:
                # TODO ADD CONSTRAINTS
                if d > (j - i):  # make sure
                    gamm.append([stats[i], stats[j]])
                    omeg.append(j - i)

                if goal_pattern is not None:
                    stats_i_enp = _E(stats[i], goal_pattern=goal_pattern, number_class=number_class)
                    stats_j_enp = _E(stats[j], goal_pattern=goal_pattern, number_class=number_class)
                    enp_diff_i_j = stats_j_enp - stats_i_enp
                    if np.sum(enp_diff_i_j) < j - i:
                        # if np.sum(enp_diff_i_j) < j - i and np.all(enp_diff_i_j > 0):
                        # print(j - i, np.sum(enp_diff_i_j))
                        # from huri.components.utils.matlibplot_utils import Plot
                        # fig = drawer.plot_states([stats[i]])
                        # p1 = Plot(fig=fig)
                        # fig = drawer.plot_states([stats[j]])
                        # p2 = Plot(fig=fig)
                        # cv2.imshow("i", p1.get_img())
                        # cv2.imshow("j", p2.get_img())
                        # cv2.waitKey(0)
                        gamm.append([stats[0], stats[j]])
                        omeg.append(j - i)
                        # for zzz, _ in enumerate(stats_i_enp == stats_j_enp):
                        #     if _:
                        #         c = number_class[zzz]
                        #         new_state = stats[j].copy()
                        #         new_state[stats[i] == c] = stats[i][stats[i] == c]
                        #         G.add_node(np_hashable(new_state))
                        # gamm.append([stats[i], new_state])
                        # omeg.append(j - i)

            else:
                raise Exception("Unexpected errors")

                # see the nodes
    # print(gamm)
    # if [stats[0], stats[-1]] not in gamm:
    #     gamm.append([stats[0], stats[-1]])
    #     omeg.append(-1)
    # debug lot nodes
    if toggle_debug:
        pos = nx.spring_layout(G)
        nx.draw(G, pos)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.show()

        # refine edge
    for i in np.arange(len(gamm)):
        if omeg[i] <= h:
            # A start
            a = TubePuzzle(elearray=gamm[i][0], goalpattern=gamm[i][1])
            is_find, path = a.atarSearch(infeasible_dict=infeasible_dict, max_iter_cnt=100)
            if is_find:
                for j in range(1, len(path)):
                    node = np_hashable(path[j])
                    if node not in G:
                        np_hash_table[node] = path[j]
                        G.add_node(node)
                    if not G.has_edge(np_hashable(path[j - 1]), np_hashable(node)):
                        G.add_edge(np_hashable(path[j - 1]), np_hashable(node), weight=1)

    r_stats_hashable = []
    if len(stats_hashable) < 2:
        return r_stats_hashable
    #
    # # try to connect all the points to the last point
    # for ii in range(max(len(stats_hashable) - 2, 0)):
    #     a = TubePuzzle(elearray=stats[ii], goalpattern=stats[-1])
    #     is_find, path = a.atarSearch(infeasible_dict=infeasible_dict, max_iter_cnt=100)
    #     if is_find:
    #         for j in range(1, len(path)):
    #             node = np_hashable(path[j])
    #             if node not in G:
    #                 np_hash_table[node] = path[j]
    #                 G.add_node(node)
    #             if not G.has_edge(np_hashable(path[j - 1]), np_hashable(node)):
    #                 G.add_edge(np_hashable(path[j - 1]), np_hashable(node), weight=1)

    r_stats_hashable_tmp = np.array(
        nx.shortest_path(G, source=stats_hashable[0], target=stats_hashable[-1], weight='weight'))

    return [np_hash_table[_] for _ in r_stats_hashable_tmp]


def a_star_solve(start_state_np: np.ndarray,
                 goal_state_np: np.ndarray,
                 max_iter_cnt: int = 100,
                 infeasible_dict: dict = None):
    start_state_np = np.asarray(start_state_np)
    goal_state_np = np.asarray(goal_state_np)
    if infeasible_dict is None:
        infeasible_dict = {}
    try:
        is_find, path = TubePuzzle(elearray=start_state_np,
                                   goalpattern=goal_state_np).atarSearch(infeasible_dict=infeasible_dict,
                                                                         max_iter_cnt=max_iter_cnt)
    except Exception as e:
        print(f"Eorr: {e}")
        is_find = False
    if is_find:
        return path
    else:
        return []


if __name__ == "__main__":
    from numpy import array
    from huri.learning.env.arrangement_planning_rack.env import RackStatePlot, RackArrangementEnv
    from huri.components.utils.matlibplot_utils import Plot
    from huri.components.task_planning.tube_puzzle_learning_solver import DQNSolver
    import huri.core.file_sys as fs

    redundant_path = fs.load_pickle("test.pkl")
    goal_state_np = array([[1, 1, 0, 0, 0, 0, 0, 0, 3, 3],
                           [1, 1, 0, 0, 0, 0, 0, 0, 3, 3],
                           [1, 1, 0, 0, 0, 0, 0, 0, 3, 3],
                           [1, 1, 0, 0, 0, 0, 0, 0, 3, 3],
                           [1, 1, 0, 0, 0, 0, 0, 0, 3, 3]])
    drawer = RackStatePlot(goal_state_np)
    fig = drawer.plot_states(redundant_path, row=12)
    p = Plot(fig=fig)
    p.save_fig("test-1.jpg", dpi=300)

    refined_path = rm_ras_actions_fixedgoal(redundant_path,
                                            h=4,
                                            goal_pattern=goal_state_np,
                                            infeasible_dict={}, )
    for _ in range(len(refined_path)):
        fig = drawer.plot_states(refined_path[_], row=12)
        p = Plot(fig=fig)
        p.save_fig(f"test{_}.jpg", dpi=300)

    exit(0)

    d = np.load('test.npy')

    goal_pattern = array([[2, 0, 2],
                          [2, 0, 2],
                          [2, 0, 0]])

    refined_path = rm_ras_actions_recur(d, h=8, goal_pattern=goal_pattern, infeasible_dict={})
    print(d[21])
    [print(_) for _ in refined_path[21]]
    exit(0)

    GOAL_PATTERN = np.array([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                             [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                             [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                             [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                             [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]])
    solver = DQNSolver()

    env = RackArrangementEnv(num_classes=3, is_goalpattern_fixed=True, is_curriculum_lr=True)
    env.goal_pattern = GOAL_PATTERN
    env.difficulty = 30
    state = env.reset()

    path = solver.solve(state.state,
                        GOAL_PATTERN,
                        {},
                        {},
                        toggle_result=False)
    if path is not None:
        r_path = rm_ras_actions(stats=path, )
        print(r_path)
        drawer = RackStatePlot(GOAL_PATTERN)
        fig = drawer.plot_states(r_path, row=12)
        p = Plot(fig=fig)
        p.save_fig("test.jpg", dpi=300)
        print(f"number of path len is {len(r_path)}/{len(path)}")
