import numpy as np
import itertools
import copy


def _get_relation(centers: np.ndarray, data_length: int):
    relationship = [[] for i in range(data_length)]
    node_relations = itertools.permutations(range(data_length), 2)

    for v in node_relations:
        node0_idx, node1_idx = v
        node0_center = centers[node0_idx]
        node1_center = centers[node1_idx]
        vec = node1_center - node0_center
        angle = np.arctan2(vec[0], vec[1])
        relationship[node0_idx].append((node1_idx, angle))

    return relationship


def directional_relationship(labels: list, centers: list, cal_center=False):
    labels = list(labels)
    if cal_center:
        centers = [np.average(np.asarray(i), axis=0) for i in centers]
    else:
        centers = list(centers)
    relationship_raw = _get_relation(np.array(centers), len(labels))
    dtype = [("index", np.int), ("weight", float)]
    relationship = np.array([np.sort(np.asarray(val, dtype=dtype), order="weight") for val in relationship_raw])
    return relationship['index']


def find_tube_pos(tube_places_dict, relation_constraints: list):
    # while True:
    tube_labels = list(tube_places_dict.keys())
    tube_hole_ids = [0] * len(tube_labels)
    constraints_label, constrains_relationmatrix = relation_constraints
    constrains_relationmatrix = constrains_relationmatrix[
        np.array([tube_labels.index(v) for v in constraints_label])
    ].copy()

    confidence_threshold = 0.7
    outliner_threshold = 0.1

    unconfidence_label_index = []
    # find confidence place
    for ii, key in enumerate(tube_labels):
        pos, prob = tube_places_dict[key]
        place_index = None
        # Case 1: if only one place: select the place
        if len(pos) == 1:
            place_index = pos[0]
        # Case 2: if the place larger than the confidence threshold.

        if 2 * np.max(prob) - np.sum(prob) > 0:
            place_index = pos[np.argmax(prob)]
        if place_index is not None:
            tube_hole_ids[ii] = place_index
        else:
            print(f"{tube_labels}, eoor")
            raise Exception(f"{tube_labels} cannot works well: {pos}, probability{prob}")
    # Deal with the uncertain tubes
    # for uncertain_label in uncertain_labels:
    #     #Case 3: exit multiple probability
    #     possible_place = pos[np.where(prob > outliner_threshold)]
    #     if len(confident_place) < 1:
    #         raise Exception("Possible place calculation Error")
    return tube_labels, tube_hole_ids
    if 0 not in tube_hole_ids:
        return True, tube_hole_ids
    tmp = {}
    for ii in unconfidence_label_index:
        pos, prob = tube_places_dict[tube_labels[ii]]
        # try the place with a decending order
        pos_d = pos[prob.argsort()[::-1]]
        tmp[ii] = pos_d

    possiblecombination = itertools.product(
        *tmp.values()
    )

    for p in possiblecombination:
        tubecenter_tmp = copy.deepcopy(tube_hole_ids)
        for idx, unconfidence_label_idx in enumerate(unconfidence_label_index):
            tubecenter_tmp[unconfidence_label_idx] = p[idx]
        print(directional_relationship(tube_labels, [np.array([c[1], c[0]]) for c in tubecenter_tmp]))
    print("-----")
    print(constrains_relationmatrix)


def locate_tube_in_rack(tube_places_dict):
    # while True:
    tube_labels = list(tube_places_dict.keys())
    tube_hole_ids = [0] * len(tube_labels)

    # find confidence place
    for ii, key in enumerate(tube_labels):
        pos, prob = tube_places_dict[key]
        place_index = None
        # Case 1: if only one place: select the place
        if len(pos) == 1:
            place_index = pos[0]
        # Case 2: if the place larger than the confidence threshold.

        if 2 * np.max(prob) - np.sum(prob) > 0:
            place_index = pos[np.argmax(prob)]
        if place_index is not None:
            tube_hole_ids[ii] = place_index
        else:
            print(f"{tube_labels}, eoor")
            raise Exception(f"{tube_labels} cannot works well: {pos}, probability{prob}")

    return tube_labels, tube_hole_ids


def _iterative_solve(tube_labels, tube_places_dict, tube_centers):
    pass

# confidence_threshold = 0.6
# tube_type = TubeType.parse_string(label)
# if len(possible_place) == 1:
#     place_index = possible_place[0]
#     TubeRack_Light_Thinner.insert_tube(place_index, tube_type)
#
# confident_place = np.where(probability > confidence_threshold)[0]
# if len(confident_place) > 0:
#     place_index = possible_place[confident_place[0]]
#     TubeRack_Light_Thinner.insert_tube(place_index, tube_type)
