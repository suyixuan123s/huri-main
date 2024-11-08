import numpy as np
import torch

from ops import GroupRepresentations


def closed_group(elements):
    """
    Checks, for the input list elements, if they are closed under composition,
    i.e. for h, g \in G, is h.g \in G.
    """
    new_elements = []
    order = len(elements)
    caley_table = np.zeros((order, order))
    for i, i_element in enumerate(elements):
        for j, j_element in enumerate(elements):
            new_element = torch.matmul(i_element, j_element)
            idx = get_element_index(new_element, elements)
            if idx is None:
                new_elements.append(new_element)
            else:
                caley_table[i][j] = idx
    if len(new_elements) > 0:
        return False
    else:
        return True


def get_element_index(el, elements):
    """
    get index of el in list elements
    """
    for idx, element in enumerate(elements):
        diff = torch.sum(torch.abs(el - element))
        if diff.item() < 1e-8:
            return idx
    return None


def rack_state_group_representations(rack_size, ):
    idx = np.arange(np.prod(rack_size)).reshape(rack_size)
    representations = [
        torch.tensor(np.flipud(idx).copy(), dtype=torch.float32),
        torch.tensor(np.fliplr(idx).copy(), dtype=torch.float32),
        torch.tensor(np.flipud(np.fliplr(idx)).copy(), dtype=torch.float32),
    ]

    return GroupRepresentations(representations, "RackStateRoll")


# def rack_state_


if __name__ == "__main__":
    a = rack_state_group_representations((3, 5))

    closed_group(a.representations)