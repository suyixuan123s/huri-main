""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230728osaka

"""
from itertools import combinations, permutations, product

import cv2
import numpy as np
import torch

from huri.components.utils.img_utils import combine_images
from huri.learning.env.rack_v3 import create_env, RackState, RackStatePlot
from huri.learning.env.rack_v3.env import from_action
from huri.learning.method.APEX_DQN.distributed.network import DDQN


def generate_unique_combinations_classification():
    elements = [0, 1]
    combinations = np.array(list(product(elements, repeat=9)))
    unique_combinations = np.unique(combinations, axis=0)
    classification = {}
    for combination in unique_combinations:
        num_ones = np.sum(combination)
        if num_ones not in classification:
            classification[num_ones] = []
        classification[num_ones].append(combination.reshape((3, 3)))

    return classification


def generate_permutations_2x3x3(list_of_arrays):
    # Get all possible permutations of two elements from the list
    permutations_list = list(permutations(list_of_arrays, 2))

    # Convert the list of permutations into a 2x3x3 NumPy array
    result_array = np.array(permutations_list).reshape((-1, 2, 3, 3))

    return result_array


def plot_heat_map(action_index, state_index, value_state_values):
    """
    Plots a heat map using matplotlib.

    Parameters:
        action_index (list): List containing the action indices.
        state_index (list): List containing the state indices.
        value_state_values (2D numpy array): A 2D array representing the values of states for different actions.

    Returns:
        None
    """

    # Create a meshgrid to represent the state indices as a grid
    X, Y = np.meshgrid(state_index, action_index)

    # Transpose the value-state values array for plotting correctly

    # Create the figure and the heat map using matplotlib's pcolormesh function
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(X, Y, value_state_values.T, cmap='viridis', shading='auto',
                   vmin=value_state_values.min(), vmax=value_state_values.max())
    plt.plot(X.flat, Y.flat, 'o', color='m')
    # Add a colorbar to show the values corresponding to colors in the heat map
    plt.colorbar()

    # Set axis labels and title
    plt.xlabel('State Index')
    plt.ylabel('Action Index')
    plt.title('Heat Map of State Values for Different Actions')

    # Show the plot
    plt.show()


def vv(a1, a2, rack_sz):
    return np.ravel_multi_index(a1, rack_sz), np.ravel_multi_index(a2, rack_sz)


if __name__ == '__main__':
    rack_sz = (3, 3)
    env = create_env(rack_sz=rack_sz,
                     num_tube_class=1,
                     num_history=1, )
    classification_dict = generate_unique_combinations_classification()

    input_shape = env.observation_space_dim
    num_actions = env.action_space_dim
    network = DDQN(input_shape, num_actions, num_filters=10, num_fc_units=128)
    network.load_state_dict(
        torch.load(r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed\run\data\model_best.chkpt')[
            'dqn_state_dict'])
    network.to('cuda')
    rstates = []
    img_list = []
    s_g = generate_permutations_2x3x3(classification_dict[1])
    vs = []
    with torch.no_grad():
        qqq = [_ for _ in s_g[:8]] + [_[[1, 0]] for _ in s_g[:8]]
        for i in qqq:
            rsp = RackStatePlot(i[1], )
            img_list.append(rsp.plot_states([i[0]]).get_img())
            Q = network(torch.tensor(i, dtype=torch.float32).unsqueeze(0).to('cuda')).cpu().numpy()[0]
            mask = np.ones(len(Q), dtype=bool)
            mask[RackState(i[0]).feasible_action_set] = 0
            Q[mask] = 0
            print(vv(*from_action(rack_sz, Q.argmax()), rack_sz=rack_sz), Q.max())
            vs.append(Q)
    plot = combine_images(img_list, columns=8)
    cv2.imwrite('t.png', plot)
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("TkAgg")

    plot_heat_map(action_index=np.arange(num_actions),
                  state_index=np.arange(len(vs)),
                  value_state_values=np.array(vs))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    colors = ['r', 'g', 'b', 'y']
    yticks = np.arange(len(vs))
    plt.xticks(np.arange(num_actions), [vv(*from_action(rack_sz, _),
                                           rack_sz) for _ in np.arange(num_actions)])
    for _, k in enumerate(yticks):
        xs = np.arange(num_actions)
        ax.bar(xs, vs[_], zs=k, zdir='y', alpha=0.8)

    ax.set_xlabel('Action')
    ax.set_ylabel('State')
    ax.set_zlabel('Q(s,a)')
    # On the y-axis let's only label the discrete values that we have data for.
    ax.set_yticks(yticks)

    plt.show()
