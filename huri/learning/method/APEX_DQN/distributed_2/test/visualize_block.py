""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20231002osaka

"""
import numpy as np
import matplotlib.pyplot as plt
from huri.components.utils.matlibplot_utils import Plot


# plot the heatmap of the output of the conv block
def visualize_lattice(lattice: np.ndarray, n_col=6, scale=3) -> Plot:
    lattice = np.squeeze(lattice)
    if len(lattice.shape) == 2:
        lattice = lattice[None, ...]
    fig, axs = plt.subplots(int(np.ceil(lattice.shape[0] / n_col)), min(n_col, lattice.shape[0]),
                            figsize=(n_col * scale, int(np.ceil(lattice.shape[0] / n_col)) * scale))
    if not isinstance(axs, np.ndarray):
        axs = np.asarray([axs])
    axs = axs.flatten()
    for i in range(lattice.shape[0]):
        im = axs[i].imshow(lattice[i])
        fig.colorbar(im, ax=axs[i], ticks=[-1, 0, 1])
        axs[i].set_title(f"Subplot {i + 1}")
        axs[i].axis('off')

    return Plot(fig=fig)


if __name__ == '__main__':
    import torch
    import cv2

    x, conv_s, filter = torch.load('debug_data/conv_block_out.pt')
    # visulize x using matplotlib
    x = x[0, 0, ...].numpy()
    print(x)
    img = visualize_lattice(x).get_img()

    conv_s = conv_s[0, ...].numpy()

    img2 = visualize_lattice(conv_s).get_img()
    img3 = visualize_lattice(filter['0.weight'].squeeze(dim=1)).get_img()

    cv2.imshow('x', img)
    cv2.imshow('x2', img2)
    cv2.imshow('x3', img3)
    cv2.waitKey(0)
