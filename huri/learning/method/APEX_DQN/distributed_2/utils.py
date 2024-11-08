""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20231026osaka

"""
import torch


def to_onehot(tensor, num_categories=2):
    """
    Convert a batch of 2D tensors with integer categories to one-hot representation.

    Parameters:
    - tensor: A (batch, 1, N, M) torch Tensor.
    - num_categories: Total number of categories.

    Returns:
    - A (batch, N, M, num_categories) torch Tensor in one-hot format.
    """

    # Get the shape of the input tensor
    batch_size, _, N, M = tensor.shape

    # Create a one-hot tensor of zeros of appropriate size
    # one_hot = torch.zeros(batch_size, N, M, num_categories, device=tensor.device)
    one_hot = torch.zeros(batch_size, num_categories + 1, N, M, dtype=tensor.dtype, device=tensor.device)

    # Fill in the ones at the right indices
    one_hot.scatter_(1, tensor.long(), 1)

    return one_hot[:, 1:]


if __name__ == '__main__':
    tensor = torch.tensor([[[[1, 2], [0, 1]]]], device="cuda:0", dtype=torch.float32)
    print(tensor)
    print(tensor.shape)
    num_categories = 2
    print(to_onehot(tensor, num_categories))
