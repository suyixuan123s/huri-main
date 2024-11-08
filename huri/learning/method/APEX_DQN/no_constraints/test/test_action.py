import numpy as np


def action_space_info(obs_dim):
    action_dim = np.prod(obs_dim)
    compressed_abs_obs_dim = int((action_dim - 1) * action_dim / 2) + action_dim
    print(compressed_abs_obs_dim)
    return np.log2(compressed_abs_obs_dim)

print(np.log2(2**31))
print(action_space_info((5, 5)))
print(2**(8.344295907915816))
