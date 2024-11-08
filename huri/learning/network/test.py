import torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
a = conv.state_dict()

