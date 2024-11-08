import time

from huri.learning.env.rack_v2.utils import mask_ucbc, mask_crcl, mask_ul, mask_ur, mask_bl, mask_br
from huri.learning.env.rack_v2.utils import ss
from huri.learning.utils import select_device
import numpy as np
import torch

device = select_device()
node = np.array([[0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 2, 2, 0, 0, 2],
                 [0, 0, 0, 0, 2, 0, 2, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
                 [2, 0, 0, 0, 0, 2, 2, 0, 0, 0]])
tnode = torch.tensor(node, dtype=torch.float32, device=device)
tnode = tnode.unsqueeze(0).unsqueeze(0)
tmask_ucbc = torch.tensor(mask_ucbc, dtype=torch.float32, device=device)
tmask_ucbc = tmask_ucbc.unsqueeze(0).unsqueeze(0)


def cal_conv_np():
    return ss.correlate2d(node, mask_ucbc)[1:-1, 1:-1]


def cal_conv_tensor():
    return torch.nn.functional.conv2d(tnode, tmask_ucbc, padding=1)


def cal_conv_np2tensor():
    r = tnode.squeeze(0).squeeze(0).cpu().numpy()
    b = ss.correlate2d(r, mask_ucbc)[1:-1, 1:-1]
    return torch.tensor(b, dtype=torch.float32, device=device)


if __name__ == "__main__":
    a = time.time()
    cal_conv_tensor()
    b = time.time()
    print((b - a) * 1000)
