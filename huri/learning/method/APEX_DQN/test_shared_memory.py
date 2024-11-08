import copy
import time

import torch
import torch.multiprocessing as mp
from huri.learning.utils import select_device, LOGGER


def _test(model, shared_net, toggle):
    if toggle:
        torch.nn.init.uniform_(model.weight)
    print(model.weight)
    shared_net.load_state_dict(model.state_dict())
    print("tttt",shared_net.weight)


def _p1(model, shared_net):
    model.load_state_dict(shared_net.state_dict())
    print(model.weight)


if __name__ == "__main__":
    device = select_device()
    shared_t = torch.tensor([100.], dtype=torch.float32, device=device)

    shared_t.share_memory_()
    model = torch.nn.Linear(10, 10)
    model.to(device=device)
    model.eval()
    model.share_memory()
    # print(model.weight)
    num_processes = 5
    processes = []
    p = mp.Process(target=_test, args=(torch.nn.Linear(10, 10).to(device=device), model, True))
    p.start()
    processes.append(p)
    time.sleep(1)
    p = mp.Process(target=_p1, args=(torch.nn.Linear(10, 10).to(device=device), model))
    p.start()
    processes.append(p)

    # for rank in range(num_processes):
    #     p = mp.Process(target=_test, args=(model,False))
    #     p.start()
    #     processes.append(p)

    for p in processes:
        p.join()
