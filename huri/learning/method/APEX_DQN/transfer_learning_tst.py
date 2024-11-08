import copy
import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
#
from huri.learning.env.arrangement_planning_rack_gc_apex.env import RackArrangementEnv, RackStatePlot

# from huri.learning.network.d3qn_attention import DuelingDQN
from huri.learning.network.d3qn_attention import DuelingDQN2
from huri.learning.utils import select_device, LOGGER
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split



def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for (state, goal, label) in test_loader:
            state, goal, label = state.to(device), goal.to(device), label.to(device)
            output = model.forward_value(state, goal)
            test_loss += F.mse_loss(output, label.reshape(-1, 1), reduction='sum').item()  # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True)
            # get the index of the max log-probability
            # correct += pred.eq(label.argmax(dim=1, keepdim=True)).sum().item()
            pred = output.flatten()
            correct += pred.eq(label).sum().item()

    test_loss /= len(test_loader.dataset)

    print(('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))))


import huri.core.file_sys as fs

dataset = fs.load_pickle("dataset_3_5.pkl")

training_data, testing_data = train_test_split(dataset, test_size=0.2, random_state=21)

# for _ in training_data:
#     state, goal, action = _
#     print(action)
#     plot([state], goal)

