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
from huri.learning.env.rack_v2.env import RackArrangementEnv, RackStatePlot, RackState

import huri.core.file_sys as fs
# from huri.learning.network.d3qn_attention import DuelingDQN
from huri.learning.method.APEX_DQN.env2.min_tst2.dqn_model import DuelingDQNMini
from huri.learning.utils import select_device, LOGGER
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split

import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Decoder, self).__init__()
        self.decoder_hidden_layer = nn.Linear(
            in_features=input_shape, out_features=256
        )
        self.decoder_output_layer = nn.Linear(
            in_features=256, out_features=output_shape
        )

    def forward(self, code):
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (state, goal, action, label, feasible_acts) in enumerate(train_loader):
        state, goal, action, label, feasible_acts = state.to(device), goal.to(device), action.to(device), label.to(
            device), feasible_acts.to(device)

        optimizer.zero_grad()
        q = model(state, goal)
        # feasible_act = feasible_act_decoder(feasible_act_feature)
        q_loss = F.smooth_l1_loss(q.gather(1, action.reshape(-1, 1)), label.reshape(-1, 1))
        # feasible_act_loss = F.mse_loss(feasible_act, feasible_acts)
        # loss = q_loss + feasible_act_loss
        loss = q_loss
        loss.backward()
        # https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for (state, goal, action, label, feasible_acts) in test_loader:
            state, goal, action, label = state.to(device), goal.to(device), action.to(device), label.to(device)
            output = model(state, goal)
            test_loss += F.smooth_l1_loss(output.gather(1, action.reshape(-1, 1)), label.reshape(-1, 1),
                                    reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)
            # get the index of the max log-probability
            correct += pred.eq(action.reshape(-1, 1)).sum().item()
            # pred = output.flatten()
            # correct += pred.eq(label).sum().item()

    test_loss /= len(test_loader.dataset)

    print(('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))))


class RackDataset(Dataset):
    def __init__(self, dataset, toggle_debug=False):
        self.dataset = dataset
        self.transform = transforms.Compose([transforms.ToTensor()])
        self._toggle_debug = toggle_debug

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        state, goal, label, feasible_act = self.dataset[idx]
        # label
        state = state.astype(np.float32)
        goal = goal.astype(np.float32)
        action = label.argmax()
        label = label.astype(np.float32)
        f = np.zeros_like(label, dtype=np.float32)
        f[feasible_act] = 1

        if self._toggle_debug:
            print("state:", state)
            print("goal:", goal)
            print(env._expr_action(label.argmax()))
            print(label.argmax())
            print(label.max())
            plot([state], goal)

        return state, goal, action, label.max(), f


def plot(states, goal):
    drawer = RackStatePlot(goal)
    fig = drawer.plot_states(states).get_img()
    cv2.imshow("window", fig)
    cv2.waitKey(0)


parser = ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                    help='learning rate (default: 1e-2)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=True,
                    help='For Saving the current Model')
args = parser.parse_args()

from env_tst import env, num_tube_classes, rack_size, action_space_dim, observation_space_dim

# env.set_goal_pattern(GOAL_PATTERN_5x10)
action_vec = np.zeros(action_space_dim, dtype=np.float32)
#
device = select_device()

net = DuelingDQNMini(obs_dim=observation_space_dim, action_dim=action_space_dim, num_classes=num_tube_classes).to(
    device)
# feasible_act_decoder = Decoder(input_shape=net.feasible_act_encoder.encoder_output_layer.out_features,
#                                output_shape=action_space_dim).to(device)
# eval_net_path = fs.Path("test_encoder3.pt")
# pre_trained = torch.load(str(eval_net_path))
# pre_trained_dict = {_[8:]:pre_trained[_] for _ in pre_trained.keys() if _.startswith('encoder.')}
# net.feasible_act_encoder.load_state_dict(pre_trained_dict)

# net.load_state_dict(torch.load("transfer_learning_weight.pt"))
optimizer = optim.Adam(net.parameters(), lr=args.lr)

scheduler = StepLR(optimizer, step_size=30, gamma=args.gamma)

dataset = fs.load_pickle("demo_training_data_3_6.pkl")

training_data, testing_data = train_test_split(dataset, test_size=0.2, random_state=21)

# for _ in training_data:
#     state, goal, action = _
#     print(action)
#     plot([state], goal)

train_kwargs = {'batch_size': args.batch_size}
test_kwargs = {'batch_size': args.test_batch_size}
train_loader = torch.utils.data.DataLoader(RackDataset(training_data, toggle_debug=False), **train_kwargs)
test_loader = torch.utils.data.DataLoader(RackDataset(testing_data), **test_kwargs)

for epoch in range(1, args.epochs + 1):
    train(args, net, device, train_loader, optimizer, epoch)
    test(net, device, test_loader)
    scheduler.step()
    if args.save_model:
        torch.save(net.state_dict(), "transfer_learning_weight.pt")
