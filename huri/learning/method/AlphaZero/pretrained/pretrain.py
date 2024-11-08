import os
import logging
import hydra
import timeit

import numpy as np
import torch
import copy

import threading
import multiprocessing as mp
from huri.learning.method.AlphaZero.network import AlphaZeroNet
from huri.learning.env.rack_v3 import create_env
from huri.learning.utils import select_device
from huri.learning.method.AlphaZero.pipeline import (run_data_collector,
                                                     run_self_play,
                                                     run_training,
                                                     run_evaluation,
                                                     calc_loss,
                                                     Transition)
from huri.learning.method.AlphaZero.utils import delete_all_files_in_directory
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import huri.core.file_sys as fs


class RackDataset(Dataset):
    def __init__(self, dataset, toggle_debug=False):
        self.dataset = dataset
        # self.transform = transforms.Compose([transforms.ToTensor()])
        self._toggle_debug = toggle_debug

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        state, pi_probs, values = self.dataset[idx]
        return Transition(state=state.astype(np.float32),
                          pi_prob=pi_probs.astype(np.float32),
                          value=values.astype(np.float32))


@hydra.main(config_path='../params', config_name='params_pretrained', version_base='1.3')
def train(cfg):
    delete_all_files_in_directory(r'E:\huri_shared\huri\learning\method\AlphaZero\run')
    device = select_device(device=cfg['device'])
    # init seed
    torch.manual_seed(cfg['torch_seed'])
    random_state = np.random.RandomState(cfg['numpy_seed'])

    # define env
    env_meta = create_env(rack_sz=cfg['env']['rack_sz'],
                          num_tube_class=cfg['env']['num_tube_class'],
                          seed=cfg['env']['seed'],
                          toggle_curriculum=cfg['env']['toggle_curriculum'],
                          toggle_goal_fixed=cfg['env']['toggle_goal_fixed'])
    # env_meta.scheduler.state_level = 1
    # input_shape =
    # num_actions =
    board = env_meta.rack_size
    input_shape = env_meta.observation_space_dim
    num_actions = env_meta.action_space_dim

    # network
    network = AlphaZeroNet(input_shape, num_actions, num_res_block=20, num_filters=128, num_fc_units=128)
    network.to(device)
    # Optimizer
    optimizer = torch.optim.SGD(network.parameters(),
                                lr=cfg['optim']['lr'],
                                momentum=cfg['optim']['momentum'],
                                weight_decay=cfg['optim']['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=tuple(cfg['optim']['lr_decay_milestones']),
                                                        gamma=cfg['optim']['lr_decay_gamma'])

    def collate_fn(samples):
        transposed = zip(*samples)
        stacked = [np.stack(xs, axis=0) for xs in transposed]
        return Transition(*stacked)

    train_kwargs = {'batch_size': cfg['train_batch'], 'shuffle': True, 'collate_fn': collate_fn}
    test_kwargs = {'batch_size': cfg['test_batch'], 'shuffle': True, 'collate_fn': collate_fn}
    dataset = fs.load_pickle(cfg['training_dataset_path'])
    training_data, testing_data = train_test_split(dataset, test_size=0.2, random_state=random_state)
    train_loader = torch.utils.data.DataLoader(RackDataset(training_data, toggle_debug=False), **train_kwargs)
    test_loader = torch.utils.data.DataLoader(RackDataset(testing_data), **test_kwargs)

    epochs = cfg['epoch']
    for epoch in range(epochs):
        loss = 0
        for batch_idx, transitions in enumerate(train_loader):
            # reshape mini-batch data to [N, 784] matrix
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            policy_loss, value_loss = calc_loss(network, device, transitions, False)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # add the mini-batch training loss to epoch loss
            loss += loss.item()

            if batch_idx % 1000 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        # compute the epoch training loss
        loss = loss / len(train_loader)

        test_loss = 0
        with torch.no_grad():
            for transitions in test_loader:
                policy_loss, value_loss = calc_loss(network, device, transitions, False)
                loss = policy_loss + value_loss
                test_loss += loss.item()  # sum up batch loss

        test_loss /= len(test_loader.dataset)
        # display the epoch training loss
        print("epoch : {}/{}, test_loss = {:.6f}".format(epoch + 1, epochs, test_loss))

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
        torch.save(network.state_dict(), "test_encoder.pt")


if __name__ == "__main__":
    train()
