import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F


class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.feature_len = kwargs["input_shape"]
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"] * 2, out_features=256
        )
        self.encoder_output_layer = nn.Linear(
            in_features=256, out_features=128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=256
        )
        self.decoder_output_layer = nn.Linear(
            in_features=256, out_features=kwargs["output_shape"]
        )

    def forward(self, state, goal):
        features = torch.cat(
            (state.reshape(state.shape[0], self.feature_len), goal.reshape(goal.shape[0], self.feature_len)), axis=1)
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed, code


if __name__ == "__main__":
    from huri.learning.env.rack_v2.env import RackArrangementEnv, GOAL_PATTERN_5x10
    from huri.learning.utils import select_device, LOGGER
    import huri.core.file_sys as fs
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    num_tube_classes = 1
    rack_size = (5, 10)
    action_space_dim = np.prod(rack_size) ** 2
    observation_space_dim = (rack_size[0], rack_size[1])
    observation_space_dim_nn = (1, *rack_size)
    env = RackArrangementEnv(rack_size=rack_size,
                             num_classes=num_tube_classes,
                             observation_space_dim=observation_space_dim,
                             action_space_dim=action_space_dim,
                             is_curriculum_lr=True,
                             is_goalpattern_fixed=False,
                             seed=888)
    input_dim = np.prod(rack_size)
    output_dim = np.prod(rack_size) ** 2

    device = select_device()
    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = AE(input_shape=input_dim, output_shape=output_dim).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # mean-squared error loss
    criterion = nn.MSELoss()

    batch_sz = 64

    train_kwargs = {'batch_size': batch_sz}
    test_kwargs = {'batch_size': batch_sz}


    class RackDataset(Dataset):
        def __init__(self, dataset, toggle_debug=False):
            self.dataset = dataset
            self.transform = transforms.Compose([transforms.ToTensor()])
            self._toggle_debug = toggle_debug

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            state, goal, act = self.dataset[idx]
            res = np.zeros(np.prod(state.shape) ** 2)
            res[act] = 1
            res = res.astype(np.float32)
            return state.astype(np.float32), goal.astype(np.float32), res.astype(np.float32)


    dataset = fs.load_pickle("ae_data_5_10.pkl")
    training_data, testing_data = train_test_split(dataset, test_size=0.2, random_state=21)
    train_loader = torch.utils.data.DataLoader(RackDataset(training_data, toggle_debug=False), **train_kwargs)
    test_loader = torch.utils.data.DataLoader(RackDataset(testing_data), **test_kwargs)

    epochs = 200
    for epoch in range(epochs):
        loss = 0

        # training
        # model.train()
        for batch_idx, (state_features, goal_features, result_features) in enumerate(train_loader):
            state_features = state_features.to(device)
            goal_features = goal_features.to(device)
            result_features = result_features.to(device)
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs, _ = model(state_features, goal_features)

            # compute training reconstruction loss
            train_loss = criterion(outputs, result_features.reshape(result_features.shape[0], output_dim))

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

            if batch_idx % 1000 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), train_loss.item()))
        # compute the epoch training loss

        # test
        # model.eval()
        test_loss = 0
        with torch.no_grad():
            for (state_features, goal_features, result_features) in test_loader:
                state_features = state_features.to(device)
                goal_features = goal_features.to(device)
                result_features = result_features.to(device)
                outputs, _ = model(state_features, goal_features)
                test_loss += F.mse_loss(outputs,
                                        result_features.reshape(result_features.shape[0],
                                                                output_dim),
                                        reduction='sum').item()  # sum up batch loss

        test_loss /= len(test_loader.dataset)
        # display the epoch training loss
        print("epoch : {}/{}, test_loss = {:.6f}".format(epoch + 1, epochs, test_loss))

        torch.save(model.state_dict(), "test_encoder2.pt")
