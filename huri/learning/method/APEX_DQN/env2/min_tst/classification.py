import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim


class C(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.feature_len = kwargs["feature_len"]
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=32
        )
        self.encoder_output_layer = nn.Linear(
            in_features=32, out_features=16
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=16, out_features=32
        )
        self.decoder_output_layer = nn.Linear(
            in_features=32, out_features=kwargs["output_shape"]
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

    from env_tst import env, num_tube_classes, rack_size, action_space_dim, observation_space_dim

    state_dim = np.prod(rack_size)

    device = select_device()
    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = C(input_shape=state_dim * 2, output_shape=state_dim ** 2, feature_len=state_dim).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

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
            state, goal, label = self.dataset[idx]
            res = state.copy()
            res[(state - goal) <= 0] = 0
            res = res.astype(np.float32)

            return state.astype(np.float32), goal.astype(np.float32), res.astype(np.float32)


    dataset = fs.load_pickle("demo_training_data_1_5.pkl")
    training_data, testing_data = train_test_split(dataset, test_size=0.2, random_state=21)
    train_loader = torch.utils.data.DataLoader(RackDataset(training_data, toggle_debug=False), **train_kwargs)
    test_loader = torch.utils.data.DataLoader(RackDataset(testing_data), **test_kwargs)

    epochs = 100
    for epoch in range(epochs):
        loss = 0
        for batch_idx, (state_features, goal_features, result_features) in enumerate(train_loader):
            # reshape mini-batch data to [N, 784] matrix
            state_features = state_features.to(device)
            goal_features = goal_features.to(device)
            result_features = result_features.to(device)
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs, _ = model(state_features, goal_features)

            # compute training reconstruction loss
            train_loss = criterion(outputs, result_features.reshape(result_features.shape[0], model.feature_len))

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
        loss = loss / len(train_loader)

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
        torch.save(model.state_dict(), "test_encoder.pt")
