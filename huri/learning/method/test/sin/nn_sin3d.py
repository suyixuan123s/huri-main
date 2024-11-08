""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230718osaka

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class SinNet(nn.Module):
    def __init__(self, depth=5, layer=40):
        super().__init__()
        self.depth = depth
        self.l1 = nn.Linear(2, layer)
        for i in range(2, depth):
            setattr(self, f'l{i}', nn.Linear(layer, layer))
        setattr(self, f'l{depth}', nn.Linear(layer, 1))

    def forward(self, x):
        for i in range(1, self.depth):
            x = F.leaky_relu(getattr(self, f'l{i}')(x))
        x = getattr(self, f'l{self.depth}')(x)
        return x


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    train_x = [0, 4]
    test_x = [0, 8]
    training_data = 200
    batch_size = 0
    epoch = 50000
    y_func = lambda x1, x2: np.sin(np.sqrt(x1 ** 2 + x2 ** 2)) + 2 * np.sin(
        2 * np.sqrt(x1 ** 2 + x2 ** 2)) + 3 * np.sin(3 * np.sqrt(x1 ** 2 + x2 ** 2))

    x1 = np.arange(test_x[0], test_x[1], .1)
    x2 = np.arange(test_x[0], test_x[1], .1)
    grid = np.dstack(np.meshgrid(x1, x2)).reshape((-1, 2))
    x1 = grid[:, 0]
    x2 = grid[:, 1]
    y = y_func(x1, x2)
    ax.scatter(x1, x2, y, s=10)

    x1 = np.random.choice(np.arange(train_x[0], train_x[1], .01), training_data, replace=False)
    x2 = np.random.choice(np.arange(train_x[0], train_x[1], .01), training_data, replace=False)
    x1 = x1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    y = y_func(x1, x2)
    ax.scatter(x1, x2, y, s=100)
    # plt.show()

    x = torch.tensor(np.dstack((x1, x2)).reshape(-1, 2), dtype=torch.float32, device='cuda')
    y = torch.tensor(y, dtype=torch.float32, device='cuda')
    critic = torch.nn.MSELoss()
    nn = SinNet().to('cuda')
    optimizer = torch.optim.Adam(nn.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)
    if batch_size <= 0:
        for epoch in range(epoch):
            optimizer.zero_grad()
            y_pred = nn(x)
            loss = critic(y_pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn.parameters(), max_norm=1.0)
            optimizer.step()
            # scheduler.step()  # Step the learning rate scheduler

            print(f" Epoch {epoch}, Loss: {loss.item()}")
    else:
        num_batches = len(x) // batch_size
        for epoch in range(epoch):
            total_loss = 0.0
            nn.train()  # Set the model to training mode

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size

                x_batch = x[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]

                optimizer.zero_grad()
                y_pred = nn(x_batch)
                loss = critic(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(nn.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            average_loss = total_loss / num_batches
            # scheduler.step()  # Step the learning rate scheduler
            print(f" Epoch {epoch}, Loss: {average_loss}")

    with torch.no_grad():
        x1 = grid[:, 0]
        x2 = grid[:, 1]
        x1 = x1.reshape(-1, 1)
        x2 = x2.reshape(-1, 1)
        torch.tensor(np.dstack((x1, x2)).reshape(-1, 2), dtype=torch.float32, device='cuda')
        y_pred = \
            nn(torch.tensor(np.dstack((x1, x2)).reshape(-1, 2), dtype=torch.float32,
                            device='cuda')).detach().cpu().numpy().reshape(
                1, -1)[0]
        ax.scatter(x1, x2, y_pred, s=30)

    # x = np.arange(test_x[0], test_x[1], .01)
    # y_real = y_func(x)
    #
    # plt.scatter(x, y_real, s=20)
    # plt.scatter(x, y_pred, s=20)
    plt.show()
