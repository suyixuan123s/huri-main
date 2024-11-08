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
    def __init__(self, depth=3, layer=50):
        super().__init__()
        self.depth = depth
        self.l1 = nn.Linear(1, layer)
        for i in range(2, depth):
            setattr(self, f'l{i}', nn.Linear(layer, layer))
        setattr(self, f'l{depth}', nn.Linear(layer, 1))

    def forward(self, x):
        for i in range(1, self.depth):
            x = F.silu(getattr(self, f'l{i}')(x))
        x = getattr(self, f'l{self.depth}')(x)
        return x


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    test_x = [0, 13]
    train_x = [0, 8]
    training_data = 200
    batch_size = 0
    epoch = 10000
    x = np.random.choice(np.arange(train_x[0], train_x[1], .01), training_data, replace=False)
    # x = np.arange(test_x[0], test_x[1], .01)

    print(x)
    y_func = lambda x: np.sin(1.31432423 * np.pi * x) - 1.31432423 * np.pi * np.sin(x) + np.sin(4 * np.pi * x) + \
                       np.sin(2 * np.pi * x) + np.sin(1.1111111 * np.pi * x) + np.sin(0.3332 * np.pi * x)
    y = y_func(x)

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    plt.scatter(x, y, s=120)
    # plt.show()

    x = torch.tensor(x, dtype=torch.float32, device='cuda')
    y = torch.tensor(y, dtype=torch.float32, device='cuda')
    critic = torch.nn.MSELoss()
    nn = SinNet().to('cuda')
    optimizer = torch.optim.Adam(nn.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    if batch_size <= 0:
        for epoch in range(epoch):
            optimizer.zero_grad()
            y_pred = nn(x)
            loss = critic(y_pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()  # Step the learning rate scheduler

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
        y_pred = \
            nn(torch.arange(test_x[0], test_x[1], .01, dtype=torch.float32, device='cuda').reshape(-1,
                                                                                                   1)).detach().cpu().numpy().reshape(
                1, -1)[0]
    x = np.arange(test_x[0], test_x[1], .01)
    y_real = y_func(x)

    plt.scatter(x, y_real, s=20)
    plt.scatter(x, y_pred, s=20)
    plt.show()
