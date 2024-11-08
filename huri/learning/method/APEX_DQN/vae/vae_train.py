""" Training VAE """
import argparse
from os.path import join, exists
from os import mkdir
from pathlib import Path

import huri.core.file_sys as fs
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
from learning import RackDataset, EarlyStopping

from vae import VAE


def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)


# latent space size
LSIZE = 32

parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--logdir', type=str, help='Directory where results are logged')
parser.add_argument('--noreload', action='store_false',
                    help='Best model is not reloaded if specified')
parser.add_argument('--nosamples', action='store_true',
                    help='Does not save samples during training if specified')

args = parser.parse_args()
cuda = torch.cuda.is_available()

torch.manual_seed(123)
# Fix numeric divergence due to bug in Cudnn
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if cuda else "cpu")

dataset = fs.load_pickle("ae_data_5_10.pkl")
training_data, testing_data = train_test_split(dataset, test_size=0.2, random_state=21)

train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
test_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
train_loader = torch.utils.data.DataLoader(RackDataset(training_data, toggle_debug=False), **train_kwargs)
test_loader = torch.utils.data.DataLoader(RackDataset(testing_data), **test_kwargs)

model = VAE(1, LSIZE).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=30)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logsigma):
    """ VAE loss function """
    BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD


def train(epoch):
    """ One training epoch """
    model.train()
    train_loss = 0
    for batch_idx, (state, abs_state) in enumerate(train_loader):
        state = state.to(device).unsqueeze(1) / 3
        abs_state = abs_state.to(device).unsqueeze(1) / 3
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(abs_state)
        loss = loss_function(recon_batch, abs_state, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # if batch_idx % 8000 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(state), len(train_loader.dataset),
        #                100. * batch_idx / len(train_loader),
        #                loss.item() / len(state)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test():
    """ One test epoch """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for (state, abs_state) in test_loader:
            state = state.to(device).unsqueeze(1) / 3
            abs_state = abs_state.to(device).unsqueeze(1) / 3
            recon_batch, mu, logvar = model(abs_state)
            test_loss += loss_function(recon_batch, abs_state, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


# check vae dir exists, if not, create it
vae_dir = Path('run')
if not vae_dir.exists():
    vae_dir.mkdir()
    vae_dir.joinpath("samples").mkdir()

reload_file = vae_dir.joinpath('best.tar')
if not args.noreload and reload_file.exists():
    state = torch.load(reload_file)
    print("Reloading model at epoch {}"
          ", with test error {}".format(
        state['epoch'],
        state['precision']))
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    earlystopping.load_state_dict(state['earlystopping'])

cur_best = None

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test_loss = test()
    scheduler.step(test_loss)
    earlystopping.step(test_loss)

    # checkpointing
    best_filename = join(vae_dir, 'best.tar')
    filename = join(vae_dir, 'checkpoint.tar')
    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss

    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'precision': test_loss,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'earlystopping': earlystopping.state_dict()
    }, is_best, filename, best_filename)

    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(epoch))
        break
