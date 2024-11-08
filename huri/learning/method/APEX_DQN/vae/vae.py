"""
Variational encoder model, used as a visual model
for our model of the world.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """ VAE decoder """

    def __init__(self, img_channels, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc1 = nn.Linear(latent_size, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 7, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 9, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, img_channels, 10, stride=2)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        reconstruction = F.sigmoid(self.deconv3(x))
        return reconstruction


class Encoder(nn.Module):  # pylint: disable=too-many-instance-attributes
    """ VAE encoder """

    def __init__(self, img_channels, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        # self.img_size = img_size
        self.img_channels = img_channels

        self.conv1 = nn.Conv2d(img_channels, 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1)

        self.fc_mu = nn.Linear(81 * 128, latent_size)
        self.fc_logsigma = nn.Linear(81 * 128, latent_size)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma


class VAE(nn.Module):
    """ Variational Autoencoder """

    def __init__(self, img_channels, latent_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(img_channels, latent_size)

        # Decoder layers
        # self.decoder_fc1 = nn.Linear(latent_size, 1280)
        # self.decoder_fc2 = nn.Linear(1280, 2500)

        self.decoder = Decoder(img_channels, latent_size)

    def forward(self, x):  # pylint: disable=arguments-differ
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        recon_x = self.decoder(z)
        return recon_x, mu, logsigma


if __name__ == '__main__':
    a = torch.zeros((32, 1, 50, 50))
    v = VAE(1, 3)
    recon_x, mu, logsigma = v(a)
    print(recon_x.shape)
