"""
cVAE model, based on ANAM (046202), Technion.
"""

import torch
import torch.nn as nn


def labels_to_one_hots(batch, num_classes=10):
    """
    Converts batch of integes numbers to one-hot vector given the vector length
    :param batch: batch of values to convert
    :param num_classes: length of the vector
    :return: one_hot_batch
    """
    one_hot_batch = torch.zeros(batch.size(0), num_classes).to(batch.device)
    for i in range(batch.size(0)):
        one_hot_batch[i, int(batch[i].data.cpu().item())] = 1
    return one_hot_batch


# reparametrization trick
def reparameterize(mu, logvar, device=torch.device("cpu")):
    """
    This function applies the reparameterization trick:
    z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
    :param mu: mean of x
    :param logvar: log variance of x
    :param device: device to perform calculations on
    :return z: the sampled latent variable
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(device)
    return mu + eps * std


# encoder - Q(z|X)
class VaeEncoder(torch.nn.Module):
    """
       This class builds the encoder for the VAE
       :param x_dim: input dimensions
       :param hidden_size: hidden layer size
       :param z_dim: latent dimensions
       :param device: cpu or gpu
       """

    def __init__(self, x_dim=1 * 1024 * 32, z_dim=10, device=torch.device("cpu")):
        super(VaeEncoder, self).__init__()
        self.z_dim = z_dim
        self.device = device

        self.features = nn.Sequential(nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0),
                                      nn.ReLU(),
                                      nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(53824 + 10, self.z_dim, bias=True)  # fully-connected to output mu
        self.fc2 = nn.Linear(53824 + 10, self.z_dim, bias=True)  # fully-connected to output logvar

    def bottleneck(self, h):
        """
        This function takes features from the encoder and outputs mu, log-var and a latent space vector z
        :param h: features from the encoder
        :return: z, mu, log-variance
        """
        mu, logvar = self.fc1(h), self.fc2(h)
        # use the reparametrization trick as torch.normal(mu, logvar.exp()) is not differentiable
        z = reparameterize(mu, logvar, device=self.device)
        return z, mu, logvar

    def forward(self, x, label):
        """
        This is the function called when doing the forward pass:
        z, mu, logvar = VaeEncoder(X)
        """
        h = self.features(x)
        h = h.reshape(h.size(0), -1)
        z, mu, logvar = self.bottleneck(torch.cat([h, label], dim=1))
        return z, mu, logvar


class VaeDecoder(torch.nn.Module):
    """
       This class builds the decoder for the VAE
       :param x_dim: input dimensions
       :param hidden_size: hidden layer size
       :param z_dim: latent dimensions
       """

    def __init__(self, x_dim=1 * 1024 * 32, z_dim=10):
        super(VaeDecoder, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim

        self.features = nn.Sequential(nn.Linear(self.z_dim, 16 * 4),
                                      nn.ReLU(),
                                      nn.Linear(16 * 4, 32 * 4),
                                      nn.ReLU(),
                                      nn.Linear(32 * 4, 32 * 32),
                                      nn.ReLU())

        self.layer1 = nn.ConvTranspose2d(1, 32, 3, stride=2, padding=1, output_padding=1)
        self.layer2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.layer3 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)

        # input dims: bs x 1 x 128 x 4
        self.decoder = nn.Sequential(self.layer1,
                                     nn.ReLU(),
                                     self.layer2,
                                     nn.ReLU(),
                                     self.layer3,
                                     nn.Sigmoid())

    def forward(self, x):
        """
        This is the function called when doing the forward pass:
        x_reconstruction = VaeDecoder(z)
        """
        x = self.features(x)
        x = x.reshape(x.size(0), 1, 32, 32)
        x = self.decoder(x)
        return x


class cVae(torch.nn.Module):
    def __init__(self, x_dim=1 * 1024 * 32, z_dim=10, device=torch.device("cpu"), cond_dim=10):
        super(cVae, self).__init__()
        self.device = device
        self.z_dim = z_dim
        if cond_dim is None:
            self.encoder = VaeEncoder(z_dim=z_dim, device=device)
            self.decoder = VaeDecoder(x_dim, z_dim=z_dim)
        else:
            self.encoder = VaeEncoder(z_dim=z_dim, device=device)
            self.decoder = VaeDecoder(x_dim, z_dim=z_dim + cond_dim)

    def encode(self, x, label):
        z, mu, logvar = self.encoder(x, label)
        return z, mu, logvar

    def decode(self, z):
        x = self.decoder(z)
        return x

    def sample(self, num_samples=1, x_cond=None):
        z = torch.randn(num_samples, self.z_dim).to(self.device)
        if x_cond is None:
            return self.decode(torch.cat([z, labels_to_one_hots(torch.randint(0, 10, num_samples)).to(self.device)]))
        return self.decode(torch.cat([z, x_cond], dim=1))

    def forward(self, x, label=None):
        if label is None:
            random_label = labels_to_one_hots(torch.randint(0, 10, (x.shape[0],))).to(self.device)
            z, mu, logvar = self.encode(x, random_label)
            x_recon = self.decode(torch.cat([z, random_label], dim=1))
        else:
            z, mu, logvar = self.encode(x, label)
            x_recon = self.decode(torch.cat([z, label], dim=1))
        return x_recon, mu, logvar, z
