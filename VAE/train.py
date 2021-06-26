"""
cVAE model training, based on ANAM (046202), Technion.
"""

import torch
from dataset import DigitsAudioDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import cVae, labels_to_one_hots
import time
import numpy as np
from tqdm import tqdm
import cv2

def loss_function(recon_x, x, mu, logvar, loss_type='bce'):
    """
    This function calculates the loss of the VAE.
    loss = reconstruction_loss - 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param recon_x: the reconstruction from the decoder
    :param x: the original input
    :param mu: the mean given X, from the encoder
    :param logvar: the log-variance given X, from the encoder
    :param loss_type: type of loss function - 'mse', 'l1', 'bce'
    :return: VAE loss
    """

    if loss_type == 'mse':
        recon_error = F.mse_loss(recon_x, x, reduction='sum')
    elif loss_type == 'l1':
        recon_error = F.l1_loss(recon_x, x, reduction='sum')
    elif loss_type == 'bce':
        recon_error = F.binary_cross_entropy(recon_x, x, reduction='sum')
    else:
        raise NotImplementedError

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return (recon_error + kl) / x.size(0)


dataset = DigitsAudioDataset('../dataset/train_spectograms_amplitude')
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
X_DIM = 1 * 128 * 128
Z_DIM = 10
cond_dim = 10

# check if there is gpu available, if there is, use it
if torch.cuda.is_available():
    torch.cuda.current_device()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("running calculations on: ", device)

# create our model and send it to the device (cpu/gpu)
cvae_ = cVae(x_dim=X_DIM, z_dim=Z_DIM, device=device, cond_dim=10).to(device)

# optimizer
cvae_optim = torch.optim.Adam(params=cvae_.parameters(), lr=LEARNING_RATE)

for epoch in range(1,NUM_EPOCHS+1):
    epoch_start_time = time.time()
    train_batch_losses = []
    cvae_.train()
    for batch_i, sampled_batch in tqdm(enumerate(dataloader)):
        # forward pass
        batch = sampled_batch['image']
        batch_labels = sampled_batch['label']

        x = batch.to(device)  # just the images
        y = batch_labels.to(device)  # just the labels
        x_recon, mu, logvar, z = cvae_(x, y)
        # calculate the loss
        total_loss = loss_function(x_recon, x, mu, logvar, 'mse')
        # optimization (same 3 steps everytime)
        cvae_optim.zero_grad()
        total_loss.backward()
        cvae_optim.step()

        # save loss
        train_batch_losses.append(total_loss.data.cpu().item())

    print("epoch: {} training loss: {:.5f} epoch time: {:.3f} sec".format(epoch, np.mean(train_batch_losses),
                                                                          time.time() - epoch_start_time))
    fname = "cvae_model_"+str(epoch)+"_epochs.pth"
    torch.save(cvae_.state_dict(), fname)

    cvae_.eval()
    mat = cvae_.sample(1,labels_to_one_hots(torch.tensor([0] * 1)).to(device)).squeeze().data.cpu().numpy()
    real = 255 * (mat - np.min(mat)) / (np.max(mat) - np.min(mat))
    cv2.imwrite('./samples/epoch_'+str(epoch)+'.png', real)

    print("saved checkpoint @", fname)

# save
fname = "cvae_model_final.pth"
torch.save(cvae_.state_dict(), fname)
print("saved checkpoint @", fname)