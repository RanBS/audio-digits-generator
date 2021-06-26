"""
Train the generator and discriminator models (cond w-gan-gp).
Based on https://github.com/gcucurull/cond-wgan-gp
"""
import argparse
import os
import numpy as np
from dataset import DigitsAudioDataset

import torch
import torch.nn as nn
import torch.autograd as autograd

from torchvision.utils import save_image
from torch.utils.data import DataLoader

from models import Generator, Discriminator

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=150, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
parser.add_argument("--dataset", type=str, choices=['mnist', 'fashion', 'digits'],
                    default='digits', help="dataset to use")
opt = parser.parse_args()
print(opt)

opt.img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
print(cuda)
opt.n_classes = 10

# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator(opt)
discriminator = Discriminator(opt)

if cuda:
    generator.cuda()
    discriminator.cuda()

dataloader = torch.utils.data.DataLoader(dataset=DigitsAudioDataset('../../dataset/train_spectograms_amplitude'),
                                         batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim)))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    with torch.no_grad():
        labels = LongTensor(labels)
        gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for WGAN GP.
       Warning: It doesn't compute the gradient w.r.t the labels, only w.r.t
       the interpolated real and fake samples, as in the WGAN GP paper.
    """
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    labels = LongTensor(labels)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = Tensor(real_samples.shape[0], 1).fill_(1.0)
    fake.requires_grad = False
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gradients = gradients[0].view(gradients[0].size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ----------
#  Training
# ----------
generator.train()
discriminator.train()
batches_done = 0

for epoch in range(opt.n_epochs):
    train_batch_d_loss = []
    train_batch_g_loss = []
    for i, sampled_batch in enumerate(dataloader):
        imgs = sampled_batch['image']
        labels = sampled_batch['label']

        # Move to GPU if necessary
        real_imgs = imgs.type(Tensor)
        labels = labels.type(LongTensor)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Sample noise and labels as generator input
        z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))
        # Generate a batch of images
        fake_imgs = generator(z, labels)

        # Real images
        real_validity = discriminator(real_imgs, labels)
        # Fake images
        fake_validity = discriminator(fake_imgs, labels)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(
            discriminator, real_imgs.data, fake_imgs.data,
            labels.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z, labels)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs, labels)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            """
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
            """

            if batches_done % opt.sample_interval == 0:
                sample_image(opt.n_classes, batches_done)
            train_batch_g_loss.append(g_loss.data.cpu().item())

            batches_done += opt.n_critic
        train_batch_d_loss.append(d_loss.data.cpu().item())

    print(
        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch, opt.n_epochs, i, len(dataloader), np.mean(train_batch_d_loss), np.mean(train_batch_g_loss))
    )

    fname = "generator_" + str(epoch) + "_epochs.pth"
    torch.save(generator.state_dict(), fname)
    fname = "discriminator_" + str(epoch) + "_epochs.pth"
    torch.save(discriminator.state_dict(), fname)
