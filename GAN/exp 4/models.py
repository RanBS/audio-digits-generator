"""
Generator and Discriminator models (cond w-gan-gp).
Based on https://github.com/gcucurull/cond-wgan-gp
"""

import numpy as np
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # gets the cond data
        self.model1 = nn.Sequential(
            *block(opt.cond_dim, 1024, normalize=False),
            *block(1024, 512)
        )

        # gets also the z vector
        self.model2 = nn.Sequential(
            *block(512+opt.latent_dim, 512),
            *block(512, 512),
            nn.Linear(512, int(np.prod(opt.img_shape))),
            nn.Tanh()
        )

    def forward(self, z, conds):
        # conds is [batch_size, cond_dim]
        first_part = self.model1(conds)
        img = self.model2(torch.cat((first_part, z),-1))
        img = img.view(img.shape[0], * self.opt.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.cond_dim + int(np.prod(opt.img_shape)), 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, img, conds):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), conds), -1)
        validity = self.model(d_in)
        return validity
