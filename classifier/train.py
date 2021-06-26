"""
Based on Deep Learning (046211) Technion.
Train the model.
"""

import torch
from dataset import DigitsAudioDataset
from torch.utils.data import DataLoader
from model import ConvNetwork
import time
import numpy as np
from tqdm import tqdm

np.random.seed(5)
torch.manual_seed(5)

train_dataset = DigitsAudioDataset('../dataset/train_spectograms_amplitude')
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

NUM_EPOCHS = 50

# check if there is gpu available, if there is, use it
if torch.cuda.is_available():
    torch.cuda.current_device()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("running calculations on: ", device)

# create our model and send it to the device (cpu/gpu)
net = ConvNetwork().to(device)

# optimizer
optim = torch.optim.Adam(params=net.parameters(), lr=0.001)
loss_function = torch.nn.BCELoss()

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    train_batch_losses = []
    net.train()
    for batch_i, sampled_batch in tqdm(enumerate(train_dataloader)):
        # forward pass
        batch = sampled_batch['image']
        batch_labels = sampled_batch['label'].float()

        x = batch.to(device)  # just the images
        y = batch_labels.to(device)  # just the labels
        output = net(x)

        loss = loss_function(output, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        # save loss
        train_batch_losses.append(loss.data.cpu().item())

    print("epoch: {} training loss: {:.5f} epoch time: {:.3f} sec".format(epoch, np.mean(train_batch_losses),
                                                                          time.time() - epoch_start_time))
    fname = "classifier_" + str(epoch) + "_epochs.pth"
    torch.save(net.state_dict(), fname)
    print("saved checkpoint @", fname)
