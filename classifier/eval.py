"""
Evaluates the classifier performances, after training.
"""

import torch
from dataset import DigitsAudioDataset
from torch.utils.data import DataLoader
from model import ConvNetwork
import numpy as np


def calculate_accuracy(model, dataloader, device):
    """
        Based on Deep Learning (046211) Technion.
        Calculates the accuracy of a model on a specific dataset.
        ========
        Arguments:
            model - a trained classifier.
            dataloader - a dataloader object. The accuracy will be evaluated on this object.
            device - the device to run the calculations on.
        Output:
            model_accuracy - the accuracy of the model in %.
            confusion_matrix - the confusions of the model, represented by a matrix.
    """
    model.eval()  # put in evaluation mode
    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([10, 10], int)
    with torch.no_grad():
        for sampled_batch in dataloader:

            images = sampled_batch['image']
            labels = sampled_batch['label']

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)

            total_correct += (predicted == labels).sum().item()
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1

    model_accuracy = total_correct / total_images * 100
    return model_accuracy, confusion_matrix


# Evaluate the train classifier on the spectograms dataset.
dataset = DigitsAudioDataset('../dataset/test_spectograms_amplitude')
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

net = ConvNetwork()
net.load_state_dict(torch.load('./weights.pth'))
net = net.to(torch.device("cuda:0"))
print(calculate_accuracy(net, dataloader, torch.device("cuda:0")))

"""
Results:
accuracy = 99.53333333333333
confusion matrix = 
[128,   0,   0,   0,   0,   0,   0,   0,   0,   0],
[  0, 143,   0,   0,   0,   1,   0,   0,   0,   2],
[  0,   0, 164,   0,   0,   0,   0,   0,   0,   0],
[  0,   0,   0, 132,   0,   0,   0,   0,   1,   0],
[  0,   0,   0,   0, 154,   0,   0,   0,   0,   0],
[  0,   0,   0,   0,   0, 165,   0,   0,   0,   1],
[  0,   0,   0,   0,   0,   0, 143,   0,   0,   0],
[  0,   0,   1,   0,   0,   0,   0, 158,   0,   0],
[  0,   0,   0,   0,   0,   0,   0,   0, 128,   0],
[  0,   0,   0,   0,   0,   0,   0,   0,   1, 178]
"""