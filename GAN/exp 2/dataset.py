"""
Class inherits from torch.utils.data.Dataset, used for loading spectograms amplitudes and phases.
"""

import numpy as np
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DigitsAudioDataset(Dataset):
    def __init__(self, root_dir,
                 transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5], [0.5, 0.5])])):
        """
            Arguments:
                root_dir - dataset path.
                transform - transforms to be applied.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.fileNames = [f for f in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, f))]

    def __len__(self):
        return len(self.fileNames)

    def __getitem__(self, idx):
        """
            Output:
                sample - a dictionary with two keys:
                'image' - the spectogram amplitude and phase (2-channel image).
                'label' - the digit's label (0-9).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.fileNames[idx]
        image = np.array(np.load(self.root_dir + '/' + img_name), dtype=np.float32)

        # Normalizing the image to [0..1] - amp and phase
        image[:, :, 0] = (image[:, :, 0] + 97.002) / (22.376 + 97.002)
        image[:, :, 1] = (image[:, :, 1] + np.pi) / (2*np.pi)
        image = np.clip(image, 0, 1)

        label = int(img_name[0])

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label}
        return sample
