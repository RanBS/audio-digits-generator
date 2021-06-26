"""
Class inherits from torch.utils.data.Dataset, used for loading spectograms amplitudes.
"""

import numpy as np
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DigitsAudioDataset(Dataset):
    def __init__(self, root_dir,
                 transform=transforms.Compose([transforms.ToTensor()])):
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
                    'image' - the spectogram amplitude.
                    'label' - the digit's label (0-9).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.fileNames[idx]
        image = np.array(np.load(self.root_dir + '/' + img_name), dtype=np.float32)

        # Normalizing the image to [0..1] - amp only
        image = (image + 97.002) / (22.376 + 97.002)
        image = np.clip(image, 0, 1)

        digit = int(img_name[0])
        label = [0 for _ in range(10)]
        label[digit] = 1
        label = np.asarray(label)

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label}
        return sample
