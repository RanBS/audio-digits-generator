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
                 transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])):
        """
            Output:
                sample - a dictionary with three keys:
                    'amp' - the spectogram amplitude.
                    'phase' - the spectogram phase.
                    'label' - the digit's label (one hot vector).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.fileNames = [f for f in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, f))]

    def __len__(self):
        return len(self.fileNames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.fileNames[idx]
        image = np.array(np.load(self.root_dir + '/' + img_name), dtype=np.float32)

        # Normalizing the image to [0..1] - amp and phase
        image[:, :, 0] = (image[:, :, 0] + 97.002) / (22.376 + 97.002)
        image[:, :, 1] = (image[:, :, 1] + np.pi) / (2 * np.pi)
        image = np.clip(image, 0, 1)

        amp = image[:, :, 0]
        phase = image[:, :, 1]

        digit = int(img_name[0])
        label = [0 for _ in range(10)]
        label[digit] = 1

        if self.transform:
            amp = self.transform(amp)
            phase = self.transform(phase)

        sample = {'amp': amp, 'phase': phase, 'label': np.array(label)}
        return sample

    def get_sample_by_label(self, label, idx):
        """
            Get a sample from the dataset with a specific label, when the sample is the 'idx' one.
            =====
            Arguments:
                label - the label of the sample (0-9)
                idx - the index of the sample (from all the samples with the right label).
            Output:
                sample - a sample with the required label.
        """
        label_names = [name for name in self.fileNames if int(name[0]) == label]
        if idx >= len(label_names):
            idx = len(label_names) - 1

        img_name = label_names[idx]
        image = np.array(np.load(self.root_dir + '/' + img_name), dtype=np.float32)

        # Normalizing the image to [0..1] - amp and phase
        image[:, :, 0] = (image[:, :, 0] + 97.002) / (22.376 + 97.002)
        image[:, :, 1] = (image[:, :, 1] + np.pi) / (2 * np.pi)
        image = np.clip(image, 0, 1)

        amp = image[:, :, 0]
        phase = image[:, :, 1]

        digit = int(img_name[0])
        label = [0 for _ in range(10)]
        label[digit] = 1

        if self.transform:
            amp = self.transform(amp)
            phase = self.transform(phase)

        sample = {'amp': amp, 'phase': phase, 'label': np.array(label)}
        return sample
