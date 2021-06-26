"""
Generates digits sounds using the generator trained model.
"""

import numpy as np
from dataset import DigitsAudioDataset
import librosa
from scipy.io.wavfile import write
from model import cVae, labels_to_one_hots
import torch


def sample(label, cvae_model, path_to_save):
    """
        Arguments:
            label - the label of the sample to generate.
            generator_model - the trained cVae model.
            path_to_save - where to save the sample.
        Output:
            None.
    """
    sample_ = cvae_model.sample(1, labels_to_one_hots(torch.tensor([label])).to(device)).squeeze().data.cpu().numpy()
    sample_ = librosa.db_to_amplitude((sample_) * (22.376 + 97.002) - 97.002)

    rec_spec = np.zeros((256, 128))
    rec_spec[0:128] = sample_
    x = librosa.istft(rec_spec)
    x = np.array(np.clip(x / np.max(x), -1, 1), dtype=np.float32)
    write(path_to_save+'.wav', 16000, x)

def sample_amplitude(label, cvae_model, path_to_save):
    """
        Saves the amplitude of a generated sample.
        Used for calculating the scores.
        ====
        Arguments:
            label - the label of the sample to generate.
            generator_model - the trained cVae model.
            path_to_save - where to save the sample.
        Output:
            None.
    """
    sample_ = cvae_model.sample(1, labels_to_one_hots(torch.tensor([label])).to(device)).squeeze().data.cpu().numpy()
    sample_ = librosa.db_to_amplitude((sample_) * (22.376 + 97.002) - 97.002)
    np.save(path_to_save + '.npy', sample_)

if __name__ == "__main__":
    labels_to_sample = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cvae_model = cVae(x_dim=128*128, z_dim=10, device=device, cond_dim=10).to(device)
    cvae_model.load_state_dict(torch.load('cvae_weights.pth'))

    for i, label in enumerate(labels_to_sample):
        sample(label, cvae_model, './samples/sample' + str(i))
        #sample_amplitude(label, cvae_model, '../dataset/data_for_metrics/exp vae/' + str(label) + '/' + str(i))
