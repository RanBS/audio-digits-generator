"""
Generates digits sounds using the generator trained model.
"""

import numpy as np
from dataset import DigitsAudioDataset
import librosa
from scipy.io.wavfile import write
from models import Generator
import argparse
import torch


def prepare_generator(path):
    """
        Arguments:
            path - generator weights path.
        Output:
            generator_model - a loaded generator model with the required weigths.
            opt - the model's hyperparameters.
    """
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
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    parser.add_argument("--dataset", type=str, choices=['mnist', 'fashion', 'digits'], default='digits',
                        help="dataset to use")

    opt = parser.parse_args()
    opt.n_classes = 10
    opt.img_shape = (opt.channels, opt.img_size, opt.img_size)

    generator_model = Generator(opt)
    generator_model.load_state_dict(torch.load(path))

    generator_model.cuda()
    generator_model.eval()

    return generator_model, opt


def sample(label, generator_model, opt, path_to_save):
    """
        Arguments:
            label - the label of the sample to generate.
            generator_model - the trained generator model.
            opt - the model's hyperparameters.
            path_to_save - where to save the sample.
        Output:
            None.
    """
    Tensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    z = Tensor(np.random.normal(0, 1, (1, opt.latent_dim)))
    labels = np.array([label])
    labels = LongTensor(labels)
    d = generator_model(z, labels).cpu().data.numpy().squeeze()
    d = librosa.db_to_amplitude((d * 0.5 + 0.5) * (22.376 + 97.002) - 97.002)

    rec_spec = np.zeros((256, 128))
    rec_spec[0:128] = d
    x = librosa.istft(rec_spec)
    x = np.array(np.clip(x / np.max(x), -1, 1), dtype=np.float32)
    write(path_to_save+'.wav', 16000, x)

def sample_amplitude(label, generator_model, opt, path_to_save):
    """
        Saves the amplitude of a generated sample.
        Used for calculating the scores.
        ====
        Arguments:
            label - the label of the sample to generate.
            generator_model - the trained generator model.
            opt - the model's hyperparameters.
            path_to_save - where to save the sample.
        Output:
            None.
    """
    Tensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    z = Tensor(np.random.normal(0, 1, (1, opt.latent_dim)))
    labels = np.array([label])
    labels = LongTensor(labels)
    d = generator_model(z, labels).cpu().data.numpy().squeeze()
    d = librosa.db_to_amplitude((d * 0.5 + 0.5) * (22.376 + 97.002) - 97.002)

    np.save(path_to_save+'.npy', d)

if __name__ == "__main__":
    labels_to_sample = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    generator_model, opt = prepare_generator('generator_weights.pth')

    for i, label in enumerate(labels_to_sample):
        sample(label, generator_model, opt, './samples/sample'+str(i))
        #sample_amplitude(label, generator_model, opt, '../../dataset/data_for_metrics/exp 1/' + str(label) + '/' + str(i))
