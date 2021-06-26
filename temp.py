from shutil import copy
import os.path


for label in range(10):
    label = str(label)
    for speaker in range(1,61):
        if speaker < 10:
            speaker = '0'+str(speaker)
        else:
            speaker = str(speaker)
        if os.path.isfile('./dataset/train_spectograms_amplitude/' + label + '_' + speaker + '_1.npy'):
            copy('./dataset/train_spectograms_amplitude/' + label + '_' + speaker + '_1.npy', './dataset/data_for_metrics/real_spectograms_amplitude/'+label)
        if os.path.isfile('./dataset/train_spectograms_amplitude/' + label + '_' + speaker + '_2.npy'):
            copy('./dataset/train_spectograms_amplitude/' + label + '_' + speaker + '_2.npy', './dataset/data_for_metrics/real_spectograms_amplitude/'+label)


