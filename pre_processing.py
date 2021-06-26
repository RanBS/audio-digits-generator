"""
Convert audio files to .py files.
Based on https://github.com/avikhemani/CS230Project.
"""

from colorama import Fore, Style
import os
import librosa
import numpy as np
import random

AUDIO_LEN = 16000

def getInputOutputData(directory):
    fileNames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    n = len(fileNames)
    random.seed(1)
    random.shuffle(fileNames)

    pad2d = lambda a, i: a[:, 0: i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))

    for (i, fileName) in enumerate(fileNames):
        if (i + 1) % 500 == 0: print(Fore.LIGHTBLUE_EX + "Loaded " + str(i + 1) + " out of " + str(n))
        timeSeries, samplingRate = librosa.load(os.path.join(directory, fileName), sr=AUDIO_LEN)

        stft = librosa.stft(timeSeries, 512)
        amp = np.abs(stft)
        amp = pad2d(amp, 128)
        amp = librosa.amplitude_to_db(amp)

        phase = np.angle(stft)
        phase = pad2d(phase, 128)

        spec = np.zeros((128, 128, 2))
        spec[:, :, 0] = amp[0:128, :]
        spec[:, :, 1] = phase[0:128, :]

        np.save(directory + '/spectogram/' + fileName[:-4] + '.npy', spec)

    print(Fore.LIGHTMAGENTA_EX + "Loading completed")
    print(Style.RESET_ALL)

def main():
    wav_folder = ' '
    getInputOutputData(wav_folder)

if __name__ == '__main__':
    main()
