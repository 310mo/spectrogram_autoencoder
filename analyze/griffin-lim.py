import numpy as np
import librosa
import librosa.display
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import wave
import struct
import os

def show_spect(spect, fs, file):
    librosa.display.specshow(spect, sr=fs)
    plt.savefig(file.split('.')[0]+'.png')

file = "test.npy"

spect = np.load(file)
spect = np.reshape(spect, (257, 301))
show_spect(spect, 16000, file)


#griffin-lim法の実装
A = librosa.db_to_amplitude(spect)
theta = 0
X = A * np.cos(theta) + A * np.sin(theta) * 1j

for i in range(100):
    x= librosa.istft(X, hop_length=160, win_length=400)
    X = librosa.stft(x, n_fft=512, hop_length=160, win_length=400)
    X = A * X / np.abs(X)

librosa.output.write_wav(file.split('.')[0]+'-reconstruct.wav', x, 16000)