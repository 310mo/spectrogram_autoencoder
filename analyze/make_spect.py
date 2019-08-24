import librosa
import librosa.display
import os
import numpy as np
import matplotlib.pyplot as plt

#入力はデータセットの冒頭3秒を切り取ったもの

def calculate_spect(x):
    stft = np.abs(librosa.stft(x, n_fft=512, hop_length=160, win_length=400))
    spect = librosa.amplitude_to_db(stft)
    return spect


file_path = "test.wav"
x, fs = librosa.load(file_path, sr=16000)
spect = calculate_spect(x)

np.save(file_path.split('.')[0], spect)