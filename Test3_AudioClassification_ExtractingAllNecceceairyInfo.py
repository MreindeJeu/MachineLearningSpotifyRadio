# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:37:51 2023

@author: Danille
"""

import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display as lplt
import numpy as np
import pandas as pd


# Reading the CSV File
df = pd.read_csv("C:/Users/Danille/Documents/minor/Block2/Tech/Test_machineLearning_audioclasivication/Data/features_3_sec.csv")
df.head()
df.shape
df.dtypes
df = df.drop(labels="filename",axis=1)

# Use a relative path to the audio file
audio_path = 'C:/Users/Danille/Documents/minor/Block2/Tech/Test_machineLearning_audioclasivication/Data/genres_original/blues/blues.00000.wav'

# Load the audio file
data, sr = librosa.load(audio_path)

# Play the audio
ipd.display(ipd.Audio(data, rate=sr))

# Plot the waveform
plt.figure(figsize=(12, 4))
plt.plot(data, color='#2b4f72')
plt.title('Audio Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Create a spectrogram
stft = librosa.stft(data)
stftdb = librosa.amplitude_to_db(abs(stft))
plt.figure(figsize=(14, 6))
librosa.display.specshow(stftdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()

# Create the spectral rolloff
spectral_rolloff = librosa.feature.spectral_rolloff(y=data, sr=sr)[0]
t2 = librosa.times_like(spectral_rolloff)
plt.figure(figsize=(14, 5))
plt.plot(t2, spectral_rolloff, color='#2b4f72')
plt.title('Spectral Rolloff')
plt.show()

# Create the Chroma Features
chroma = librosa.feature.chroma_stft(y=data, sr=sr)
plt.figure(figsize=(16, 6))
lplt.specshow(chroma, sr=sr, x_axis="time", y_axis="chroma", cmap="coolwarm")
plt.colorbar()
plt.title('Chroma Features')
plt.show()

# Zero Crossing Rate over the entire audio signal
plt.figure(figsize=(14, 5))
plt.plot(data, color="#2b4f72")
plt.grid()

zero_cross_rate = librosa.feature.zero_crossing_rate(data)
print("The mean zero-crossing rate is:", np.mean(zero_cross_rate))

