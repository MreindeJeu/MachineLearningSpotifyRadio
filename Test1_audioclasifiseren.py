# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:16:51 2023

@author: Danille
"""



import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from sklearn.preprocessing import scale

# Gebruik een relatief pad naar het audiobestand
audio_path = 'trap-future-bass-royalty-free-music-167020.mp3'

x, sr = librosa.load(audio_path)
print(type(x), type(sr))
print(x.shape, sr)

# Audio afspelen
ipd.Audio(audio_path)

# Plot de golfvorm
plt.figure(figsize=(14, 5))
plt.plot(x, color='b')
plt.title('Audio Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Maak een spectrogram
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()

# Zoomen in
n0 = 9000
n1 = 9100
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1])
plt.grid()

# Spectral Centroid
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]

# Spectral Rolloff
spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
normalized_rolloff = scale(spectral_rolloff, axis=0)
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(np.linspace(0, len(x) / sr, len(normalized_rolloff)), normalized_rolloff, color='r')

# MFCCs
mfccs = librosa.feature.mfcc(x, sr=sr)
print(mfccs.shape)
mfccs = scale(mfccs, axis=1)
print(mfccs.mean(axis=1))
print(mfccs.var(axis=1))  # Corrected the missing parenthesis
librosa.display.specshow(mfccs, sr=sr, x_axis='time')

# Compute STFT
stft = librosa.stft(x, hop_length=hop_length)
chromagram = librosa.feature.chroma_stft(S=stft, sr=sr)

# Plot Chromagram
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
