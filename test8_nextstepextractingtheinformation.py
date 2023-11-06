# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 17:11:02 2023

@author: Danille
"""

#Bron:https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8
#Verder aangepast naar werkende code met de help console in Spyder en ChatGPT

import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from sklearn.preprocessing import scale

# Use a relative path to the audio file
audio_path = 'trap-future-bass-royalty-free-music-167020.mp3'

# Load the audio file
x, sr = librosa.load(audio_path)
print(type(x), type(sr))
print(x.shape, sr)

# Play the audio
ipd.Audio(audio_path)

# Plot the waveform
plt.figure(figsize=(14, 5))
plt.plot(x, color='b')
plt.title('Audio Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Create a spectrogram
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()

# Plot the signal
plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)
plt.title('Waveform')
plt.show()

# Zoom in
n0 = 9000
n1 = 9100
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1])
plt.grid()
plt.title('Zoomed In Waveform')
plt.show()

zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
print(sum(zero_crossings))

# Compute the spectral centroid
spectral_centroids = librosa.feature.spectral_centroid(y=x, sr=sr)[0]

# Compute the time variable for visualization
t1 = librosa.times_like(spectral_centroids)

# Normalize the spectral centroid for visualization
normalized_spectral_centroids = (spectral_centroids - np.min(spectral_centroids)) / (np.max(spectral_centroids) - np.min(spectral_centroids))

# Plot the waveform
plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr, alpha=0.4)
plt.plot(t1, normalized_spectral_centroids, color='r')
plt.title('Spectral Centroid')
plt.show()

# Compute the spectral rolloff
spectral_rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)[0]

# Compute the time variable for visualization
t2 = librosa.times_like(spectral_rolloff)

# Normalize the spectral rolloff for visualization
def normalize(x, axis=0):
    return (x - np.min(x, axis=axis)) / (np.max(x, axis=axis) - np.min(x, axis=axis))

# Plot the waveform
plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr, alpha=0.4)
plt.plot(t2, normalize(spectral_rolloff), color='r')
plt.title('Spectral Rolloff')
plt.show()

# Compute MFCCs
mfccs = librosa.feature.mfcc(y=x, sr=sr)

print(mfccs.shape)

# Display the MFCCs
plt.figure(figsize=(14, 5))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.title('MFCCs')
plt.show()

# Normalize the MFCCs
mfccs = scale(mfccs, axis=1)

print("Mean of MFCCs:", mfccs.mean(axis=1))
print("Variance of MFCCs:", mfccs.var(axis=1))

# Display the normalized MFCCs
plt.figure(figsize=(14, 5))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.title('Normalized MFCCs')
plt.show()

# Set the hop length
hop_length = 512

# Compute the chromagram
chromagram = librosa.feature.chroma_stft(y=x, sr=sr, hop_length=hop_length)

# Create a figure and display the chromagram
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
plt.title('Chromagram')
plt.show()
