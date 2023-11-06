# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:13:38 2023

@author: Danille
"""
# Import the required libraries
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

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

# Load the audio signal
x, sr = librosa.load('trap-future-bass-royalty-free-music-167020.mp3')

# Compute the spectral centroid
spectral_centroids = librosa.feature.spectral_centroid(y=x, sr=sr)[0]

# Compute the time variable for visualization
t = librosa.times_like(spectral_centroids)

# Normalize the spectral centroid for visualization
normalized_spectral_centroids = (spectral_centroids - np.min(spectral_centroids)) / (np.max(spectral_centroids) - np.min(spectral_centroids))

# Plot the waveform
plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr, alpha=0.4)

# Plot the normalized spectral centroid in red
plt.plot(t, normalized_spectral_centroids, color='r')

plt.show()





