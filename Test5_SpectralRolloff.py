# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:25:32 2023

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

# Compute the spectral rolloff
spectral_rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)[0]

# Compute the time variable for visualization
t = librosa.times_like(spectral_rolloff)

# Normalize the spectral rolloff for visualization
def normalize(x, axis=0):
    return (x - np.min(x, axis=axis)) / (np.max(x, axis=axis) - np.min(x, axis=axis))

# Plot the waveform
plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr, alpha=0.4)

# Plot the normalized spectral rolloff in red
plt.plot(t, normalize(spectral_rolloff), color='r')

plt.show()











