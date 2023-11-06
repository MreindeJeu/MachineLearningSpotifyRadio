# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 17:07:20 2023

@author: Danille
"""

# Import the required libraries
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

# Load the audio signal
x, sr = librosa.load('trap-future-bass-royalty-free-music-167020.mp3')

# Set the hop length
hop_length = 512

# Compute the chromagram
chromagram = librosa.feature.chroma_stft(y=x, sr=sr, hop_length=hop_length)

# Create a figure and display the chromagram
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')

# Show the plot
plt.show()