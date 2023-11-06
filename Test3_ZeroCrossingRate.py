# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:03:27 2023

@author: Danille
"""

# Import the required libraries
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display


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

librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()

# Load the signal
x, sr = librosa.load('trap-future-bass-royalty-free-music-167020.mp3')

# Plot the signal:
plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)
plt.show()

# Zooming in
n0 = 9000
n1 = 9100
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1])
plt.grid()

zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
print(sum(zero_crossings))