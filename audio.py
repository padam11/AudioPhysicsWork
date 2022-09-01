import librosa
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.signal import find_peaks

data, sampling_frequency = librosa.load('./Tuning fork1.mp3')

T = 1/sampling_frequency
N = len(data)
t = N / sampling_frequency

Y_k = np.fft.fft(data)[0:int(N/2)]/N
Y_k[1:] = 2*Y_k[1:]
Pxx = np.abs(Y_k)

f = sampling_frequency * np.arange((N/2)) / N
fig,ax = plt.subplots()
plt.plot(f[0:5000], Pxx[0:5000], linewidth=2)
plt.ylabel('Amplitude')
plt.xlabel('Frequency [Hz]')
plt.show()

auto = sm.tsa.acf(data, nlags=2000)
peaks = find_peaks(auto)[0]
lag = peaks[0]

pitch = sampling_frequency / lag 