import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.fftpack

time_step = 0.02
period = 5.
time_vec = np.arange(0, 20, time_step)
sig = np.sin(2 * np.pi / period * time_vec) + 0.5 * np.random.randn(time_vec.size)
plt.plot(sig)
plt.show()

sample_freq = sp.fftpack.fftfreq(sig.size, d=time_step)
sig_fft = sp.fftpack.fft(sig)
pidxs = np.where(sample_freq > 0)
freqs, power = sample_freq[pidxs], np.abs(sig_fft)[pidxs]
freq = freqs[power.argmax()]
plt.stem(freqs[:50], power[:50])
plt.xlabel('frequency')
plt.ylabel('power')
plt.show()