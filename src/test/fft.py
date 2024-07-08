import numpy as np
from scipy.fft import ifft
import matplotlib.pyplot as plt

np.random.seed(1)
signal_with_noise = np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.5, 1000)
freq_domain_signal = np.fft.fft(signal_with_noise)

cleaned_freq_domain_signal = np.where(
    np.abs(freq_domain_signal) > 50, freq_domain_signal, 0
)
cleaned_signal = ifft(cleaned_freq_domain_signal)
# Plot the FFT
plt.figure(figsize=(10, 5))
plt.plot(signal_with_noise)
plt.plot(cleaned_signal)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Fast Fourier Transform (FFT)")
plt.grid(True)
plt.show()
