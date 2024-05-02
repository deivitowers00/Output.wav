import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io import wavfile
import wave

# .wav file
file_path = r"C:\Users\david\OneDrive\Desktop\Myke Towers, Bad Bunny - ADIVINO (LETRAS) 🎵 (320).wav"
sample_rate, audio_data = wavfile.read(file_path)
time = np.arange(len(audio_data)) / sample_rate

#max_value = np.max(np.abs(audio_data))
#audio_data_normalized = audio_data / 27000
#audio_data_normalized = np.clip(audio_data_normalized, -1, 1)

# Plot .wav file in the time domain
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, audio_data)
plt.title('Input Audio Signal (Time Domain)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Setting average
audio_data = np.mean(audio_data, axis=1)

# FFT to plot frequency domain
n = len(audio_data)
frequencies = np.fft.fftfreq(n, d=1/sample_rate)
audio_fft = np.fft.fft(audio_data)
plt.subplot(2, 1, 2)
plt.plot(frequencies[:n//2], np.abs(audio_fft[:n//2]))
plt.title('Input Audio Signal (Frequency Domain)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.show()

# Digital low-pass filter parameters
fc = 10000  # Center frequency in Hz
Q = 1.0    # Q factor
bw = fc / Q  # Bandwidth in Hz

# Digital filter
b, a = signal.iirfilter(4, [fc - 0.5*bw, fc + 0.5*bw], btype='bandpass', fs=sample_rate)

# Filter to audio signal
filtered_audio = signal.filtfilt(b, a, audio_data)


filtered_audio_max = np.max(np.abs(filtered_audio))
filtered_audio = filtered_audio / filtered_audio_max


# Plot filtered signal (time)
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, filtered_audio)
plt.title('Filtered Audio Signal (Time Domain)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Now FFT to plot frequency domain for filtered signal
filtered_fft = np.fft.fft(filtered_audio)
plt.subplot(2, 1, 2)
plt.plot(frequencies[:n//2], np.abs(filtered_fft[:n//2]))
plt.title('Filtered Audio Signal (Frequency Domain)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.show()

# Play and export filtered audio
with wave.open('output_filtered.wav', 'w') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(44100)
    wf.writeframes((filtered_audio * 32767).astype(np.int16).tobytes())

