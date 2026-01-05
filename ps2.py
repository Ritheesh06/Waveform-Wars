import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft, butter, lfilter
import matplotlib.pyplot as plt

def normalize(x):
    return x / (np.max(np.abs(x)) + 1e-10)

def suppress_harmonics(Z, lifter=24):
    log_mag = np.log(np.abs(Z) + 1e-6)
    cep = np.fft.ifft(log_mag, axis=0).real
    cep[:lifter, :] = 0
    cep[-lifter:, :] = 0
    smooth_mag = np.exp(np.fft.fft(cep, axis=0).real)
    return smooth_mag * np.exp(1j * np.angle(Z))

def highpass(signal, fs, cutoff=2500):
    b, a = butter(2, cutoff/(fs/2), btype='high')
    return lfilter(b, a, signal)

fs, audio = wavfile.read("farend.wav")
if audio.ndim == 2:
    audio = audio[:, 0]

audio = normalize(audio.astype(np.float32))

nperseg = 1024
noverlap = 768
f, t, Zxx = stft(audio, fs, nperseg=nperseg, noverlap=noverlap)
mag = np.abs(Zxx)
eps = 1e-10

cutoff = 2100
low_band = f[:, None] < cutoff
high_band = f[:, None] >= cutoff

energy_low = np.sum(mag * low_band, axis=0)
energy_high = np.sum(mag * high_band, axis=0)
voice_frames = energy_low > 1.3 * energy_high

E_low = mag * low_band
E_high = mag * high_band

mask_low = E_low**2 / (E_low**2 + E_high**2 + eps)
mask_high = E_high**2 / (E_low**2 + E_high**2 + eps)

mask_high = mask_high ** 3.5
mask_high[:, voice_frames] *= 0.03

Z_voice = mask_low * Zxx
Z_instr = mask_high * Zxx

Z_instr = suppress_harmonics(Z_instr)

_, voice = istft(Z_voice, fs, nperseg=nperseg, noverlap=noverlap)
_, instrument = istft(Z_instr, fs, nperseg=nperseg, noverlap=noverlap)

voice = normalize(voice)
instrument = normalize(instrument)

instrument = highpass(instrument, fs)
instrument = normalize(instrument)

wavfile.write("VOICE_FINAL.wav", fs, voice.astype(np.float32))
wavfile.write("INSTRUMENT_FINAL.wav", fs, instrument.astype(np.float32))

print("✔ Blind DSP-only separation completed")
print("Saved: VOICE_FINAL.wav, INSTRUMENT_FINAL.wav")

time_audio = np.arange(len(audio)) / fs
time_voice = np.arange(len(voice)) / fs
time_instr = np.arange(len(instrument)) / fs

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time_audio, audio, color='gray')
plt.title("Original Mixed Signal")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time_voice, voice, color='blue')
plt.title("Separated Voice Signal")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time_instr, instrument, color='green')
plt.title("Separated Instrument Signal (Voice Suppressed)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()
