import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import resample

FS = 16000                 # Sampling rate (speech quality)
RECORD_TIME = 10           # Duration in seconds
FRAME_SIZE = 512           # Block size for processing

FILTER_LENGTH = 1024       # Models room echo path
MU = 0.25                  # NLMS step size
DELTA = 1e-2               # Stability constant

DOUBLE_TALK_THRESHOLD = 1.5  # Energy ratio threshold

far_end, fs_org = sf.read("handel.wav")
far_end = resample(far_end, int(len(far_end) * FS / fs_org))
far_end = far_end / np.max(np.abs(far_end)) * 0.4

far_len = len(far_end)
total_samples = FS * RECORD_TIME
num_frames = total_samples // FRAME_SIZE

mic_signal = np.zeros(total_samples)
far_signal = np.zeros(total_samples)
output_signal = np.zeros(total_samples)

print("\nStarting recording in 3 seconds...")
sd.sleep(3000)
print("Recording started. Speak normally.\n")
ptr = 0
for k in range(num_frames):

    if ptr + FRAME_SIZE >= far_len:
        ptr = 0

    far_frame = far_end[ptr:ptr + FRAME_SIZE]
    ptr += FRAME_SIZE

    sd.play(far_frame, FS)
    mic_frame = sd.rec(FRAME_SIZE, FS, channels=1)
    sd.wait()

    start = k * FRAME_SIZE
    end = start + FRAME_SIZE

    mic_signal[start:end] = mic_frame.flatten()
    far_signal[start:end] = far_frame

print("Recording completed.\n")


print("Performing echo cancellation...")

weights = np.zeros(FILTER_LENGTH)
buffer = np.zeros(FILTER_LENGTH)

for k in range(num_frames):

    start = k * FRAME_SIZE
    end = start + FRAME_SIZE

    d = mic_signal[start:end]      # Microphone signal
    x = far_signal[start:end]      # Reference signal

    error_frame = np.zeros(FRAME_SIZE)

    
    mic_energy = np.mean(d**2) + 1e-12
    far_energy = np.mean(x**2) + 1e-12

    double_talk = mic_energy / far_energy > DOUBLE_TALK_THRESHOLD

    for n in range(FRAME_SIZE):

        buffer = np.roll(buffer, 1)
        buffer[0] = x[n]

        echo_estimate = np.dot(weights, buffer)
        error = d[n] - echo_estimate

        
        if not double_talk:
            norm = np.dot(buffer, buffer) + DELTA
            weights += (MU / norm) * error * buffer

        error_frame[n] = error

    
    if np.mean(error_frame**2) < 0.005:
        error_frame *= 0.5

    output_signal[start:end] = error_frame

output_signal /= np.max(np.abs(output_signal) + 1e-12)
sf.write("removed_echo.wav", output_signal, FS)

print('✓ Echo cancelled output saved as "removed_echo.wav"\n')

time = np.arange(len(output_signal)) / FS

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time, mic_signal)
plt.title("Microphone Signal (Echo + Near-End Speech)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time, output_signal)
plt.title("After Acoustic Echo Cancellation")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()
