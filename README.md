🎧 WAVEFORM WARS – DSP Design Challenge
Classical Digital Signal Processing for Real-World Audio Problems

ELAN & nVision | IIT Hyderabad

📌 Project Summary

This repository presents a pure Digital Signal Processing (DSP) solution to the Waveform Wars Design Challenge, focusing on solving real-world audio problems using only classical signal processing techniques.

🚫 No Machine Learning. No Neural Networks. No Pretrained Models.
All solutions are deterministic, interpretable, and grounded in DSP fundamentals.

🧩 Problem Statements
🔊 1. Acoustic Echo Cancellation (AEC)

In hands-free systems (smart speakers, conferencing devices, automotive infotainment), microphones capture:

Near-end user speech

Echoed far-end audio from the loudspeaker

This results in degraded speech quality, reduced intelligibility, and echo artifacts.

Objective:
Design an adaptive DSP-based system that estimates the unknown acoustic echo path and cancels the echo in real time, while remaining stable during double-talk scenarios.

🎵 2. Blind Audio Source Separation

In many recordings, multiple audio sources overlap in time (e.g., speech + music), and only a single microphone signal is available.

Constraints:

Single-channel input

No training data

No source models

Objective:
Separate overlapping sources using classical time–frequency DSP techniques.

🧠 DSP Techniques Used
Acoustic Echo Cancellation

FIR echo path modeling

LMS / NLMS adaptive filtering

Error signal minimization

Energy-based double-talk detection

Normalization for numerical stability

Audio Source Separation

Short-Time Fourier Transform (STFT)

Frequency band energy analysis

Soft time–frequency masking

Cepstral liftering for harmonic suppression

Inverse STFT (ISTFT) reconstruction

📂 Repository Structure
Waveform-Wars-DSP/
│
├── Acoustic_Echo_Cancellation/
│   ├── aec_nlms.py              # NLMS-based echo cancellation
│   ├── handel.wav               # Far-end reference signal
│   └── removed_echo.wav         # Echo-cancelled output
│
├── Audio_Source_Separation/
│   ├── source_separation.py     # Blind DSP-based source separation
│   ├── farend.wav               # Mixed input signal
│   ├── VOICE_FINAL.wav          # Separated voice output
│   └── INSTRUMENT_FINAL.wav     # Separated instrument output
│
├── requirements.txt
└── README.md

⚙️ Setup Instructions
✅ Prerequisites

Python 3.8 or higher

Functional microphone and speakers (for AEC demo)

📥 Clone the Repository
git clone https://github.com/<your-username>/Waveform-Wars-DSP.git
cd Waveform-Wars-DSP

📦 Install Dependencies
pip install -r requirements.txt

📚 Required Python Libraries

numpy

scipy

matplotlib

soundfile

sounddevice

▶️ How to Run
🔊 Acoustic Echo Cancellation
cd Acoustic_Echo_Cancellation
python aec_nlms.py


Outputs:

removed_echo.wav

Time-domain waveform plots showing echo suppression

🎵 Blind Audio Source Separation
cd Audio_Source_Separation
python source_separation.py


Outputs:

VOICE_FINAL.wav

INSTRUMENT_FINAL.wav

Waveform visualizations for analysis

📊 Evaluation Methodology

Due to the absence of clean ground-truth sources, evaluation follows standard blind source separation metrics:

SDR – Signal-to-Distortion Ratio

SIR – Signal-to-Interference Ratio

SAR – Signal-to-Artifacts Ratio

Results reflect realistic limitations of single-channel DSP processing.

🚫 Design Constraints Followed

✔ No machine learning
✔ No neural networks
✔ No pretrained models
✔ No data-driven optimization

All results are obtained using classical, explainable DSP techniques.

📈 Key Takeaways

Adaptive DSP can effectively cancel real acoustic echoes

Blind single-channel source separation is achievable using spectral analysis

Classical DSP remains lightweight, interpretable, and real-time capable

🔮 Limitations & Future Work
Limitations

Single-channel separation restricts achievable performance

Fixed frequency cutoff limits adaptability

Residual harmonic overlap between sources

Future Enhancements

Adaptive frequency partitioning

Improved soft masking techniques

Advanced double-talk detection

Real-time embedded system deployment

📚 References

S. Haykin, Adaptive Filter Theory

A. V. Oppenheim & A. S. Willsky, Signals and Systems

A. V. Oppenheim & R. Schafer, Discrete-Time Signal Processing

IEEE Signal Processing Society Resources
