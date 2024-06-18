'''
    BLG 354E - Signal&Systems for Comp.Eng.
    Project-1 
    CRN: 21350
    
    Yusuf YILDIZ
    150210006
    Part-1, Part-2, Part-4
'''

import librosa
import soundfile as sf
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def extract_hidden_message(audio_path, sr):
    """
    Extracts the hidden message from an audio file using frequency domain manipulation.

    Args:
        audio_path (str): Path to the audio file containing the hidden message.
        sr (int): Sampling rate of the audio file.

    Returns:
        np.ndarray: The extracted hidden message in the time domain.
    """

    # Load the audio file
    y, sr = librosa.load(audio_path)
    # print(y.shape, sr)

    # Split the audio into one-second segments
    hop_length = sr
    segments = []
    for i in range(0, len(y), hop_length):
        segment_end = min(
            i + hop_length, len(y)
        )  # Ensure segment doesn't exceed audio length
        segments.append((i, segment_end))  # Append start and end indices

    # Initialize empty array to store the extracted message
    message = np.zeros_like(y) # initially zero

    for segment in segments:
        # Get the current audio segment
        audio_segment = y[segment[0] : segment[1]]

        # Perform Fast Fourier Transform (FFT) on the segment and message (assuming message is available)
        fft_audio = np.fft.fft(audio_segment)
        fft_message = np.zeros_like(
            fft_audio
        )  # Placeholder for actual message FFT it is padded with Zeros 

        # Split the FFTs into halves
        half_length = len(fft_audio) // 2
        audio_first_half = fft_audio[:half_length]
        audio_second_half = fft_audio[half_length:]

        # Replace second half of audio FFT with first half of message FFT (assuming message is known)
        message_second_half = fft_message[
            half_length:
        ]  # Placeholder, replace with actual message FFT later
        message_first_half = fft_message[:half_length]
        new_audio_fft = np.concatenate(
            (audio_second_half, message_first_half)
        )  # padded with zero for dimension compatibility

        # Perform inverse FFT to get the modified segment in the time domain
        modified_segment = np.fft.ifft(new_audio_fft).real

        # Accumulate the modified segments to form the extracted message
        message[segment[0] : segment[1]] += modified_segment

    return message

audio_path = "./bayrakfm.wav"

# Load the audio and get the sampling rate
y, sr = librosa.load(audio_path)

# Call the function passing the loaded sr
hidden_message = extract_hidden_message(audio_path, sr)

# Save the extracted message
extracted_message_file = "extracted_message.wav"
sf.write(extracted_message_file, hidden_message, sr)
print("Hidden message extracted and saved to:", extracted_message_file)


# ---------- PART 2 ----------

# Define the transfer function coefficients in Z-domain
numerator = [1, -7 / 4, -1 / 2]
denominator = [1, 1 / 4, -1 / 8]

# Create a TransferFunction object
transfer_function = signal.TransferFunction(numerator, denominator, dt=1.0)

# Load the WAV file
audio_file = "extracted_message.wav"
data, sample_rate = sf.read(audio_file)

# Apply the transfer function to the audio signal
filtered_data = signal.lfilter(transfer_function.num, transfer_function.den, data)

# Save the filtered audio to a new WAV file
filtered_audio_file = "filtered_message.wav"
sf.write(filtered_audio_file, filtered_data, sample_rate)
print("Filtered audio saved to:", filtered_audio_file)


# ---------- PART 4 ----------

# Define transfer functions
H1_num = [10, 10]  # Numerator of H1(jw)
H1_den = [1 / 100, 1 / 5, 1]  # Denominator of H1(jw)
H2_num = [1]  # Numerator of H2(jw)
H2_den = [1 / 10, 1]  # Denominator of H2(jw)

# Create TransferFunctionDiscrete objects
H1 = signal.TransferFunction(H1_num, H1_den)
H2 = signal.TransferFunction(H2_num, H2_den)

# Frequency response of H1
w1, mag1, phase1 = signal.bode(H1)

# Frequency response of H2
w2, mag2, phase2 = signal.bode(H2)

# Cascaded transfer function
H_cascade = signal.TransferFunction(
    np.convolve(H1_num, H2_num), np.convolve(H1_den, H2_den)
)

w_cascade, mag_cascade, phase_cascade = signal.bode(H_cascade)

"""
Explanation of the observations:
The results are saved in phase_response.png and magnitude_response.png files.

Only the H1 and cascaded transfer functions have initial coefficient K = 10. This causes an initialization at 20dB in the magnitude response.
H2 has no such initialization so it starts from 0dB.

In H1, there is a zero at w = 1 rad/s and this causes a 20dB increment. But it has 2 poles at in w = 10 rad/s so one of them causes 
finish the increment of the zero and other pole causes 20 dB decrement resulting in 20dB total.

As can be seen in H2 it has a pole at w = 10 rad/s and this causes a 20dB decrement.

When H1 and H2 are cascaded, it has zero at w = 1 rad/s and 3 poles at w = 10 rad/s, one more than the H1. As a result,
the response is lower than the H1 response with resulting in 0dB total. All the observations are consistent with the theory.

"""

# Plot Bode plots
plt.figure()
plt.semilogx(w1, mag1, label="H1(jω)")
plt.semilogx(w2, mag2, label="H2(jω)")
plt.semilogx(w_cascade, mag_cascade, label="Cascade")
plt.xlabel("Frequency [rad/s]")
plt.ylabel("Magnitude [dB]")
plt.title("Bode Plot")
plt.legend()
plt.grid()

plt.figure()
plt.semilogx(w1, phase1, label="H1(jω)")
plt.semilogx(w2, phase2, label="H2(jω)")
plt.semilogx(w_cascade, phase_cascade, label="Cascade")
plt.xlabel("Frequency [rad/s]")
plt.ylabel("Phase [degrees]")
plt.title("Bode Plot")
plt.legend()
plt.grid()

plt.show()
