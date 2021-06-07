#Code help from HaythemFayek Melfilterbank tutorial
## https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html




import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct

sample_rate, signal = scipy.io.wavfile.read('00095.wav')
signal = signal[0:int(3.5 * sample_rate)]
print(signal)

## Apply a pre-emphasis filter to amplify high frequencies. This is used to balance the 
#frequecy to lower magnitudes and lower frequencies.

#Typical values for this filter is 0.95 and 0.97

pre_emphasis = 0.97
emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
print(emphasized_signal)

## Framing
# Split into 20- 40 ms frames since sometimes we have unwanted fluff that can be 
# misleading when getting coefficcients, do not want unwanted noise. Frames are typically
# 20 - 40 ms long

frame_size = 0.025
frame_stride = 0.01

frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
signal_length = len(emphasized_signal)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

pad_signal_length = num_frames * frame_step + frame_length
z = np.zeros((pad_signal_length - signal_length))
pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(np.int32, copy=False)]

## Window
# The above code sliced the signals into frames and now we need to apply a window function
# to counteract assumptions made by fft

frames *= np.hamming(frame_length)

## FFT and Power Spectrum
#SFFT : Shorth time fourier transform, N is typically 256 or 512
NFFT = 512

mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

## Mel Filter Banks Calculations
# Typical number of filters used is 40

nfilt = 40

low_freq_mel = 0
high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
bin = np.floor((NFFT + 1) * hz_points / sample_rate)

fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
filter_banks = np.dot(pow_frames, fbank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
filter_banks = 20 * np.log10(filter_banks)  # dB

print(filter_banks)
import matplotlib.pyplot as plt
plt.specgram(filter_banks)
#plt.show()

def filter_bank_computation(file, pre_emph_n, frame_size_n, frame_stride_n, NFFT_n, nfilt_n):
    sample_rate, signal = scipy.io.wavfile.read(str(file))
    signal = signal[0:int(3.5 * sample_rate)]
    print(signal)

    ## Apply a pre-emphasis filter to amplify high frequencies. This is used to balance the 
    #frequecy to lower magnitudes and lower frequencies.

    #Typical values for this filter is 0.95 and 0.97

    pre_emphasis = pre_emph_n
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    print(emphasized_signal)

    ## Framing
    # Split into 20- 40 ms frames since sometimes we have unwanted fluff that can be 
    # misleading when getting coefficcients, do not want unwanted noise. Frames are typically
    # 20 - 40 ms long

    frame_size = frame_size_n
    frame_stride = frame_stride_n

    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    ## Window
    # The above code sliced the signals into frames and now we need to apply a window function
    # to counteract assumptions made by fft

    frames *= np.hamming(frame_length)

    ## FFT and Power Spectrum
    #SFFT : Shorth time fourier transform, N is typically 256 or 512
    NFFT = NFFT_n

    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    ## Mel Filter Banks Calculations
    # Typical number of filters used is 40

    nfilt = nfilt_n

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
