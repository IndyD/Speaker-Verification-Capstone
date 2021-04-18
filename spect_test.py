import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import pdb
import sys

fname = sys.argv[1]
y,sr = librosa.load(fname)
S = librosa.feature.melspectrogram(
    y, sr=sr, ## 22050 Hz
    n_fft=512, ## recommended by librosa for speech, results in 23ms frames @22050
    n_mels=130, ## too many mels resulted in empty banks
    win_length=512, 
    hop_length=222, ## tried to do 10 ms step as per VGGVox
)
log_S = librosa.power_to_db(S, ref=np.max)
print(log_S.shape)
pdb.set_trace()

plt.figure(figsize=(12,4))
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+02.0f dB')
plt.tight_layout()
plt.savefig('test_spectogram.png')
