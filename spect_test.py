import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

fname = 'test_utterance.wav'
y,sr = librosa.load(fname)
S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
log_S = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(12,4))
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+02.0f dB')
plt.tight_layout()
plt.savefig('test_spectogram.png')
