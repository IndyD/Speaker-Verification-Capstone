import librosa  
import os
import sys
import numpy as np
import pickle
import subprocess

import matplotlib.pyplot as plt
import librosa.display

def trim_spectrogram(spect):
    """
    Trims spectograms so they are all the same length, if too short return None
    """
    if spect.shape[1] < 300:
        return None
    else:
        return spect[:,:300]

def generate_spectrogram(wavpath):
    """
    Take one file and generate a spectram for it 
    """
    y, sr = librosa.load(wavpath)
    S = librosa.feature.melspectrogram(
        y, 
        sr, ## 22050 Hz
        n_fft=512, ## recommended by librosa for speech, results in 23ms frames @22050
        n_mels=130, ## too many mels resulted in empty banks
        win_length=512, 
        hop_length=222, ## tried to do 10 ms step as per VGGVox
    )
    spect = librosa.power_to_db(S, ref=np.max)

    spect = trim_spectrogram(spect)
    spect = spect / -80.0 ##normalize 
    ## Add an extra channel so the CNN works
    spect = np.expand_dims(spect, axis=-1)

    return spect

if __name__ == '__main__':
    fpath_path = sys.argv[1]

    if fpath_path.endswith('.m4a'):
        wav_path = fpath_path.replace('.m4a','.wav')
        call_str = 'ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s >/dev/null 2>/dev/null' %(fpath_path, wav_path)
        out = subprocess.call(call_str, shell=True) 
    elif fpath_path.endswith('.wav'):
        wav_path = fpath_path

    spect = generate_spectrogram(wav_path)
    plt.figure()
    librosa.display.specshow(spect)

    '''
    ### save picked sprectrogram
    spect_path = wav_path.replace('.wav','.pkl')
    with open(spect_path, 'wb') as fout:
        pickle.dump(spect, fout)
    '''

    