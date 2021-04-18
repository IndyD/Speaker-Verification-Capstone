import json
import logging
import os
import pickle
import sys
from collections import namedtuple, defaultdict

import librosa
import numpy as np
import librosa.display

import pdb
logging.basicConfig(level=logging.INFO)


def generate_spectograms(audio_dir, spectogram_path, params):
    speaker_spectograms = defaultdict(list)

    speaker_dirs = listdir_no_hidden(audio_dir)
    for speaker in speaker_dirs:
        logging.info('Generating spectograms for speaker: {sp}'.format(sp=speaker))
        speaker_dir = os.path.join(audio_dir, speaker)
        speech_session_dirs = listdir_no_hidden(speaker_dir)

        for speech_session in speech_session_dirs:
            speech_session_dir = os.path.join(speaker_dir, speech_session)
            utterance = get_biggest_file(speech_session_dir)
            spect = generate_spectogram(utterance, params)
            spect = trim_spectogram(spect, params)
            if spect is not None:
                speaker_spectograms[speaker].append(spect)

    return speaker_spectograms

def get_biggest_file(speech_session_dir):
    wav_sizes = []
    audio_files = listdir_no_hidden(speech_session_dir)
    wavs = [audio_file for audio_file in audio_files if audio_file.endswith('.wav')]
    for wav in wavs:
        wav_path = os.path.join(speech_session_dir, wav)
        wav_sizes.append( (wav_path, os.path.getsize(wav_path)) )
    
    wav_sizes.sort(key=lambda tup: tup[1], reverse=True)
    return wav_sizes[0][0]


def listdir_no_hidden(input):
    dirlist = os.listdir(input)
    dirlist = [dir for dir in dirlist if not dir.startswith('.')]
    return dirlist

def generate_spectogram(wavpath, params):
    """
    Take one file and generate a spectram for it 
    """
    y, sr = librosa.load(wavpath)
    S = librosa.feature.melspectrogram(
        y, 
        sr, ## 22050 Hz
        n_fft=params.DATA_GENERATOR.n_fft, ## recommended by librosa for speech, results in 23ms frames @22050
        n_mels=params.DATA_GENERATOR.n_mels, ## too many mels resulted in empty banks
        win_length=params.DATA_GENERATOR.win_length, 
        hop_length=params.DATA_GENERATOR.hop_length, ## tried to do 10 ms step as per VGGVox
    )
    log_S = librosa.power_to_db(S, ref=np.max)

    return log_S

def trim_spectogram(spect, params):
    """
    Trims spectograms so they are all the same length
    """
    if spect.shape[1] < params.DATA_GENERATOR.max_frames:
        return None
    else:
        return spect[:,:params.DATA_GENERATOR.max_frames]



if __name__ == "__main__":
    configuration_file = str(sys.argv[1])
    if configuration_file == "":
        raise ValueError("ERROR: you need to define param: config_model_datatype.json ")
    
    with open(configuration_file, 'r') as f:
        f = f.read()
        PARAMS = json.loads(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

    audio_dir = os.path.join(PARAMS.PATHS.base_dir, PARAMS.PATHS.audio_dir)
    output_dir = os.path.join(PARAMS.PATHS.base_dir, PARAMS.PATHS.output_dir)
    spectogram_path = os.path.join(PARAMS.PATHS.base_dir, PARAMS.PATHS.spectogram_path)

    if not os.path.isfile(spectogram_path):
        speaker_spectograms = generate_spectograms(audio_dir, spectogram_path, PARAMS)
        with open(spectogram_path, 'wb') as fout:
            pickle.dump(speaker_spectograms, fout)
    else:
        with open(spectogram_path, 'rb') as fin:
            speaker_spectograms = pickle.load(fin)
    pdb.set_trace()