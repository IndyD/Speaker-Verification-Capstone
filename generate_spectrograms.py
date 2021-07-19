import itertools
import json
import logging
import os
import pickle
import random
import sys
import scipy
import numpy as np

from collections import namedtuple, defaultdict

import utils 

import pdb
logging.basicConfig(level=logging.INFO)


def get_biggest_file(speech_session_dir):
    """
    For each speaker, we want to choose only one utterance or else the model
    will learn to classify using the background noise. This fn chooses the 
    biggest utterance for each speech session to ensure there is enough audio
    """
    wav_sizes = []
    audio_files = utils.listdir_nohidden(speech_session_dir)
    wavs = [audio_file for audio_file in audio_files if audio_file.endswith('.wav')]
    for wav in wavs:
        wav_path = os.path.join(speech_session_dir, wav)
        wav_sizes.append( (wav_path, os.path.getsize(wav_path)) )
    
    wav_sizes.sort(key=lambda tup: tup[1], reverse=True)

    if len(wav_sizes) == 0:
        return None
    else:
        return wav_sizes[0][0]

def generate_spectrogram(wavpath, params):
    """
    Take one file and generate a spectram for it 
    """
    hamming = scipy.signal.windows.hamming(params.DATA_GENERATOR.WIN_LENGTH, False)
    y, sr = librosa.load(wavpath)
    S = librosa.feature.melspectrogram(
        y, 
        sr, ## 22050 Hz
        n_fft=params.DATA_GENERATOR.N_FFT, ## recommended by librosa for speech, results in 23ms frames @22050
        n_mels=params.DATA_GENERATOR.N_MELS, ## too many mels resulted in empty banks
        window = hamming,
        win_length=params.DATA_GENERATOR.WIN_LENGTH, 
        hop_length=params.DATA_GENERATOR.HOP_LENGTH, ## tried to do 10 ms step as per VGGVox
    )
    log_S = librosa.power_to_db(S, ref=np.max)

    return log_S

def trim_spectrogram(spect, params):
    """
    Trims spectograms so they are all the same length, if too short return None
    """
    if spect.shape[1] < params.DATA_GENERATOR.MAX_FRAMES:
        return None
    else:
        return spect[:,:params.DATA_GENERATOR.MAX_FRAMES]

def generate_spectrograms(audio_dir, params):
    """
    Iteratively go thorugh each speaker dir, speech session dir, find the biggest
    utterance per speech session, convert that to a trimmed spectogram, and store in dict
    """
    speaker_spectrograms = defaultdict(list)

    speaker_dirs = utils.listdir_nohidden(audio_dir)
    for i, speaker in enumerate(speaker_dirs):
        if i % 50 == 10:
            logging.info('{i} of {n} speakers complete...'.format(i=i, n=len(speaker_dirs)))

        logging.debug('Generating spectograms for speaker: {sp}'.format(sp=speaker))
        speaker_dir = os.path.join(audio_dir, speaker)
        speech_session_dirs = utils.listdir_nohidden(speaker_dir)

        for speech_session in speech_session_dirs:
            speech_session_dir = os.path.join(speaker_dir, speech_session)
            utterance = get_biggest_file(speech_session_dir)
            if not utterance:
                continue
            spect = generate_spectrogram(utterance, params)
            spect = trim_spectrogram(spect, params)
            spect = spect / -80.0 ##normalize 
            ## Add an extra channel so the CNN works
            spect = np.expand_dims(spect, axis=-1)

            if spect is not None:
                speaker_spectrograms[speaker].append(spect)
                #spect_pkl = utterance.replace('.wav','.pkl')
                #spect_path = os.path.join(os.path.dirname(__file__), spect_pkl)
                #speaker_spectrograms[speaker].append(spect_path)
                #utils.save(spect, spect_path)

    return speaker_spectrograms


if __name__ == "__main__":
    import librosa  
    import librosa.display

    ### Set variables from config file ###
    PARAMS = utils.config_init(sys.argv)
    audio_dir = os.path.join(os.path.dirname(__file__), PARAMS.PATHS.AUDIO_DIR)
    output_dir = os.path.join(os.path.dirname(__file__), PARAMS.PATHS.OUTPUT_DIR)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    train_spectrogram_path = os.path.join(output_dir, 'speaker_spectrograms_train.pkl')
    test_spectrogram_path = os.path.join(output_dir, 'speaker_spectrograms_test.pkl')
    val_spectrogram_path = os.path.join(output_dir, 'speaker_spectrograms_val.pkl')
    overwrite_spect = PARAMS.DATA_GENERATOR.OVERWRITE_SPECT

    ### Generate or load spectograms ###
    if overwrite_spect == 'T' or not os.path.isfile(train_spectrogram_path):
        logging.info("Generating spectograms...")
        speaker_spectrograms = generate_spectrograms(audio_dir, PARAMS)
        train_speakers, test_speakers, val_speakers = utils.test_train_val_split(
            list(speaker_spectrograms.keys()),
            PARAMS.DATA_GENERATOR.TEST_SPLIT,
            PARAMS.DATA_GENERATOR.VALIDATION_SPLIT
        )
        train_spectrograms = dict((k, speaker_spectrograms[k]) for k in train_speakers)
        test_spectrograms = dict((k, speaker_spectrograms[k]) for k in test_speakers)
        val_spectrograms = dict((k, speaker_spectrograms[k]) for k in val_speakers)
        utils.save(train_spectrograms, train_spectrogram_path)
        utils.save(test_spectrograms, test_spectrogram_path)
        utils.save(val_spectrograms, val_spectrogram_path)