import itertools
import json
import logging
import os
import pickle
import random
import sys
from collections import namedtuple, defaultdict

import librosa
import numpy as np
import librosa.display
from sklearn.model_selection import train_test_split

import pdb
logging.basicConfig(level=logging.INFO)



def listdir_no_hidden(input):
    """ Lists files in a dir, ignoring hidden files that can cause issues """
    dirlist = os.listdir(input)
    dirlist = [dir for dir in dirlist if not dir.startswith('.')]
    return dirlist

def get_biggest_file(speech_session_dir):
    """
    For each speaker, we want to choose only one utterance or else the model
    will learn to classify using the background noise. This fn chooses the 
    biggest utterance for each speech session to ensure there is enough audio
    """
    wav_sizes = []
    audio_files = listdir_no_hidden(speech_session_dir)
    wavs = [audio_file for audio_file in audio_files if audio_file.endswith('.wav')]
    for wav in wavs:
        wav_path = os.path.join(speech_session_dir, wav)
        wav_sizes.append( (wav_path, os.path.getsize(wav_path)) )
    
    wav_sizes.sort(key=lambda tup: tup[1], reverse=True)
    return wav_sizes[0][0]

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
    Trims spectograms so they are all the same length, if too short return None
    """
    if spect.shape[1] < params.DATA_GENERATOR.max_frames:
        return None
    else:
        return spect[:,:params.DATA_GENERATOR.max_frames]

def generate_spectograms(audio_dir, spectogram_path, params):
    """
    Iteratively go thorugh each speaker dir, speech session dir, find the biggest
    utterance per speech session, convert that to a trimmed spectogram, and store in dict
    """
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

def find_positive_pairs(corpus_data):
    """ Find the positions of all possible  """
    positive_pair_locs = []
    unformatted_pair_locs = []
    for speaker, speaker_data in corpus_data.items():
        index_list = list(range(len(speaker_data)))
        index_combs = list(itertools.combinations(index_list, 2))
        speaker_pair_locs = [(speaker, idx[0], idx[1]) for idx in index_combs]
        unformatted_pair_locs.append(speaker_pair_locs)

    most_pairs = max(len(x) for x in unformatted_pair_locs)
    for i in range(most_pairs):
        for speaker_pair_locs in unformatted_pair_locs:
            if len(speaker_pair_locs) >= i+1:
                positive_pair_locs.append(speaker_pair_locs[i])

    return positive_pair_locs

def find_negative_pair(corpus_data):
    spk1 = random.choice(list(corpus_data.keys()))
    spk2 = random.choice([spk for spk in corpus_data.keys() if spk != spk1])

    spk1_data = random.choice(corpus_data[spk1])
    spk2_data = random.choice(corpus_data[spk2])

    speakers = (spk1, spk2)
    data = (spk1_data, spk2_data)

    return speakers, data

def make_contrastive_pairs(corpus_data, n_pairs):
    pair_data = []
    pair_labels = []
    
    positive_pair_locs = find_positive_pairs(corpus_data)

    if len(positive_pair_locs) > int(n_pairs / 2):
        positive_pair_locs = positive_pair_locs[0:int(n_pairs / 2)]
        for loc in positive_pair_locs:
            pair_data.append(
                (corpus_data[loc[0]][loc[1]], corpus_data[loc[0]][loc[2]])
            )
            pair_labels.append([1])
    
    ## This ensures the same speaker pairs can't be considered more than twice (reverse is possible)
    neg_speaks = set([])
    while len(pair_data) < n_pairs:
        speakers, data = find_negative_pair(corpus_data)
        if speakers not in neg_speaks:
            pdb.set_trace()
            neg_speaks.add(speakers)
            pair_data.append(data)
            pair_labels.append([0])

    return pair_data, pair_labels

if __name__ == "__main__":
    ### Initialize config file ###
    if len(sys.argv) == 1:
        raise ValueError("ERROR, need config file. Ie: python generate_data.py config.json")    
    configuration_file = str(sys.argv[1])
    with open(configuration_file, 'r') as f:
        f = f.read()
        PARAMS = json.loads(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

    ### Set variables from config file ###
    audio_dir = os.path.join(PARAMS.PATHS.base_dir, PARAMS.PATHS.audio_dir)
    output_dir = os.path.join(PARAMS.PATHS.base_dir, PARAMS.PATHS.output_dir)
    spectogram_path = os.path.join(PARAMS.PATHS.base_dir, PARAMS.PATHS.spectogram_path)

    ### Generate or load spectograms ###
    if not os.path.isfile(spectogram_path):
        speaker_spectograms = generate_spectograms(audio_dir, spectogram_path, PARAMS)
        with open(spectogram_path, 'wb') as fout:
            pickle.dump(speaker_spectograms, fout)
    else:
        with open(spectogram_path, 'rb') as fin:
            speaker_spectograms = pickle.load(fin)
    
    pairs, labels = make_contrastive_pairs(
        speaker_spectograms, 
        PARAMS.DATA_GENERATOR.n_pairs
    )

    pairs_train, pairs_test, y_train, y_test = train_test_split(
        pairs, labels, test_size=PARAMS.DATA_GENERATOR.test_split, random_state=1
    )
