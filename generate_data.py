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
import utils 

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
        n_fft=params.DATA_GENERATOR.N_FFT, ## recommended by librosa for speech, results in 23ms frames @22050
        n_mels=params.DATA_GENERATOR.N_MELS, ## too many mels resulted in empty banks
        win_length=params.DATA_GENERATOR.WIN_LENGTH, 
        hop_length=params.DATA_GENERATOR.HOP_LENGTH, ## tried to do 10 ms step as per VGGVox
    )
    log_S = librosa.power_to_db(S, ref=np.max)

    return log_S

def trim_spectogram(spect, params):
    """
    Trims spectograms so they are all the same length, if too short return None
    """
    if spect.shape[1] < params.DATA_GENERATOR.MAX_FRAMES:
        return None
    else:
        return spect[:,:params.DATA_GENERATOR.MAX_FRAMES]

def generate_spectograms(audio_dir, spectogram_path, params):
    """
    Iteratively go thorugh each speaker dir, speech session dir, find the biggest
    utterance per speech session, convert that to a trimmed spectogram, and store in dict
    """
    speaker_spectograms = defaultdict(list)

    speaker_dirs = listdir_no_hidden(audio_dir)
    for i, speaker in enumerate(speaker_dirs):
        if i % 50 == 25:
            logging.info('{i} of {n} speakers complete...'.format(i=i, n=len(speaker_dirs)))

        logging.debug('Generating spectograms for speaker: {sp}'.format(sp=speaker))
        speaker_dir = os.path.join(audio_dir, speaker)
        speech_session_dirs = listdir_no_hidden(speaker_dir)

        for speech_session in speech_session_dirs:
            speech_session_dir = os.path.join(speaker_dir, speech_session)
            utterance = get_biggest_file(speech_session_dir)
            spect = generate_spectogram(utterance, params)
            spect = trim_spectogram(spect, params)
            spect = spect / -80.0 ##normalize 
            ## Add an extra channel so the CNN works
            spect = np.expand_dims(spect, axis=-1)

            if spect is not None:
                speaker_spectograms[speaker].append(spect)

    return speaker_spectograms

def find_positive_pairs(corpus_data):
    """ Find the positions of all possible positive pairs, 
    interleaves speakers to maximize diversity """
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

    ## keep the data balanced
    if len(positive_pair_locs) < int(n_pairs / 2) and len(positive_pair_locs) > n_pairs:
        logging.warning("There aren't enough postive examples to keep the experiment balanced! Consider decreasing N_SAMPLES!")
    if n_pairs > len(positive_pair_locs):
        raise ValueError('Choosing an N_SAMPLES that is higher than possible with the data! Choose a smaller value!')

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
            neg_speaks.add(speakers)
            pair_data.append(data)
            pair_labels.append([0])

    return pair_data, pair_labels

def find_random_negative(corpus_data, exclude):
    candidates = [cand for cand in corpus_data if cand != exclude]
    speaker_id = random.choice(candidates)
    random_negative = random.choice(corpus_data[speaker_id])

    return random_negative

def make_contrastive_triplets(corpus_data, n_triplets):
    triplets = []
    positive_pair_locs = find_positive_pairs(corpus_data)

    if n_triplets > len(positive_pair_locs):
        raise ValueError('Choosing an N_SAMPLES that is higher than possible with the data! Choose a smaller value!')
    for i in range(n_triplets):
        spkr = positive_pair_locs[i][0]
        idx1 = positive_pair_locs[i][1]
        idx2 = positive_pair_locs[i][2]

        negative = find_random_negative(corpus_data, spkr)
        triplet = (corpus_data[spkr][idx1], corpus_data[spkr][idx2], negative)
        triplets.append(triplet)

    return triplets

def find_two_random_negatives(corpus_data, exclude):
    candidates = [cand for cand in corpus_data if cand != exclude]
    speaker_ids = random.sample(candidates, 2)
    random_negativeA = random.choice(corpus_data[speaker_ids[0]])
    random_negativeB = random.choice(corpus_data[speaker_ids[1]])

    return random_negativeA, random_negativeB

def make_contrastive_quadruplets(corpus_data, n_quadruplets):
    quadruplets = []
    positive_pair_locs = find_positive_pairs(corpus_data)

    if n_quadruplets > len(positive_pair_locs):
        raise ValueError('Choosing an N_SAMPLES that is higher than possible with the data! Choose a smaller value!')
    for i in range(n_quadruplets):
        spkr = positive_pair_locs[i][0]
        idx1 = positive_pair_locs[i][1]
        idx2 = positive_pair_locs[i][2]

        negativeA, negativeB = find_two_random_negatives(corpus_data, spkr)

        quadruplet = (
            corpus_data[spkr][idx1], 
            corpus_data[spkr][idx2], 
            negativeA,
            negativeB
        )
        quadruplets.append(quadruplet)

    return quadruplets


if __name__ == "__main__":
    ### Set variables from config file ###
    PARAMS = utils.config_init(sys.argv)
    audio_dir = os.path.join(os.path.dirname(__file__), PARAMS.PATHS.AUDIO_DIR)
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    spectogram_path = os.path.join(output_dir, 'speaker_spectograms.pkl')
    pairs_path = os.path.join(output_dir, 'contrastive_pairs.pkl')
    triplets_path = os.path.join(output_dir, 'contrastive_triplets.pkl')
    quadruplets_path = os.path.join(output_dir, 'contrastive_quadruplets.pkl')
    overwrite_spect = PARAMS.DATA_GENERATOR.OVERWRITE_SPECT
    overwrite_datasets = PARAMS.DATA_GENERATOR.OVERWRITE_DATASETS

    ### Generate or load spectograms ###
    if overwrite_spect == 'T' or not os.path.isfile(spectogram_path):
        logging.info("Generating spectograms...")
        speaker_spectograms = generate_spectograms(audio_dir, spectogram_path, PARAMS)
        utils.save(speaker_spectograms, spectogram_path)
    else:
        speaker_spectograms = utils.load(spectogram_path)
    
    ### Generate or contrastive pairs ###
    if PARAMS.TRAINING.LOSS_TYPE == 'contrastive':
        if overwrite_datasets == 'T' or not os.path.isfile(pairs_path):
            logging.info("Generating pairs for contrastive loss...")
            pairs, labels = make_contrastive_pairs(
                speaker_spectograms, 
                PARAMS.DATA_GENERATOR.N_SAMPLES
            )
            utils.save((pairs, labels), pairs_path)

    ### Generate or contrastive triplets ###
    if PARAMS.TRAINING.LOSS_TYPE == 'triplet':
        if overwrite_datasets == 'T' or not os.path.isfile(triplets_path):
            logging.info("Generating triplets for triplet loss...")
            triplets = make_contrastive_triplets(
                speaker_spectograms, 
                PARAMS.DATA_GENERATOR.N_SAMPLES,
            )
            utils.save(triplets, triplets_path)

    ### Generate or contrastive quadruplets ###
    if PARAMS.TRAINING.LOSS_TYPE == 'quadruplet':
        if overwrite_datasets == 'T' or not os.path.isfile(quadruplets_path):
            logging.info("Generating quadruplets for quadruplet loss...")
            quadruplets = make_contrastive_quadruplets(
                speaker_spectograms, 
                PARAMS.DATA_GENERATOR.N_SAMPLES,
            )
            utils.save(quadruplets, quadruplets_path)