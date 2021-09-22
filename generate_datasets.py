import itertools
import glob
import json
import logging
import os
import pickle
import random
import sys

import tensorflow as tf
import numpy as np

from collections import namedtuple, defaultdict

import utils 

import pdb
logging.basicConfig(level=logging.INFO)



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

    spk1_idx = random.choice(list(enumerate(corpus_data[spk1])))[0]
    spk2_idx = random.choice(list(enumerate(corpus_data[spk2])))[0]

    speakers = (spk1, spk2)
    data = ((spk1, spk1_idx), (spk2, spk2_idx), 0)

    return speakers, data

def find_random_negative(corpus_data, exclude):
    candidates = [cand for cand in corpus_data if cand != exclude]
    speaker_id = random.choice(candidates)
    random_negative = random.choice(range(len(corpus_data[speaker_id])))
    return speaker_id, random_negative


def find_two_random_negatives(corpus_data, exclude):
    candidates = [cand for cand in corpus_data if cand != exclude]
    speaker_ids = random.sample(candidates, 2)
    random_negativeA = random.choice(range(len(corpus_data[speaker_ids[0]])))
    random_negativeB = random.choice(range(len(corpus_data[speaker_ids[1]])))

    return (speaker_ids[0], random_negativeA), (speaker_ids[1], random_negativeB)

def make_contrastive_pairs(corpus_data, n_pairs):
    pair_data = []    
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
            ((loc[0], loc[1]), (loc[0], loc[2]), 1)
        )
    
    ## This ensures the same speaker pairs can't be considered more than twice (reverse is possible)
    neg_speaks = set([])
    while len(pair_data) < n_pairs:
        speakers, data = find_negative_pair(corpus_data)
        if speakers not in neg_speaks:
            neg_speaks.add(speakers)
            pair_data.append(data)

    return pair_data

def make_contrastive_triplets(corpus_data, n_triplets):
    triplets = []
    positive_pair_locs = find_positive_pairs(corpus_data)

    if n_triplets > len(positive_pair_locs):
        raise ValueError('Choosing an N_SAMPLES that is higher than possible with the data! Choose a smaller value!')
    
    for i in range(n_triplets):
        spkr = positive_pair_locs[i][0]
        idx1 = positive_pair_locs[i][1]
        idx2 = positive_pair_locs[i][2]

        neg_spkr, neg_idx = find_random_negative(corpus_data, spkr)
        triplet = ((spkr, idx1), (spkr, idx2), (neg_spkr, neg_idx))
        triplets.append(triplet)

    return triplets

def make_contrastive_quadruplets(corpus_data, n_quadruplets):
    quadruplets = []
    positive_pair_locs = find_positive_pairs(corpus_data)

    if n_quadruplets > len(positive_pair_locs):
        raise ValueError('Choosing an N_SAMPLES that is higher than possible with the data! Choose a smaller value!')
    for i in range(n_quadruplets):
        spkr = positive_pair_locs[i][0]
        idx1 = positive_pair_locs[i][1]
        idx2 = positive_pair_locs[i][2]

        (spkrN1, idx3), (spkrN2, idx4) = find_two_random_negatives(corpus_data, spkr)

        quadruplet = (
            (spkr, idx1), 
            (spkr, idx2), 
            (spkrN1, idx3),
            (spkrN2, idx4)
        )
        quadruplets.append(quadruplet)

    return quadruplets

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def write_pairs_pkl(pairs, pairs_path, speaker_spectrograms):
    pairs_data = []

    print('Writing', pairs_path)
    for pair in pairs:
        spectL = speaker_spectrograms[pair[0][0]][pair[0][1]]
        spectR = speaker_spectrograms[pair[1][0]][pair[1][1]]
        label = pair[2]
        pairs_data.append((spectL, spectR, label))

    utils.save(pairs_data, pairs_path)

def write_triplet_pkl(triplets, triplets_path, speaker_spectrograms):
    triplets_data = []

    print('Writing', triplets_path)
    for pair_data in triplets:
        spectA = speaker_spectrograms[pair_data[0][0]][pair_data[0][1]]
        spectP = speaker_spectrograms[pair_data[1][0]][pair_data[1][1]]
        spectN = speaker_spectrograms[pair_data[2][0]][pair_data[2][1]]
        triplets_data.append((spectA, spectP, spectN))

    utils.save(triplets_data, triplets_path)

def write_quadruplet_pkl(quadruplets, quadruplets_path, speaker_spectrograms):
    quadruplets_data = []

    print('Writing', quadruplets_path)
    for quadruplet_locs in quadruplets:
        spectA = speaker_spectrograms[quadruplet_locs[0][0]][quadruplet_locs[0][1]]
        spectP = speaker_spectrograms[quadruplet_locs[1][0]][quadruplet_locs[1][1]]
        spectN1 = speaker_spectrograms[quadruplet_locs[2][0]][quadruplet_locs[2][1]]
        spectN2 = speaker_spectrograms[quadruplet_locs[3][0]][quadruplet_locs[3][1]]
        quadruplets_data.append((spectA, spectP, spectN1, spectN2))

    utils.save(quadruplets_data, quadruplets_path)

def write_pairs_dataset(pairs, pairs_path, speaker_spectrograms):
    print('Writing', pairs_path)
    with tf.io.TFRecordWriter(pairs_path) as writer:
        for pair_data in pairs:
            spect1_b = speaker_spectrograms[pair_data[0][0]][pair_data[0][1]].tobytes()
            spect2_b = speaker_spectrograms[pair_data[1][0]][pair_data[1][1]].tobytes()
            label = pair_data[2]

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'spect1': _bytes_feature(spect1_b),
                        'spect2': _bytes_feature(spect2_b),
                        'label': _float_feature(label)
                    }
                )
            )
            writer.write(example.SerializeToString())

def write_triplets_dataset(triplets, triplets_path, speaker_spectrograms):
    print('Writing', triplets_path)
    with tf.io.TFRecordWriter(triplets_path) as writer:
        for pair_data in triplets:
            spectA_b = speaker_spectrograms[pair_data[0][0]][pair_data[0][1]].tobytes()
            spectP_b = speaker_spectrograms[pair_data[1][0]][pair_data[1][1]].tobytes()
            spectN_b = speaker_spectrograms[pair_data[2][0]][pair_data[2][1]].tobytes()

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'spectA': _bytes_feature(spectA_b),
                        'spectP': _bytes_feature(spectP_b),
                        'spectN': _bytes_feature(spectN_b)
                    }
                )
            )
            writer.write(example.SerializeToString())

def write_quadruplets_dataset(quadruplets, quadruplets_path, speaker_spectrograms):
    print('Writing', quadruplets_path)
    with tf.io.TFRecordWriter(quadruplets_path) as writer:
        for quadruplet_data in quadruplets:
            spectA_b = speaker_spectrograms[quadruplet_data[0][0]][quadruplet_data[0][1]].tobytes()
            spectP_b = speaker_spectrograms[quadruplet_data[1][0]][quadruplet_data[1][1]].tobytes()
            spectN1_b = speaker_spectrograms[quadruplet_data[2][0]][quadruplet_data[2][1]].tobytes()
            spectN2_b = speaker_spectrograms[quadruplet_data[3][0]][quadruplet_data[3][1]].tobytes()

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'spectA': _bytes_feature(spectA_b),
                        'spectP': _bytes_feature(spectP_b),
                        'spectN1': _bytes_feature(spectN1_b),
                        'spectN2': _bytes_feature(spectN2_b)
                    }
                )
            )
            writer.write(example.SerializeToString())


if __name__ == "__main__":
    ### Set variables from config file ###
    PARAMS = utils.config_init(sys.argv)
    audio_dir = os.path.join(os.path.dirname(__file__), PARAMS.PATHS.AUDIO_DIR)
    output_dir = os.path.join(os.path.dirname(__file__), PARAMS.PATHS.OUTPUT_DIR)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    train_spectrogram_path = os.path.join(output_dir, 'speaker_spectrograms_train.pkl')
    test_spectrogram_path = os.path.join(output_dir, 'speaker_spectrograms_test.pkl')
    val_spectrogram_path = os.path.join(output_dir, 'speaker_spectrograms_val.pkl')

    test_pairs_path = os.path.join(output_dir, 'contrastive_pairs_test.pkl')
    test_triplets_path = os.path.join(output_dir, 'contrastive_triplets_test.pkl')
    test_quadruplets_path = os.path.join(output_dir, 'contrastive_quadruplets_test.pkl')

    train_pairs_path = os.path.join(output_dir, 'contrastive_pairs_train.tfrecord')
    train_triplets_path = os.path.join(output_dir, 'contrastive_triplets_train.tfrecord')
    train_quadruplets_path = os.path.join(output_dir, 'contrastive_quadruplets_train.tfrecord')

    val_pairs_path = os.path.join(output_dir, 'contrastive_pairs_val.tfrecord')
    val_triplets_path = os.path.join(output_dir, 'contrastive_triplets_val.tfrecord')
    val_quadruplets_path = os.path.join(output_dir, 'contrastive_quadruplets_val.tfrecord')

    overwrite_datasets = PARAMS.DATA_GENERATOR.OVERWRITE_DATASETS
    train_speaker_spectrograms = utils.load(train_spectrogram_path)
    test_speaker_spectrograms = utils.load(test_spectrogram_path)
    val_speaker_spectrograms = utils.load(val_spectrogram_path)
    
    ### Generate or contrastive pairs ###
    if PARAMS.MODEL.LOSS_TYPE == 'contrastive':
        if overwrite_datasets == 'T' or not os.path.isfile(test_pairs_path):
            logging.info("Generating pairs for contrastive loss...")
            pairs_train = make_contrastive_pairs(
                train_speaker_spectrograms, 
                int(PARAMS.DATA_GENERATOR.N_SAMPLES * (1 - PARAMS.DATA_GENERATOR.TEST_SPLIT - PARAMS.DATA_GENERATOR.VALIDATION_SPLIT)),
            )
            pairs_val = make_contrastive_pairs(
                val_speaker_spectrograms, 
                int(PARAMS.DATA_GENERATOR.N_SAMPLES *  PARAMS.DATA_GENERATOR.VALIDATION_SPLIT),
            )
            pairs_test = make_contrastive_pairs(
                test_speaker_spectrograms, 
                int(PARAMS.DATA_GENERATOR.N_SAMPLES * PARAMS.DATA_GENERATOR.TEST_SPLIT)
            )
            write_pairs_dataset(pairs_train, train_pairs_path, train_speaker_spectrograms)
            write_pairs_dataset(pairs_test, val_pairs_path, test_speaker_spectrograms)
            write_pairs_pkl(pairs_test, test_pairs_path, val_speaker_spectrograms)

    ### Generate or contrastive triplets ###
    if PARAMS.MODEL.LOSS_TYPE == 'triplet':
        if overwrite_datasets == 'T' or not os.path.isfile(test_triplets_path):
            logging.info("Generating triplets for triplet loss...")
            triplets_train = make_contrastive_triplets(
                train_speaker_spectrograms, 
                int(PARAMS.DATA_GENERATOR.N_SAMPLES * (1 - PARAMS.DATA_GENERATOR.TEST_SPLIT - PARAMS.DATA_GENERATOR.VALIDATION_SPLIT)),
            )
            triplets_val = make_contrastive_triplets(
                val_speaker_spectrograms, 
                int(PARAMS.DATA_GENERATOR.N_SAMPLES * PARAMS.DATA_GENERATOR.VALIDATION_SPLIT),
            )
            triplets_test = make_contrastive_triplets(
                test_speaker_spectrograms, 
                int(PARAMS.DATA_GENERATOR.N_SAMPLES * PARAMS.DATA_GENERATOR.TEST_SPLIT),
            )
            write_triplets_dataset(triplets_train, train_triplets_path, train_speaker_spectrograms)
            write_triplets_dataset(triplets_val, val_triplets_path, val_speaker_spectrograms)
            write_triplet_pkl(triplets_test, test_triplets_path, test_speaker_spectrograms)

    ### Generate or contrastive quadruplets ###
    if PARAMS.MODEL.LOSS_TYPE == 'quadruplet':
        if overwrite_datasets == 'T' or not os.path.isfile(test_quadruplets_path):
            logging.info("Generating quadruplets for quadruplet loss...")
            quadruplets_train = make_contrastive_quadruplets(
                train_speaker_spectrograms, 
                int(PARAMS.DATA_GENERATOR.N_SAMPLES * (1 - PARAMS.DATA_GENERATOR.TEST_SPLIT)),
            )
            quadruplets_val = make_contrastive_quadruplets(
                val_speaker_spectrograms, 
                int(PARAMS.DATA_GENERATOR.N_SAMPLES * PARAMS.DATA_GENERATOR.TEST_SPLIT),
            ),
            quadruplets_test = make_contrastive_quadruplets(
                test_speaker_spectrograms, 
                int(PARAMS.DATA_GENERATOR.N_SAMPLES * PARAMS.DATA_GENERATOR.TEST_SPLIT),
            )

            write_triplets_dataset(triplets_train, train_triplets_path, train_speaker_spectrograms)
            write_triplets_dataset(quadruplets_val, val_quadruplets_path, val_speaker_spectrograms)
            write_quadruplet_pkl(quadruplets_test, test_quadruplets_path, test_speaker_spectrograms)
