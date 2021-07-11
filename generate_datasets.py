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
    data = ((spk1, spk1_idx), (spk2, spk2_idx), [0])

    return speakers, data

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
            ((loc[0], loc[1]), (loc[0], loc[2]), [1])
        )
    
    ## This ensures the same speaker pairs can't be considered more than twice (reverse is possible)
    neg_speaks = set([])
    while len(pair_data) < n_pairs:
        speakers, data = find_negative_pair(corpus_data)
        if speakers not in neg_speaks:
            neg_speaks.add(speakers)
            pair_data.append(data)

    return pair_data

def find_random_negative(corpus_data, exclude):
    candidates = [cand for cand in corpus_data if cand != exclude]
    speaker_id = random.choice(candidates)
    random_negative = random.choice(range(len(corpus_data[speaker_id])))
    return speaker_id, random_negative

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

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_datasets(items, data_type, output_dir, speaker_spectrograms, PARAMS):
    items_train, items_test, items_val = utils.test_train_val_split(
        items, PARAMS.DATA_GENERATOR.TEST_SPLIT, PARAMS.DATA_GENERATOR.VALIDATION_SPLIT
    )

    train_path = os.path.join(output_dir, 'contrastive_' + data_type + '_train.tfrecord')
    val_path = os.path.join(output_dir, 'contrastive_' + data_type + '_val.tfrecord')
    test_path = os.path.join(output_dir, 'contrastive_' + data_type + '_test.pkl')

    if data_type == 'pairs':
        write_pairs_dataset(items_train, train_path, speaker_spectrograms)
        write_pairs_dataset(items_val, val_path, speaker_spectrograms)
        write_pairs_dataset(items_test, test_path, speaker_spectrograms)

    elif data_type == 'triplets':
        write_triplets_dataset(items_train, train_path, speaker_spectrograms)
        write_triplets_dataset(items_val, val_path, speaker_spectrograms)
        write_triplets_dataset(items_test, test_path, speaker_spectrograms)

    else:
        raise ValueError('Invalid datatype')

def write_pairs_dataset(pairs, pairs_path, speaker_spectrograms):
    print('Writing', pairs_path)
    with tf.io.TFRecordWriter(pairs_path) as writer:
        for pair_data in pairs:
            spect1_b = speaker_spectrograms[pair_data[0][0]][pair_data[0][1]].tobytes()
            spect2_b = speaker_spectrograms[pair_data[1][0]][pair_data[1][1]].tobytes()
            label_b = np.array(pair_data[2]).tobytes()

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'spect1': _bytes_feature(spect1_b),
                        'spect2': _bytes_feature(spect2_b),
                        'label': _bytes_feature(label_b)
                    }
                )
            )
            writer.write(example.SerializeToString())

def write_triplet_pkl(triplets, triplets_path, speaker_spectrograms):
    triplets_data = []

    print('Writing', triplets_path)
    for pair_data in triplets:
        spectA = speaker_spectrograms[pair_data[0][0]][pair_data[0][1]]
        spectP = speaker_spectrograms[pair_data[1][0]][pair_data[1][1]]
        spectN = speaker_spectrograms[pair_data[2][0]][pair_data[2][1]]
        triplets_data.append((spectA, spectP, spectN))

    utils.save(triplets_data, triplets_path)

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

if __name__ == "__main__":
    ### Set variables from config file ###
    PARAMS = utils.config_init(sys.argv)
    audio_dir = os.path.join(os.path.dirname(__file__), PARAMS.PATHS.AUDIO_DIR)
    output_dir = os.path.join(os.path.dirname(__file__), PARAMS.PATHS.OUTPUT_DIR)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    spectrogram_path = os.path.join(output_dir, 'speaker_spectrograms.pkl')
    pairs_path = os.path.join(output_dir, 'contrastive_pairs.pkl')
    triplets_path = os.path.join(output_dir, 'contrastive_triplets.pkl')
    quadruplets_path = os.path.join(output_dir, 'contrastive_quadruplets.pkl')
    overwrite_datasets = PARAMS.DATA_GENERATOR.OVERWRITE_DATASETS
    speaker_spectrograms = utils.load(spectrogram_path)
    
    ### Generate or contrastive pairs ###
    if PARAMS.MODEL.LOSS_TYPE == 'contrastive':
        if overwrite_datasets == 'T' or not os.path.isfile(pairs_path):
            logging.info("Generating pairs for contrastive loss...")
            pairs = make_contrastive_pairs(
                speaker_spectrograms, 
                PARAMS.DATA_GENERATOR.N_SAMPLES
            )
            write_datasets(pairs, 'pairs', output_dir, speaker_spectrograms, PARAMS)
            utils.save(pairs, pairs_path)

    ### Generate or contrastive triplets ###
    if PARAMS.MODEL.LOSS_TYPE == 'triplet':
        if overwrite_datasets == 'T' or not os.path.isfile(triplets_path):
            logging.info("Generating triplets for triplet loss...")
            triplets = make_contrastive_triplets(
                speaker_spectrograms, 
                PARAMS.DATA_GENERATOR.N_SAMPLES,
            )
            write_datasets(triplets, 'triplets', output_dir, speaker_spectrograms, PARAMS)
            utils.save(triplets, triplets_path)

    ### Generate or contrastive quadruplets ###
    if PARAMS.MODEL.LOSS_TYPE == 'quadruplet':
        if overwrite_datasets == 'T' or not os.path.isfile(quadruplets_path):
            logging.info("Generating quadruplets for quadruplet loss...")
            quadruplets = make_contrastive_quadruplets(
                speaker_spectrograms, 
                PARAMS.DATA_GENERATOR.N_SAMPLES,
            )
            utils.save(quadruplets, quadruplets_path)
