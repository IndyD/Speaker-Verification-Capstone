import logging
import os
import pickle
import pprint
import random 
import sys
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping

import generate_data
import tf_models
import utils

import pdb

logging.basicConfig(level=logging.DEBUG)

def calculate_EER(dist, labels):
    # scale distances so EER works
    preds = dist / dist.max()
    fpr, tpr, _ = roc_curve(labels, preds, pos_label=0)
    fnr = 1 - tpr
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return EER

def transfer_embedding_layers(embedding_layers, IMG_SHAPE):
    img_input = Input(IMG_SHAPE)
    emb_model = Sequential([Input(IMG_SHAPE)] + embedding_layers.layers)
    trained_embedding_model = Model(inputs=img_input, outputs=emb_model(img_input))
    return trained_embedding_model

def compute_contrastive_embeddings(embedding_model, anchors, positives, negatives):
    embeddings_a = embedding_model.predict(anchors)
    embeddings_p = embedding_model.predict(positives)
    embeddings_n = embedding_model.predict(negatives)

    pos_pairs = zip(embeddings_a, embeddings_p)
    dist_p = [np.linalg.norm(emb[0] - emb[1]) for emb in pos_pairs]

    neg_pairs = zip(embeddings_a, embeddings_n)
    dist_n = [np.linalg.norm(emb[0] - emb[1]) for emb in neg_pairs]

    return dist_p, dist_n


def compute_labelled_distances(embedding_model, anchors, positives, negatives):
    dist_p, dist_n = compute_contrastive_embeddings(embedding_model, anchors, positives, negatives)

    dist = np.array(dist_p + dist_n)
    labels = np.concatenate((np.ones(len(dist_p)), np.zeros(len(dist_n))))

    return dist, labels

def mine_triplets(embedding_model, PARAMS):
    semihard_triplets = []
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    speaker_spectograms = utils.load(os.path.join(output_dir, 'speaker_spectograms.pkl'))
    positive_pair_locs = generate_data.find_positive_pairs(speaker_spectograms)

    i = PARAMS.DATA_GENERATOR.N_SAMPLES

    ## keep going down the list and adding the semi-hard triplets
    while len(semihard_triplets) < PARAMS.DATA_GENERATOR.N_SAMPLES:
        spkr = positive_pair_locs[i][0]
        idx1 = positive_pair_locs[i][1]
        idx2 = positive_pair_locs[i][2]

        negative = generate_data.find_random_negative(speaker_spectograms, spkr)
        cand_triplet = (speaker_spectograms[spkr][idx1], speaker_spectograms[spkr][idx2], negative)
        input_a = np.expand_dims(cand_triplet[0], axis=0)
        input_p = np.expand_dims(cand_triplet[1], axis=0)
        input_n = np.expand_dims(cand_triplet[2], axis=0)

        dist_p, dist_n = compute_contrastive_embeddings(embedding_model, input_a, input_p, input_n)

        if (dist_p[0] < dist_n[0]) and (dist_n[0] < dist_p[0] + PARAMS.TRAINING.MARGIN):
            semihard_triplets.append(cand_triplet)
        i += 1
    return semihard_triplets

def mine_quadruplets(embedding_model, PARAMS):
    semihard_quadruplets = []
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    speaker_spectograms = utils.load(os.path.join(output_dir, 'speaker_spectograms.pkl'))
    positive_pair_locs = generate_data.find_positive_pairs(speaker_spectograms)

    i = PARAMS.DATA_GENERATOR.N_SAMPLES

    ## keep going down the list and adding the semi-hard triplets
    while len(semihard_quadruplets) < PARAMS.DATA_GENERATOR.N_SAMPLES:
        spkr = positive_pair_locs[i][0]
        idx1 = positive_pair_locs[i][1]
        idx2 = positive_pair_locs[i][2]

        negativeA, negativeB = generate_data.find_two_random_negatives(speaker_spectograms, spkr)

        cand_quadruplet = (
            speaker_spectograms[spkr][idx1], 
            speaker_spectograms[spkr][idx2], 
            negativeA,
            negativeB
        )

        input_a = np.expand_dims(cand_quadruplet[0], axis=0)
        input_p = np.expand_dims(cand_quadruplet[1], axis=0)
        input_n1 = np.expand_dims(cand_quadruplet[2], axis=0)
        input_n2 = np.expand_dims(cand_quadruplet[3], axis=0)

        dist_pn1, dist_n1 = compute_contrastive_embeddings(embedding_model, input_a, input_p, input_n1)
        dist_pn2, dist_n2 = compute_contrastive_embeddings(embedding_model, input_a, input_p, input_n2)

        if (dist_pn1[0] < dist_n1[0]) and (dist_pn2[0] < dist_n2[0]) and (dist_n1[0] < dist_pn1[0] + PARAMS.TRAINING.MARGIN) and (dist_n2[0] < dist_pn2[0] + PARAMS.TRAINING.MARGIN):
            semihard_quadruplets.append(cand_quadruplet)
        i += 1
    return semihard_quadruplets

def train_triplet_model(model, triplets, PARAMS):
    ## train-test split
    random.shuffle(triplets)
    test_split = int(len(triplets) * PARAMS.DATA_GENERATOR.TEST_SPLIT)
    val_split = test_split + int(len(triplets) * PARAMS.DATA_GENERATOR.VALIDATION_SPLIT)
    triplets_test = triplets[:test_split]
    triplets_val = triplets[test_split:val_split]
    triplets_train = triplets[val_split:]

    ####  split and normalize the spectograms  ####
    train_a = np.array([triplet[0] for triplet in triplets_train])
    train_p = np.array([triplet[1] for triplet in triplets_train])
    train_n = np.array([triplet[2] for triplet in triplets_train])

    val_a = np.array([triplet[0] for triplet in triplets_val])
    val_p = np.array([triplet[1] for triplet in triplets_val])
    val_n = np.array([triplet[2] for triplet in triplets_val])

    ####  compile and fit model  ####
    model.compile(optimizer=PARAMS.TRAINING.OPTIMIZER)
    history = model.fit(
        [train_a, train_p, train_n],
        validation_data=([val_a, val_p, val_n]),
        epochs=PARAMS.TRAINING.EPOCHS,
        verbose=1,
        callbacks=[EarlyStopping(patience=PARAMS.TRAINING.EARLY_STOP_ROUNDS)],
    )
    return model, triplets_test

def train_quadruplet_model(model, quadruplets, PARAMS):
    ## train-test split
    random.shuffle(quadruplets)
    test_split = int(len(quadruplets) * PARAMS.DATA_GENERATOR.TEST_SPLIT)
    val_split = test_split + int(len(quadruplets) * PARAMS.DATA_GENERATOR.VALIDATION_SPLIT)
    quadruplets_test = quadruplets[:test_split]
    quadruplets_val = quadruplets[test_split:val_split]
    quadruplets_train = quadruplets[val_split:]

    ####  split and normalize the spectograms  ####
    train_a = np.array([quadruplet[0] for quadruplet in quadruplets_train])
    train_p = np.array([quadruplet[1] for quadruplet in quadruplets_train])
    train_n1 = np.array([quadruplet[2] for quadruplet in quadruplets_train])
    train_n2 = np.array([quadruplet[3] for quadruplet in quadruplets_train])

    val_a = np.array([quadruplet[0] for quadruplet in quadruplets_val])
    val_p = np.array([quadruplet[1] for quadruplet in quadruplets_val])
    val_n1 = np.array([quadruplet[2] for quadruplet in quadruplets_val])
    val_n2 = np.array([quadruplet[3] for quadruplet in quadruplets_val])

    ####  compile and fit model  ####
    model.compile(optimizer=PARAMS.TRAINING.OPTIMIZER)
    history = model.fit(
        [train_a, train_p, train_n1, train_n2],
        validation_data=([val_a, val_p, val_n1, val_n2]),
        epochs=PARAMS.TRAINING.EPOCHS,
        verbose=1,
        callbacks=[EarlyStopping(patience=PARAMS.TRAINING.EARLY_STOP_ROUNDS)],

    )
    return model, quadruplets_test



def run_siamsese_model(IMG_SHAPE, PARAMS):
    model = tf_models.build_siamese_model(IMG_SHAPE)
    (pairs, labels) = utils.load(
        os.path.join(os.path.dirname(__file__), 'output', 'contrastive_pairs.pkl')
    )

    pairs_train, pairs_test, labels_train, labels_test = train_test_split(
        pairs, labels, test_size=PARAMS.DATA_GENERATOR.TEST_SPLIT, random_state=123
    )
    val_split = (1 / (1.0 - PARAMS.DATA_GENERATOR.TEST_SPLIT)) * PARAMS.DATA_GENERATOR.VALIDATION_SPLIT
    pairs_train, pairs_val, labels_train, labels_val = train_test_split(
        pairs_train, labels_train, test_size=val_split, random_state=123
    )

    ####  split and normalize the spectograms  ####
    pairs_train_l = np.array([pair[0] for pair in pairs_train])
    pairs_train_r = np.array([pair[1] for pair in pairs_train])
    pairs_test_l = np.array([pair[0] for pair in pairs_test])
    pairs_test_r = np.array([pair[1] for pair in pairs_test])
    pairs_val_l = np.array([pair[0] for pair in pairs_val])
    pairs_val_r = np.array([pair[1] for pair in pairs_val])
    labels_train = tf.cast(np.array(labels_train), tf.float32)
    labels_test = tf.cast(np.array(labels_test), tf.float32)
    labels_val = tf.cast(np.array(labels_val), tf.float32)

    ####  compile and fit model  ####
    model.compile(loss=tf_models.contrastive_loss_with_margin(margin=PARAMS.TRAINING.MARGIN), optimizer=PARAMS.TRAINING.OPTIMIZER)
    logging.info("Training contrastive pair model...")

    history = model.fit(
        [pairs_train_l, pairs_train_r], labels_train,
        validation_data=([pairs_val_l, pairs_val_r], labels_val),
        epochs=PARAMS.TRAINING.EPOCHS,
        verbose=1,
        callbacks=[EarlyStopping(patience=PARAMS.TRAINING.EARLY_STOP_ROUNDS)],
    )

    dist_test = model.predict([pairs_test_l, pairs_test_r])
    return dist_test, labels_test



def run_triplet_model(IMG_SHAPE, PARAMS):
    model = tf_models.build_triplet_model(IMG_SHAPE, PARAMS)
    triplets = utils.load(
        os.path.join(os.path.dirname(__file__), 'output', 'contrastive_triplets.pkl')
    )

    #### Initial training on all tripplets
    logging.info("Training tripet loss model on all triplets...")
    model, triplets_test = train_triplet_model(model, triplets, PARAMS)
    embedding_layers = model.layers[3]
    embedding_model = transfer_embedding_layers(embedding_layers, IMG_SHAPE)

    #### Calculate initial EER
    test_a = np.array([triplet[0] for triplet in triplets_test])
    test_p = np.array([triplet[1] for triplet in triplets_test])
    test_n = np.array([triplet[2] for triplet in triplets_test])

    ####  Transfer learning- take inital embedding layers and score pairs similarly to contrastive loss
    dist_test_initial, labels_test_initial = compute_labelled_distances(embedding_model, test_a, test_p, test_n)
    if PARAMS.TRAINING.MINE_SEMIHARD == 'T':
        initial_EER = calculate_EER(dist_test_initial, labels_test_initial)
        logging.info("<<<< Initial EER: {EER} >>>>...".format(EER=initial_EER))

        #### Triplet mining
        logging.info("Mining semi-hard triplets...")
        semihard_triplets = mine_triplets(embedding_model, PARAMS)

        #### Train again with mined triplets
        logging.info("Training tripet loss model on semi-hard triplets...")
        if PARAMS.TRAINING.SEMIHARD_FERSH_MODEL == 'T':
            model = tf_models.build_triplet_model(IMG_SHAPE, PARAMS)
            model, _ = train_triplet_model(model, semihard_triplets, PARAMS)
        else:
            model, _ = train_triplet_model(model, semihard_triplets, PARAMS)

        ####  Transfer learning- take final embedding layers and score pairs similarly to contrastive loss
        dist_test, labels_test = compute_labelled_distances(embedding_model, test_a, test_p, test_n)
    else:
        dist_test = dist_test_initial
        labels_test = labels_test_initial

    return dist_test, labels_test



def run_quadruplet_model(IMG_SHAPE, PARAMS):
    model = tf_models.build_quadruplet_model(IMG_SHAPE, PARAMS)
    quadruplets = utils.load(
        os.path.join(os.path.dirname(__file__), 'output', 'contrastive_quadruplets.pkl')
    )

    #### Initial training on all quadruplets
    logging.info("Training quadruplet loss model on all quadruplets...")
    model, quadruplets_test = train_quadruplet_model(model, quadruplets, PARAMS)
    embedding_layers = model.layers[4]
    embedding_model = transfer_embedding_layers(embedding_layers, IMG_SHAPE)

    ####  Transfer learning- take inital embedding layers and score pairs similarly to contrastive loss
    test_a = np.array([quadruplet[0] for quadruplet in quadruplets_test])
    test_p = np.array([quadruplet[1] for quadruplet in quadruplets_test])
    test_n1 = np.array([quadruplet[2] for quadruplet in quadruplets_test])

    dist_test_initial, labels_test_initial = compute_labelled_distances(embedding_model, test_a, test_p, test_n1)
    if PARAMS.TRAINING.MINE_SEMIHARD == 'T':
        initial_EER = calculate_EER(dist_test_initial, labels_test_initial)
        logging.info("<<<< Initial EER: {EER} >>>>...".format(EER=initial_EER))

        #### Quadruplet mining
        logging.info("Mining semi-hard quadruplets...")
        semihard_quadruplets = mine_quadruplets(embedding_model, PARAMS)

        #### Train again with mined quadruplets
        logging.info("Training quadruplet loss model on semi-hard quadruplets...")
        if PARAMS.TRAINING.SEMIHARD_FERSH_MODEL == 'T':
            model = tf_models.build_triplet_model(IMG_SHAPE, PARAMS)
            model, _ = train_triplet_model(model, semihard_quadruplets, PARAMS)
        else:
            model, _ = train_triplet_model(model, semihard_quadruplets, PARAMS)

        ####  Transfer learning- take embedding layers and score pairs similarly to contrastive loss
        embedding_layers = model.layers[4]
        embedding_model = transfer_embedding_layers(embedding_layers, IMG_SHAPE)
        dist_test, labels_test = compute_labelled_distances(embedding_model, test_a, test_p, test_n1)
    else:
        dist_test = dist_test_initial
        labels_test = labels_test_initial

    return dist_test, labels_test






if __name__ == '__main__':
    PARAMS = utils.config_init(sys.argv)
    IMG_SHAPE = (
        PARAMS.DATA_GENERATOR.N_MELS,
        PARAMS.DATA_GENERATOR.MAX_FRAMES,
        1
    )

    ####  build model  ####
    if PARAMS.TRAINING.LOSS_TYPE == 'contrastive':
        dist_test, labels_test = run_siamsese_model(IMG_SHAPE, PARAMS)
    if PARAMS.TRAINING.LOSS_TYPE == 'triplet':
        dist_test, labels_test = run_triplet_model(IMG_SHAPE, PARAMS)
    if PARAMS.TRAINING.LOSS_TYPE == 'quadruplet':
        dist_test, labels_test = run_quadruplet_model(IMG_SHAPE, PARAMS)

    ####  Find EER   ####
    EER = calculate_EER(dist_test, labels_test)

    print('#'*80)
    print('Config parameters:')
    pprint.pprint(PARAMS)
    print('-'*60)
    print('<<<<<  The EER is:  {EER} !  >>>>>'.format(EER=EER))
    print('#'*80)
