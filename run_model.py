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
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay

import generate_data
import tf_models
import utils

import pdb

logging.basicConfig(level=logging.DEBUG)


def set_optimizer(PARAMS):
    lr_schedule = ExponentialDecay(
        PARAMS.TRAINING.LEARNING_RATE,
        decay_steps=PARAMS.TRAINING.LEARNING_DECAY_STEPS,
        decay_rate=PARAMS.TRAINING.LEARNING_DECAY_RATE,
        staircase=False
    )

    if PARAMS.TRAINING.OPTIMIZER == 'adam':
        opt = Adam(
            learning_rate=lr_schedule, 
        )
        print(1)
    elif PARAMS.TRAINING.OPTIMIZER == 'sgd':
        opt = SGD(
            learning_rate=lr_schedule, 
            momentum=PARAMS.TRAINING.MOMENTUM
        )
    else:
        raise ValueError('ERROR: Invalid PARAMS.TRAINING.OPTIMIZER argument!')

    return opt

def set_crossentropy_optimizer(PARAMS):
    lr_schedule = ExponentialDecay(
        PARAMS.TRAINING.CROSSENTROPY_LEARNING_RATE,
        decay_steps=PARAMS.TRAINING.CROSSENTROPY_LEARNING_DECAY_STEPS,
        decay_rate=PARAMS.TRAINING.CROSSENTROPY_LEARNING_DECAY_RATE,
        staircase=False
    )

    if PARAMS.TRAINING.CROSSENTROPY_OPTIMIZER == 'adam':
        opt = Adam(
            learning_rate=lr_schedule, 
        )
        print(1)
    elif PARAMS.TRAINING.CROSSENTROPY_OPTIMIZER == 'sgd':
        opt = SGD(
            learning_rate=lr_schedule, 
            momentum=PARAMS.TRAINING.CROSSENTROPY_MOMENTUM
        )
    else:
        raise ValueError('ERROR: Invalid PARAMS.TRAINING.OPTIMIZER argument!')

    return opt

def calculate_EER(dist, labels):
    # scale distances so EER works
    preds = dist / dist.max()
    fpr, tpr, _ = roc_curve(labels, preds, pos_label=0)
    fnr = 1 - tpr
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return EER

def transfer_embedding_layers(embedding_layers, IMG_SHAPE):
    img_input = Input(IMG_SHAPE)
    emb_model = Sequential([Input(IMG_SHAPE)] + embedding_layers)
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
    output_dir = os.path.join(os.path.dirname(__file__), PARAMS.PATHS.OUTPUT_DIR)
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

        if (dist_p[0] < dist_n[0]) and (dist_n[0] < dist_p[0] + PARAMS.MODEL.MARGIN):
            semihard_triplets.append(cand_triplet)
        i += 1
    return semihard_triplets

def mine_quadruplets(embedding_model, PARAMS):
    semihard_quadruplets = []
    output_dir = os.path.join(os.path.dirname(__file__), PARAMS.PATHS.OUTPUT_DIR)
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
    triplets_test = triplets[:test_split]
    triplets_train = triplets[test_split:]

    ####  split and normalize the spectograms  ####
    train_a = np.array([triplet[0] for triplet in triplets_train])
    train_p = np.array([triplet[1] for triplet in triplets_train])
    train_n = np.array([triplet[2] for triplet in triplets_train])

    ####  compile and fit model  ####
    model.compile(optimizer=set_optimizer(PARAMS))
    history = model.fit(
        [train_a, train_p, train_n],
        validation_split=PARAMS.DATA_GENERATOR.VALIDATION_SPLIT,
        epochs=PARAMS.TRAINING.EPOCHS,
        verbose=1,
        callbacks=[EarlyStopping(patience=PARAMS.TRAINING.EARLY_STOP_ROUNDS)],
    )
    return model, triplets_test

def train_quadruplet_model(model, quadruplets, PARAMS):
    ## train-test split
    random.shuffle(quadruplets)
    test_split = int(len(quadruplets) * PARAMS.DATA_GENERATOR.TEST_SPLIT)
    quadruplets_test = quadruplets[:test_split]
    quadruplets_train = quadruplets[test_split:]

    ####  split and normalize the spectograms  ####
    train_a = np.array([quadruplet[0] for quadruplet in quadruplets_train])
    train_p = np.array([quadruplet[1] for quadruplet in quadruplets_train])
    train_n1 = np.array([quadruplet[2] for quadruplet in quadruplets_train])
    train_n2 = np.array([quadruplet[3] for quadruplet in quadruplets_train])

    ####  compile and fit model  ####
    model.compile(optimizer=set_optimizer(PARAMS))
    history = model.fit(
        [train_a, train_p, train_n1, train_n2],
        validation_split=PARAMS.DATA_GENERATOR.VALIDATION_SPLIT,
        epochs=PARAMS.TRAINING.EPOCHS,
        verbose=1,
        callbacks=[EarlyStopping(patience=PARAMS.TRAINING.EARLY_STOP_ROUNDS)],

    )
    return model, quadruplets_test

def run_cross_entropy_model(IMG_SHAPE, PARAMS):
    output_dir = os.path.join(os.path.dirname(__file__), PARAMS.PATHS.OUTPUT_DIR)
    speaker_spectograms = utils.load(os.path.join(output_dir, 'speaker_spectograms.pkl'))
    n_classes = len(speaker_spectograms.keys())
    X = []
    y = []

    ## This changes the label to integers so cross-entropy loss works
    for i, spk in enumerate(speaker_spectograms.keys()):
        for spect in speaker_spectograms[spk]:
            X.append(spect)
            y.append(i)

    X = np.array(X)
    y = np.array(y)

    model = tf_models.build_crossentropy_model(n_classes, IMG_SHAPE, PARAMS)

    model.compile(
        loss='sparse_categorical_crossentropy', 
        optimizer=set_crossentropy_optimizer(PARAMS)
    )
    logging.info("Training cross-entropy model to pretrain for distnace metric models...")

    history = model.fit(
        X, y,
        validation_split=PARAMS.DATA_GENERATOR.VALIDATION_SPLIT,
        epochs=PARAMS.TRAINING.CROSSENTROPY_EPOCHS,
        verbose=1,
        callbacks=[EarlyStopping(patience=PARAMS.TRAINING.CROSSENTROPY_EARLY_STOP_ROUNDS)],
    )
    return model


def run_siamsese_model(IMG_SHAPE, PARAMS):
    model = tf_models.build_siamese_model(IMG_SHAPE, PARAMS)
    (pairs, labels) = utils.load(
        os.path.join(os.path.dirname(__file__), PARAMS.PATHS.OUTPUT_DIR, 'contrastive_pairs.pkl')
    )
    pairs_train, pairs_test, labels_train, labels_test = train_test_split(
        pairs, labels, test_size=PARAMS.DATA_GENERATOR.TEST_SPLIT, random_state=123
    )

    ####  split and normalize the spectograms  ####
    pairs_train_l = np.array([pair[0] for pair in pairs_train])
    pairs_train_r = np.array([pair[1] for pair in pairs_train])
    pairs_test_l = np.array([pair[0] for pair in pairs_test])
    pairs_test_r = np.array([pair[1] for pair in pairs_test])
    labels_train = tf.cast(np.array(labels_train), tf.float32)
    labels_test = tf.cast(np.array(labels_test), tf.float32)

    ####  compile and fit model  ####
    model.compile(
        loss=tf_models.contrastive_loss_with_margin(margin=PARAMS.MODEL.MARGIN), 
        optimizer=set_optimizer(PARAMS)
    )
    logging.info("Training contrastive pair model...")

    history = model.fit(
        [pairs_train_l, pairs_train_r], labels_train,
        validation_split=PARAMS.DATA_GENERATOR.VALIDATION_SPLIT,
        epochs=PARAMS.TRAINING.EPOCHS,
        verbose=1,
        callbacks=[EarlyStopping(patience=PARAMS.TRAINING.EARLY_STOP_ROUNDS)],
    )

    dist_test = model.predict([pairs_test_l, pairs_test_r])
    return dist_test, labels_test



def run_triplet_model(IMG_SHAPE, PARAMS):
    model = tf_models.build_triplet_model(IMG_SHAPE, PARAMS)
    triplets = utils.load(
        os.path.join(os.path.dirname(__file__), PARAMS.PATHS.OUTPUT_DIR, 'contrastive_triplets.pkl')
    )

    #### Initial training on all tripplets
    logging.info("Training tripet loss model on all triplets...")
    model, triplets_test = train_triplet_model(model, triplets, PARAMS)
    embedding_layers = model.layers[3].layers
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
        if PARAMS.TRAINING.SEMIHARD_FRESH_MODEL == 'T':
            model = tf_models.build_triplet_model(IMG_SHAPE, PARAMS)
            model, _ = train_triplet_model(model, semihard_triplets, PARAMS)
        else:
            model, _ = train_triplet_model(model, semihard_triplets, PARAMS)

        ####  Transfer learning- take final embedding layers and score pairs similarly to contrastive loss
        embedding_layers = model.layers[3].layers
        embedding_model = transfer_embedding_layers(embedding_layers, IMG_SHAPE)
        dist_test, labels_test = compute_labelled_distances(embedding_model, test_a, test_p, test_n)
    else:
        dist_test = dist_test_initial
        labels_test = labels_test_initial

    return dist_test, labels_test



def run_quadruplet_model(IMG_SHAPE, PARAMS):
    model = tf_models.build_quadruplet_model(IMG_SHAPE, PARAMS)
    quadruplets = utils.load(
        os.path.join(os.path.dirname(__file__), PARAMS.PATHS.OUTPUT_DIR, 'contrastive_quadruplets.pkl')
    )

    #### Initial training on all quadruplets
    logging.info("Training quadruplet loss model on all quadruplets...")
    model, quadruplets_test = train_quadruplet_model(model, quadruplets, PARAMS)
    embedding_layers = model.layers[4].layers
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
        if PARAMS.TRAINING.SEMIHARD_FRESH_MODEL == 'T':
            model = tf_models.build_triplet_model(IMG_SHAPE, PARAMS)
            model, _ = train_triplet_model(model, semihard_quadruplets, PARAMS)
        else:
            model, _ = train_triplet_model(model, semihard_quadruplets, PARAMS)

        ####  Transfer learning- take embedding layers and score pairs similarly to contrastive loss
        embedding_layers = model.layers[4].layers
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

    if PARAMS.MODEL.CROSSENTROPY_PRETRAIN == 'T':
        model = run_cross_entropy_model(IMG_SHAPE, PARAMS)

        triplets = utils.load(
            os.path.join(os.path.dirname(__file__), PARAMS.PATHS.OUTPUT_DIR, 'contrastive_triplets.pkl')
        )
        random.shuffle(triplets)
        test_split = int(len(triplets) * PARAMS.DATA_GENERATOR.TEST_SPLIT)
        triplets_test = triplets[:test_split]

        test_a = np.array([triplet[0] for triplet in triplets_test])
        test_p = np.array([triplet[1] for triplet in triplets_test])
        test_n = np.array([triplet[2] for triplet in triplets_test])

        embedding_layers = model.layers[:-1]
        embedding_model = transfer_embedding_layers(embedding_layers, IMG_SHAPE)

        dist_test_crossentropy, labels_test_crossentropy = compute_labelled_distances(embedding_model, test_a, test_p, test_n)
        crossentropy_EER = calculate_EER(dist_test_crossentropy, labels_test_crossentropy)
        logging.info("<<<< Cross-entropy pretrain EER: {EER} >>>>...".format(EER=crossentropy_EER))


    ####  build model  ####
    if PARAMS.MODEL.LOSS_TYPE == 'contrastive':
        dist_test, labels_test = run_siamsese_model(IMG_SHAPE, PARAMS)
    if PARAMS.MODEL.LOSS_TYPE == 'triplet':
        dist_test, labels_test = run_triplet_model(IMG_SHAPE, PARAMS)
    if PARAMS.MODEL.LOSS_TYPE == 'quadruplet':
        dist_test, labels_test = run_quadruplet_model(IMG_SHAPE, PARAMS)

    ####  Find EER   ####
    EER = calculate_EER(dist_test, labels_test)

    print('#'*80)
    print('Config parameters:')
    pprint.pprint(PARAMS)
    print('-'*60)
    print('<<<<<  The EER is:  {EER} !  >>>>>'.format(EER=EER))
    print('#'*80)
