import logging
import os
import pickle
import pprint
import random 
import sys
import tensorflow as tf
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD, Adamax, Nadam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

import generate_datasets
import tf_models
import utils

import pdb

logging.basicConfig(level=logging.DEBUG)


def test_train_val_split(items, test_split, val_split, seed=123):
    random.Random(seed).shuffle(items)
    test_split_idx = int(len(items) * PARAMS.DATA_GENERATOR.TEST_SPLIT)
    val_split_idx = int(len(items) * PARAMS.DATA_GENERATOR.TEST_SPLIT * PARAMS.DATA_GENERATOR.VALIDATION_SPLIT)
    test = items[:test_split_idx]
    validation = items[test_split_idx:val_split_idx]
    train = items[val_split_idx:]
    return train, test, validation

def set_optimizer(OPTIMIZER, LEARNING_RATE, LEARNING_DECAY_RATE, LEARNING_DECAY_STEPS, MOMENTUM, BETA_1, BETA_2):
    lr_schedule = ExponentialDecay(
        LEARNING_RATE,
        decay_steps=LEARNING_DECAY_STEPS,
        decay_rate=LEARNING_DECAY_RATE,
        staircase=False
    )

    if OPTIMIZER == 'adam':
        opt = Adam(
            learning_rate=lr_schedule, 
            #learning_rate=LEARNING_RATE, 
            beta_1=BETA_1,
            beta_2=BETA_2,
        )
    elif OPTIMIZER == 'sgd':
        opt = SGD(
            learning_rate=lr_schedule, 
            #learning_rate=LEARNING_RATE, 
            momentum=MOMENTUM
        )
    elif OPTIMIZER == 'adamax':
        opt = Adamax(
            learning_rate=lr_schedule, 
            #learning_rate=LEARNING_RATE, 
            beta_1=BETA_1,
            beta_2=BETA_2,
        )
    elif OPTIMIZER == 'nadam':
        opt = Nadam(
            learning_rate=LEARNING_RATE, 
            beta_1=BETA_1,
            beta_2=BETA_2,
        )
    else:
        raise ValueError('ERROR: Invalid PARAMS.TRAINING.OPTIMIZER argument!')

    return opt


def calculate_EER(dist, labels):
    fpr, tpr, threshold = roc_curve(labels, dist, pos_label=0)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    preds = np.where(dist >= eer_threshold, 0, 1)
    accuracy = accuracy_score(labels, preds)

    return EER, eer_threshold, accuracy
'''
def calculate_EER(dist, y_true):
    fpr = []
    tpr = []
    thresholds = np.sort(dist)[::-1]

    for threshold in thresholds:
        y_pred = np.where(dist >= threshold, 0, 1)
        
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        
        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))
    
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    return EER, eer_threshold
'''

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

def _decode_img(img_bytes, IMG_SHAPE):
    img = tf.io.decode_raw(img_bytes, tf.float32)
    img.set_shape([IMG_SHAPE[0] * IMG_SHAPE[1] * IMG_SHAPE[2]])
    img = tf.reshape(img, IMG_SHAPE)
    return img

def _read_pair_tfrecord(serialized_example):
    feature_description = {
        'spect1': tf.io.FixedLenFeature((), tf.string),
        'spect2': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.float32),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    
    spect1 = _decode_img(example['spect1'], IMG_SHAPE)
    spect2 = _decode_img(example['spect2'], IMG_SHAPE)
    label = example['label']
    return {'input1':spect1, 'input2':spect2}, label

def _read_triplet_tfrecord(serialized_example):
    feature_description = {
        'spectA': tf.io.FixedLenFeature((), tf.string),
        'spectP': tf.io.FixedLenFeature((), tf.string),
        'spectN': tf.io.FixedLenFeature((), tf.string),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    
    spectA = _decode_img(example['spectA'], IMG_SHAPE)
    spectP = _decode_img(example['spectP'], IMG_SHAPE)
    spectN = _decode_img(example['spectN'], IMG_SHAPE)

    output = {
        "anchor_input": spectA,
        "positive_input": spectP,
        "negative_input": spectN
    }
    return output
    #return spectA , spectP, spectN

def _read_quadruplet_tfrecord(serialized_example):
    feature_description = {
        'spectA': tf.io.FixedLenFeature((), tf.string),
        'spectP': tf.io.FixedLenFeature((), tf.string),
        'spectN1': tf.io.FixedLenFeature((), tf.string),
        'spectN2': tf.io.FixedLenFeature((), tf.string),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    
    spectA = _decode_img(example['spectA'], IMG_SHAPE)
    spectP = _decode_img(example['spectP'], IMG_SHAPE)
    spectN1 = _decode_img(example['spectN1'], IMG_SHAPE)
    spectN2 = _decode_img(example['spectN2'], IMG_SHAPE)

    output = {
        "anchor_input": spectA,
        "positive_input": spectP,
        "negative_input1": spectN1,
        "negative_input2": spectN2,
    }
    return output

def compute_labelled_distances(embedding_model, anchors, positives, negatives):
    dist_p, dist_n = compute_contrastive_embeddings(embedding_model, anchors, positives, negatives)
    dist = np.array(dist_p + dist_n)
    labels = np.concatenate((np.ones(len(dist_p)), np.zeros(len(dist_n))))
    return dist, labels

def mine_triplets(embedding_model, PARAMS):
    semihard_triplets = []
    output_dir = os.path.join(os.path.dirname(__file__), PARAMS.PATHS.OUTPUT_DIR)
    speaker_spectrograms = utils.load(os.path.join(output_dir, 'speaker_spectrograms.pkl'))
    positive_pair_locs = generate_datasets.find_positive_pairs(speaker_spectrograms)

    i = PARAMS.DATA_GENERATOR.N_SAMPLES

    ## keep going down the list and adding the semi-hard triplets
    while len(semihard_triplets) < PARAMS.DATA_GENERATOR.N_SAMPLES:
        spkr = positive_pair_locs[i][0]
        idx1 = positive_pair_locs[i][1]
        idx2 = positive_pair_locs[i][2]

        negative = generate_datasets.find_random_negative(speaker_spectrograms, spkr)
        cand_triplet = (speaker_spectrograms[spkr][idx1], speaker_spectrograms[spkr][idx2], negative)
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
    speaker_spectrograms = utils.load(os.path.join(output_dir, 'speaker_spectrograms.pkl'))
    positive_pair_locs = generate_datasets.find_positive_pairs(speaker_spectrograms)

    i = PARAMS.DATA_GENERATOR.N_SAMPLES

    ## keep going down the list and adding the semi-hard triplets
    while len(semihard_quadruplets) < PARAMS.DATA_GENERATOR.N_SAMPLES:
        spkr = positive_pair_locs[i][0]
        idx1 = positive_pair_locs[i][1]
        idx2 = positive_pair_locs[i][2]

        negativeA, negativeB = generate_datasets.find_two_random_negatives(speaker_spectrograms, spkr)

        cand_quadruplet = (
            speaker_spectrograms[spkr][idx1], 
            speaker_spectrograms[spkr][idx2], 
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

def train_triplet_model(model, train_dataset, val_dataset, PARAMS, modifier=None):
    opt = set_optimizer(
        OPTIMIZER=PARAMS.TRAINING.OPTIMIZER, 
        LEARNING_RATE=PARAMS.TRAINING.LEARNING_RATE, 
        LEARNING_DECAY_RATE=PARAMS.TRAINING.LEARNING_DECAY_RATE, 
        LEARNING_DECAY_STEPS=PARAMS.TRAINING.LEARNING_DECAY_STEPS, 
        MOMENTUM=PARAMS.TRAINING.MOMENTUM, 
        BETA_1=PARAMS.TRAINING.BETA_1, 
        BETA_2=PARAMS.TRAINING.BETA_2
    )
    ####  compile and fit model  ####
    model.compile(optimizer=opt)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=PARAMS.TRAINING.EPOCHS,
        #batch_size=PARAMS.TRAINING.BATCH_SIZE,
        verbose=1,
        callbacks=[EarlyStopping(patience=PARAMS.TRAINING.EARLY_STOP_ROUNDS)],
    )
    return model

def train_quadruplet_model(model, train_dataset, val_dataset, PARAMS):
    opt = set_optimizer(
        OPTIMIZER=PARAMS.TRAINING.OPTIMIZER, 
        LEARNING_RATE=PARAMS.TRAINING.LEARNING_RATE, 
        LEARNING_DECAY_RATE=PARAMS.TRAINING.LEARNING_DECAY_RATE, 
        LEARNING_DECAY_STEPS=PARAMS.TRAINING.LEARNING_DECAY_STEPS, 
        MOMENTUM=PARAMS.TRAINING.MOMENTUM, 
        BETA_1=PARAMS.TRAINING.BETA_1, 
        BETA_2=PARAMS.TRAINING.BETA_2
    )
    ####  compile and fit model  ####
    model.compile(optimizer=opt)
    history = model.fit(
        #[train_a, train_p, train_n1, train_n2],
        #validation_split=PARAMS.DATA_GENERATOR.VALIDATION_SPLIT,
        train_dataset,
        validation_data=val_dataset,
        epochs=PARAMS.TRAINING.EPOCHS,
        #batch_size=PARAMS.TRAINING.BATCH_SIZE,
        verbose=1,
        callbacks=[EarlyStopping(patience=PARAMS.TRAINING.EARLY_STOP_ROUNDS)],

    )
    return model

def run_cross_entropy_model(IMG_SHAPE, PARAMS):
    output_dir = os.path.join(os.path.dirname(__file__), PARAMS.PATHS.OUTPUT_DIR)
    speaker_spectrograms = utils.load(os.path.join(output_dir, 'speaker_spectrograms_train.pkl'))
    n_classes = len(speaker_spectrograms.keys())
    X = []
    y = []

    ## This changes the label to integers so cross-entropy loss works
    for i, spk in enumerate(speaker_spectrograms.keys()):
        for spect in speaker_spectrograms[spk]:
            X.append(spect)
            y.append(i)

    X = np.array(X)
    y = np.array(y)
    del speaker_spectrograms

    model = tf_models.build_crossentropy_model(n_classes, IMG_SHAPE, PARAMS)

    opt = set_optimizer(
        OPTIMIZER=PARAMS.TRAINING.CROSSENTROPY_OPTIMIZER, 
        LEARNING_RATE=PARAMS.TRAINING.CROSSENTROPY_LEARNING_RATE, 
        LEARNING_DECAY_RATE=PARAMS.TRAINING.CROSSENTROPY_LEARNING_DECAY_RATE, 
        LEARNING_DECAY_STEPS=PARAMS.TRAINING.CROSSENTROPY_LEARNING_DECAY_STEPS, 
        MOMENTUM=PARAMS.TRAINING.CROSSENTROPY_MOMENTUM, 
        BETA_1=PARAMS.TRAINING.CROSSENTROPY_BETA_1, 
        BETA_2=PARAMS.TRAINING.CROSSENTROPY_BETA_2
    )
    model.compile(
        loss='sparse_categorical_crossentropy', 
        optimizer=opt
    )
    logging.info("Training cross-entropy model to pretrain for distnace metric models...")

    history = model.fit(
        X, y,
        validation_split=PARAMS.DATA_GENERATOR.VALIDATION_SPLIT,
        epochs=PARAMS.TRAINING.CROSSENTROPY_EPOCHS,
        verbose=1,
        batch_size=PARAMS.TRAINING.CROSSENTROPY_BATCH_SIZE,
        callbacks=[EarlyStopping(patience=PARAMS.TRAINING.CROSSENTROPY_EARLY_STOP_ROUNDS)],
    )
    return model


def run_siamsese_model(IMG_SHAPE, PARAMS, config_name, embedding_model=None):
    '''
    model = tf_models.build_siamese_model(IMG_SHAPE, PARAMS)
    val_size = PARAMS.DATA_GENERATOR.VALIDATION_SPLIT / (1.0 - PARAMS.DATA_GENERATOR.TEST_SPLIT)
    (pairs, labels) = utils.load(
        os.path.join(os.path.dirname(__file__), PARAMS.PATHS.OUTPUT_DIR, 'contrastive_pairs.pkl')
    )
    pairs_train, pairs_test, labels_train, labels_test = train_test_split(
        pairs, labels, test_size=PARAMS.DATA_GENERATOR.TEST_SPLIT, random_state=123
    )
    pairs_train, pairs_val, labels_train, labels_val = train_test_split(
        pairs_train, labels_train, test_size=val_size, random_state=123
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

    def load_data(input_pair):
        img1 = pickle.load(open(input_pair[0], "rb"))
        img2 = pickle.load(open(input_pair[1], "rb"))

        audio_dir = os.path.join(os.path.dirname(__file__), PARAMS.PATHS.AUDIO_DIR)
        ## check if the speaker id paths match to generate the label
        if input_pair[0].split(audio_dir)[1].split('/')[0] == input_pair[1].split(audio_dir)[1].split('/')[0]:
            label = tf.cast([1], tf.int32)
        else:
            label = tf.cast([0], tf.int32)
        return img1, img2, label

    pdb.set_trace()

    train_dataset = tf.data.Dataset.from_tensor_slices(pairs_train)
    train_dataset = train_dataset.map(lambda x: tf.py_function(func=load_data, inp=[x], Tout=(tf.float32, tf.float32, tf.int32)))

    val_dataset = tf.data.Dataset.from_tensor_slices(pairs_val)
    val_dataset = val_dataset.map(lambda x: tf.py_function(func=load_data, inp=[x], Tout=(tf.float32, tf.float32, tf.int32)))

    pdb.set_trace()
    '''
    model = tf_models.build_siamese_model(IMG_SHAPE, PARAMS)

    output_dir = os.path.join(os.path.dirname(__file__), PARAMS.PATHS.OUTPUT_DIR)
    train_file_path = os.path.join(output_dir, 'contrastive_pairs_train.tfrecord')
    val_file_path = os.path.join(output_dir, 'contrastive_pairs_val.tfrecord')
    test_file_path = os.path.join(output_dir, 'contrastive_pairs_test.pkl')
    model_path = os.path.join(output_dir, 'contrastive_pairs_model{c}'.format(c = config_name))

    pairs_test = utils.load(test_file_path)

    train_dataset = tf.data.TFRecordDataset([train_file_path])
    train_dataset = train_dataset.map(_read_pair_tfrecord)
    train_dataset = train_dataset.batch(PARAMS.TRAINING.BATCH_SIZE, drop_remainder=True).prefetch(1)

    val_dataset = tf.data.TFRecordDataset([val_file_path])
    val_dataset = val_dataset.map(_read_pair_tfrecord)
    val_dataset = val_dataset.batch(PARAMS.TRAINING.BATCH_SIZE, drop_remainder=True).prefetch(1)

    opt = set_optimizer(
        OPTIMIZER=PARAMS.TRAINING.OPTIMIZER, 
        LEARNING_RATE=PARAMS.TRAINING.LEARNING_RATE, 
        LEARNING_DECAY_RATE=PARAMS.TRAINING.LEARNING_DECAY_RATE, 
        LEARNING_DECAY_STEPS=PARAMS.TRAINING.LEARNING_DECAY_STEPS, 
        MOMENTUM=PARAMS.TRAINING.MOMENTUM, 
        BETA_1=PARAMS.TRAINING.BETA_1, 
        BETA_2=PARAMS.TRAINING.BETA_2
    )
    ####  compile and fit model  ####
    model.compile(
        loss=tf_models.contrastive_loss_with_margin(margin=PARAMS.MODEL.MARGIN), 
        optimizer=opt
    )
    logging.info("Training contrastive pair model...")

    history = model.fit(
        train_dataset,
        #validation_split=PARAMS.DATA_GENERATOR.VALIDATION_SPLIT,
        validation_data=val_dataset,
        epochs=PARAMS.TRAINING.EPOCHS,
        #batch_size=PARAMS.TRAINING.BATCH_SIZE,
        verbose=1,
        callbacks=[EarlyStopping(patience=PARAMS.TRAINING.EARLY_STOP_ROUNDS)],
    )

    pairs_test_l = np.array([pair[0] for pair in pairs_test])
    pairs_test_r = np.array([pair[1] for pair in pairs_test])
    labels_test = np.array([pair[2] for pair in pairs_test])

    dist_test = model.predict([pairs_test_l, pairs_test_r])
    model.save(model_path)

    return dist_test, labels_test


def run_triplet_model(IMG_SHAPE, PARAMS, config_name, embedding_model=None):
    model = tf_models.build_triplet_model(IMG_SHAPE, PARAMS, embedding_model=embedding_model)

    output_dir = os.path.join(os.path.dirname(__file__), PARAMS.PATHS.OUTPUT_DIR)
    train_paths = os.path.join(output_dir, 'contrastive_triplets_train.tfrecord')
    val_paths = os.path.join(output_dir, 'contrastive_triplets_val.tfrecord')
    test_paths = os.path.join(output_dir, 'contrastive_triplets_test.pkl')
    model_path = os.path.join(output_dir, 'contrastive_triplets_model_{c}'.format(c = config_name))

    triplets_test = utils.load(test_paths)

    val_dataset = tf.data.TFRecordDataset([val_paths])
    val_dataset = val_dataset.map(_read_triplet_tfrecord)
    val_dataset = val_dataset.batch(PARAMS.TRAINING.BATCH_SIZE, drop_remainder=True).prefetch(1)

    train_dataset = tf.data.TFRecordDataset([train_paths])
    train_dataset = train_dataset.map(_read_triplet_tfrecord)
    train_dataset = train_dataset.batch(PARAMS.TRAINING.BATCH_SIZE, drop_remainder=True).prefetch(1)

    #### Initial training on all triplets
    logging.info("Training tripet loss model on all triplets...")
    model = train_triplet_model(model, train_dataset, val_dataset, PARAMS)
    embedding_layers = model.layers[3].layers
    embedding_model = transfer_embedding_layers(embedding_layers, IMG_SHAPE)

    #### Calculate initial EER
    test_a = np.array([triplet[0] for triplet in triplets_test])
    test_p = np.array([triplet[1] for triplet in triplets_test])
    test_n = np.array([triplet[2] for triplet in triplets_test])

    ####  Transfer learning- take inital embedding layers and score pairs similarly to contrastive loss
    dist_test_initial, labels_test_initial = compute_labelled_distances(embedding_model, test_a, test_p, test_n)
    if PARAMS.TRAINING.MINE_SEMIHARD == 'T':
        initial_EER, eer_threshold, acc = calculate_EER(dist_test_initial, labels_test_initial)
        print("<<<< Initial EER: {EER} >>>> << threshold: {eth} >>, << accuracy: {acc} >>".format(
                EER=initial_EER, eth=eer_threshold, acc = acc
            )
        )

        #### Triplet mining
        logging.info("Mining semi-hard triplets...")
        semihard_triplets = mine_triplets(embedding_model, PARAMS)

        #### Train again with mined triplets
        logging.info("Training tripet loss model on semi-hard triplets...")
        if PARAMS.TRAINING.SEMIHARD_FRESH_MODEL == 'T':
            model = tf_models.build_triplet_model(IMG_SHAPE, PARAMS)
            model = train_triplet_model(model, semihard_triplets, PARAMS, modifier='_semihard')
        else:
            model = train_triplet_model(model, semihard_triplets, PARAMS, modifier='_semihard')

        ####  Transfer learning- take final embedding layers and score pairs similarly to contrastive loss
        embedding_layers = model.layers[3].layers
        embedding_model = transfer_embedding_layers(embedding_layers, IMG_SHAPE)
        dist_test, labels_test = compute_labelled_distances(embedding_model, test_a, test_p, test_n)
    else:
        dist_test = dist_test_initial
        labels_test = labels_test_initial

    embedding_model.save(model_path)
    return dist_test, labels_test



def run_quadruplet_model(IMG_SHAPE, PARAMS, config_name, embedding_model=None):
    model = tf_models.build_quadruplet_model(IMG_SHAPE, PARAMS)
    #quadruplets = utils.load(
    #    os.path.join(os.path.dirname(__file__), PARAMS.PATHS.OUTPUT_DIR, 'contrastive_quadruplets.pkl')
    #)

    output_dir = os.path.join(os.path.dirname(__file__), PARAMS.PATHS.OUTPUT_DIR)
    train_paths = os.path.join(output_dir, 'contrastive_quadruplets_train.tfrecord')
    val_paths = os.path.join(output_dir, 'contrastive_quadruplets_val.tfrecord')
    test_paths = os.path.join(output_dir, 'contrastive_quadruplets_test.pkl')
    model_path = os.path.join(output_dir, 'contrastive_quadruplets_model_{c}'.format(c = config_name))

    quadruplets_test = utils.load(test_paths)

    val_dataset = tf.data.TFRecordDataset([val_paths])
    val_dataset = val_dataset.map(_read_quadruplet_tfrecord)
    val_dataset = val_dataset.batch(PARAMS.TRAINING.BATCH_SIZE, drop_remainder=True).prefetch(1)

    train_dataset = tf.data.TFRecordDataset([train_paths])
    train_dataset = train_dataset.map(_read_quadruplet_tfrecord)
    train_dataset = train_dataset.batch(PARAMS.TRAINING.BATCH_SIZE, drop_remainder=True).prefetch(1)

    #### Initial training on all quadruplets
    logging.info("Training quadruplet loss model on all quadruplets...")
    model = train_quadruplet_model(model, train_dataset, val_dataset, PARAMS)
    embedding_layers = model.layers[4].layers
    embedding_model = transfer_embedding_layers(embedding_layers, IMG_SHAPE)

    ####  Transfer learning- take inital embedding layers and score pairs similarly to contrastive loss
    test_a = np.array([quadruplet[0] for quadruplet in quadruplets_test])
    test_p = np.array([quadruplet[1] for quadruplet in quadruplets_test])
    test_n1 = np.array([quadruplet[2] for quadruplet in quadruplets_test])

    dist_test_initial, labels_test_initial = compute_labelled_distances(embedding_model, test_a, test_p, test_n1)
    if PARAMS.TRAINING.MINE_SEMIHARD == 'T':
        initial_EER, eer_threshold, acc = calculate_EER(dist_test_initial, labels_test_initial)
        print("<<<< Initial EER: {EER} >>>> << threshold: {eth} >>, << accuracy: {acc} >>".format(
                EER=initial_EER, eth=eer_threshold, acc = acc
            )
        )
        #### Quadruplet mining
        logging.info("Mining semi-hard quadruplets...")
        semihard_quadruplets = mine_quadruplets(embedding_model, PARAMS)

        #### Train again with mined quadruplets
        logging.info("Training quadruplet loss model on semi-hard quadruplets...")
        if PARAMS.TRAINING.SEMIHARD_FRESH_MODEL == 'T':
            model = tf_models.build_triplet_model(IMG_SHAPE, PARAMS)
            model, _ = train_quadruplet_model(model, semihard_quadruplets, PARAMS)
        else:
            model, _ = train_quadruplet_model(model, semihard_quadruplets, PARAMS)

        ####  Transfer learning- take embedding layers and score pairs similarly to contrastive loss
        embedding_layers = model.layers[4].layers
        embedding_model = transfer_embedding_layers(embedding_layers, IMG_SHAPE)
        dist_test, labels_test = compute_labelled_distances(embedding_model, test_a, test_p, test_n1)
    else:
        dist_test = dist_test_initial
        labels_test = labels_test_initial

    embedding_model.save(model_path)
    return dist_test, labels_test




def pretrain_model(IMG_SHAPE, PARAMS):
    model = run_cross_entropy_model(IMG_SHAPE, PARAMS)
    test_triplet_path = os.path.join(os.path.dirname(__file__), PARAMS.PATHS.OUTPUT_DIR, 'contrastive_triplets_test.pkl')
    triplets_test = utils.load(test_triplet_path)

    test_a = np.array([triplet[0] for triplet in triplets_test])
    test_p = np.array([triplet[1] for triplet in triplets_test])
    test_n = np.array([triplet[2] for triplet in triplets_test])

    if PARAMS.TRAINING.FREEZE_PRETRAIN_BASE == 'T':
        model.trainable = False

    pretrained_embedding_layers = model.layers[:-1]
    pretrained_embedding_model = transfer_embedding_layers(pretrained_embedding_layers, IMG_SHAPE)

    dist_test_crossentropy, labels_test_crossentropy = compute_labelled_distances(
        pretrained_embedding_model, test_a, test_p, test_n
    )
    crossentropy_EER, eer_threshold, acc = calculate_EER(dist_test_crossentropy, labels_test_crossentropy)
    print("<<<< Cross-entropy pre-train EER: {EER} >>>> << threshold: {eth} >> << accuracy: {acc} >>".format(
            EER=crossentropy_EER, eth=eer_threshold, acc= acc
        )
    )

    cnn_embedding_layers = model.layers[:-2]
    pretrained_cnn_model_seq = Sequential([Input(IMG_SHAPE)] + cnn_embedding_layers)

    return pretrained_cnn_model_seq




if __name__ == '__main__':
    PARAMS = utils.config_init(sys.argv)
    config_name = os.path.splitext(str(sys.argv[1]))[0]
    print('#'*80)
    print('<<<< Config parameters:{p}'.format(p=PARAMS))
    IMG_SHAPE = (
        PARAMS.DATA_GENERATOR.N_MELS,
        PARAMS.DATA_GENERATOR.MAX_FRAMES,
        1
    )
    
    if PARAMS.MODEL.CROSSENTROPY_PRETRAIN == 'T':
        pretrain_model_path = os.path.join(os.path.dirname(__file__), PARAMS.PATHS.OUTPUT_DIR, 'pretrain_embedding_model')
        if PARAMS.TRAINING.USE_PREVIOUS_PRETRAIN_MODEL == 'T':
            logging.info('Loading previous pretraining model...')
            pretrained_embedding_model_seq = tf.keras.models.load_model(pretrain_model_path)
        else:
            logging.info('Pretraining cross-entropy loss model...')
            pretrained_embedding_model_seq = pretrain_model(IMG_SHAPE, PARAMS)
            pretrained_embedding_model_seq.save(pretrain_model_path)
    else:
        pretrained_embedding_model_seq = None

    ####  build model  ####
    if PARAMS.MODEL.LOSS_TYPE == 'contrastive':
        dist_test, labels_test = run_siamsese_model(IMG_SHAPE, PARAMS, config_name, pretrained_embedding_model_seq)
    if PARAMS.MODEL.LOSS_TYPE == 'triplet':
        dist_test, labels_test = run_triplet_model(IMG_SHAPE, PARAMS, config_name, pretrained_embedding_model_seq)
    if PARAMS.MODEL.LOSS_TYPE == 'quadruplet':
        dist_test, labels_test = run_quadruplet_model(IMG_SHAPE, PARAMS, config_name, pretrained_embedding_model_seq)
    
    ####  Find EER   ####
    EER, eer_threshold, acc = calculate_EER(dist_test, labels_test)

    print('-'*60)
    print('<<<<<  The EER is:  {EER} !  >>>>> << threshold: {eth} >> << accuracy: {acc} >>'.format(
            EER=EER, eth=eer_threshold, acc=acc
        )
    )
    print('#'*80)
