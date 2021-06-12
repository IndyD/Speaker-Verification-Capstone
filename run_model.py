import pickle
import pprint
import sys
import os
import pdb
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input

import tf_models
import utils
import logging

logging.basicConfig(level=logging.INFO)


def run_siamsese_model(IMG_SHAPE, PARAMS):
    model = tf_models.build_siamese_model(IMG_SHAPE)
    (pairs, labels) = utils.load(
        os.path.join(PARAMS.PATHS.BASE_DIR, PARAMS.PATHS.OUTPUT_DIR, 'contrastive_pairs.pkl')
    )

    pairs_train, pairs_test, labels_train, labels_test = train_test_split(
        pairs, labels, test_size=PARAMS.DATA_GENERATOR.TEST_SPLIT, random_state=123
    )

    ####  split and normalize the spectograms  ####
    pairs_train_l = np.array([pair[0] / -80.0 for pair in pairs_train])
    pairs_train_r = np.array([pair[1] / -80.0 for pair in pairs_train])
    pairs_test_l = np.array([pair[0] / -80.0 for pair in pairs_test])
    pairs_test_r = np.array([pair[1] / -80.0 for pair in pairs_test])
    labels_train = tf.cast(np.array(labels_train), tf.float32)
    labels_test = tf.cast(np.array(labels_test), tf.float32)

    ####  compile and fit model  ####
    model.compile(loss=tf_models.contrastive_loss_with_margin(margin=PARAMS.TRAINING.MARGIN), optimizer="adam")
    logging.info("Training contrastive pair model...")

    history = model.fit(
        [pairs_train_l, pairs_train_r], labels_train,
        validation_data=([pairs_test_l, pairs_test_r], labels_test),
        epochs=PARAMS.TRAINING.EPOCHS,
        verbose=1,
    )

    dist_test = model.predict([pairs_test_l, pairs_test_r])

    return dist_test, labels_test



def run_triplet_model(IMG_SHAPE, PARAMS):
    model = tf_models.build_triplet_model(IMG_SHAPE, PARAMS)
    triplets = utils.load(
        os.path.join(PARAMS.PATHS.BASE_DIR, PARAMS.PATHS.OUTPUT_DIR, 'contrastive_triplets.pkl')
    )

    triplets_train = triplets[int(len(triplets) * PARAMS.DATA_GENERATOR.TEST_SPLIT):]
    triplets_test = triplets[:int(len(triplets) * PARAMS.DATA_GENERATOR.TEST_SPLIT)]

    ####  split and normalize the spectograms  ####
    train_a = np.array([triplet[0] / -80.0 for triplet in triplets_train])
    train_p = np.array([triplet[1] / -80.0 for triplet in triplets_train])
    train_n = np.array([triplet[2] / -80.0 for triplet in triplets_train])

    test_a = np.array([triplet[0] / -80.0 for triplet in triplets_test])
    test_p = np.array([triplet[1] / -80.0 for triplet in triplets_test])
    test_n = np.array([triplet[2] / -80.0 for triplet in triplets_test])

    ####  compile and fit model  ####
    model.compile(optimizer="adam")
    logging.info("Training tripet loss model...")

    history = model.fit(
        [train_a, train_p, train_n],
        validation_data=([test_a, test_p, test_n]),
        epochs=PARAMS.TRAINING.EPOCHS,
        verbose=1,
    )

    pdb.set_trace()

    ####  Transfer learning- take embedding layers and score pairs similarly to contrastive loss

    embedding_layers = model.layers[3]
    img_input = Input(IMG_SHAPE)
    emb_model = Sequential([Input(IMG_SHAPE)] + embedding_layers.layers)
    trained_embedding_model = Model(inputs=img_input, outputs=emb_model(img_input))
    
    embeddings_a = trained_embedding_model.predict(test_a)
    embeddings_p = trained_embedding_model.predict(test_p)
    embeddings_n = trained_embedding_model.predict(test_n)

    pos_pairs = zip(embeddings_a, embeddings_p)
    dist_p = [np.linalg.norm(emb[0] - emb[1]) for emb in pos_pairs]

    neg_pairs = zip(embeddings_a, embeddings_n)
    dist_n = [np.linalg.norm(emb[0] - emb[1]) for emb in neg_pairs]

    dist_test = np.array(dist_p + dist_n)
    labels_test = np.concatenate((np.ones(len(dist_p)), np.zeros(len(dist_n))))

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
        dist_test = None
        labels_test = None

    ## scale the distances to compute EERs
    preds = dist_test / dist_test.max()

    ####  Find EER   ####
    fpr, tpr, threshold = roc_curve(labels_test, preds, pos_label=0)
    fnr = 1 - tpr
    #eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]


    print('#'*80)
    print('Config parameters:')
    pprint.pprint(PARAMS)
    print('-'*60)
    print('<<<<<  The EER is:  {EER} !  >>>>>'.format(EER=EER))
    print('#'*80)
