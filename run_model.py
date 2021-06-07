import pickle
import sys
import pdb
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

import tf_models
import utils



def run_siamsese_model(IMG_SHAPE, PARAMS):
    model = tf_models.build_siamese_model(IMG_SHAPE)
    (pairs, labels) = utils.load(PARAMS.PATHS.PAIRS_PATH)

    pairs_train, pairs_test, label_train, label_test = train_test_split(
        pairs, labels, test_size=PARAMS.DATA_GENERATOR.TEST_SPLIT, random_state=123
    )

    ####  split and normalize the spectograms  ####
    pairs_train_l = np.array([pair[0] / -80.0 for pair in pairs_train])
    pairs_train_r = np.array([pair[1] / -80.0 for pair in pairs_train])
    pairs_test_l = np.array([pair[0] / -80.0 for pair in pairs_test])
    pairs_test_r = np.array([pair[1] / -80.0 for pair in pairs_test])
    label_train = tf.cast(np.array(label_train), tf.float32)
    label_test = tf.cast(np.array(label_test), tf.float32)

    ####  compile and fit model  ####
    model.compile(loss=tf_models.contrastive_loss_with_margin(margin=1), optimizer="adam")
    print("Training model...")

    history = model.fit(
        [pairs_train_l, pairs_train_r], label_train,
        validation_data=([pairs_test_l, pairs_test_r], label_test),
        epochs=PARAMS.TRAINING.EPOCHS,
        verbose=1,
        )

    dist = model.predict([pairs_test_l, pairs_test_r])
    preds = np.clip(dist, 0, 1)

    ####  Find EER   ####
    fpr, tpr, threshold = roc_curve(label_test, preds, pos_label=0)
    fnr = 1 - tpr
    #eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    return EER


if __name__ == '__main__':
    PARAMS = utils.config_init(sys.argv)
    IMG_SHAPE = (
        PARAMS.DATA_GENERATOR.N_MELS,
        PARAMS.DATA_GENERATOR.MAX_FRAMES,
        1
    )

    ####  build model  ####
    if PARAMS.LOSS_TYPE == 'contrastive':
        EER = run_siamsese_model(IMG_SHAPE, PARAMS)
    if PARAMS.LOSS_TYPE == 'triplet':
        EER = None
    if PARAMS.LOSS_TYPE == 'quadruplet':
        EER = None

    print('#'*80)
    print('<<<<<  The EER is:  {EER} !  >>>>>'.format(EER=EER))
    print('#'*80)
