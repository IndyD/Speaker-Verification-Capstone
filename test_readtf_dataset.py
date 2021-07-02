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
from tensorflow.keras.optimizers import Adam, SGD, Adamax, Nadam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

import generate_datasets
import tf_models
import utils

import pdb

logging.basicConfig(level=logging.DEBUG)

IMG_SHAPE = (130, 300, 1)

file_paths = ['output/contrastive_pairs.tfrecord']
tfrecord_dataset = tf.data.TFRecordDataset(file_paths)

def _decode_img(img_bytes, IMG_SHAPE):
    img = tf.io.decode_raw(img_bytes, tf.float32)
    img.set_shape([IMG_SHAPE[0] * IMG_SHAPE[1] * IMG_SHAPE[2]])
    img = tf.reshape(img, IMG_SHAPE)
    return img

def _read_contrastive_tfrecord(serialized_example):
    feature_description = {
        'spect1': tf.io.FixedLenFeature((), tf.string),
        'spect2': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.string),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    
    spect1 = _decode_img(example['spect1'], IMG_SHAPE)
    spect2 = _decode_img(example['spect2'], IMG_SHAPE)
    label = tf.io.decode_raw(example['label'], tf.int64)

    return spect1 , spect2, label

parsed_dataset = tfrecord_dataset.map(_read_contrastive_tfrecord)
for data in parsed_dataset.take(2):
    pdb.set_trace()
    print(data[0])
