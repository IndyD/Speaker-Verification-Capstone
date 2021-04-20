import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Lambda

import pdb

def euclidean_distance(vectors):
	# unpack the vectors into separate lists
	(featsA, featsB) = vectors
	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		keepdims=True)
	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def contrastive_loss(y, preds, margin=1):
	# explicitly cast the true class label data type to the predicted
	# class label data type (otherwise we run the risk of having two
	# separate data types, causing TensorFlow to error out)
	y = tf.cast(y, preds.dtype)
	# calculate the contrastive loss between the true labels and
	# the predicted labels
	squaredPreds = K.square(preds)
	squaredMargin = K.square(K.maximum(margin - preds, 0))
	loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
	# return the computed contrastive loss to the calling function
	return loss

def build_vgg7_embedding_model(IMG_SHAPE):
    ## This architechture is from the VGG7 implementaion of Velez
    VGG16 = tf.keras.applications.VGG16(input_shape=IMG_SHAPE, include_top=False, weights=None)
    VGG7 = VGG16.layers[0:7]
    embedding_layers = [
        Flatten(name='flatten'),
        Dense(1024),
        Activation('relu'),
        Dense(1024),
        Activation('relu'),
    ]
    embedding_model = Sequential(VGG7 + embedding_layers)
    return embedding_model

def build_siamese_vgg7_model(IMG_SHAPE):
    embedding_model = build_vgg7_embedding_model(IMG_SHAPE)
    imgA = Input(shape=IMG_SHAPE)
    imgB = Input(shape=IMG_SHAPE)
    featsA = embedding_model(imgA)
    featsB = embedding_model(imgB)
    
    distance = Lambda(euclidean_distance)([featsA, featsB])
    siamese_vgg7_model = Model(inputs=[imgA, imgB], outputs=distance)
    return siamese_vgg7_model