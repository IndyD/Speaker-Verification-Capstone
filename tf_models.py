import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, Lambda, Layer

import pdb


def euclidean_distance(vectors):
    ''' Distance measure used for siamese model '''
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1,
        keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    return contrastive_loss


class TripletLossLayer(Layer):
    def __init__(self, margin, **kwargs):
        self.margin = margin
        super(TripletLossLayer, self).__init__(**kwargs)
    
    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor-positive), axis=-1)
        n_dist = K.sum(K.square(anchor-negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.margin, 0), axis=0)
    
    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

class QuadrupletLossLayer(Layer):
    def __init__(self, margin, **kwargs):
        self.margin = margin
        super(QuadrupletLossLayer, self).__init__(**kwargs)
    
    def quadruplet_loss(self, inputs):
        anchor, positive, negative1, negative2 = inputs
        p_dist = K.sum(K.square(anchor-positive), axis=-1)
        n1_dist = K.sum(K.square(anchor-negative1), axis=-1)
        n2_dist = K.sum(K.square(anchor-negative2), axis=-1)
        return K.sum(K.maximum(p_dist - n1_dist + self.margin, 0), axis=0) + K.sum(K.maximum(p_dist - n2_dist + self.margin, 0), axis=0)
    
    def call(self, inputs):
        loss = self.quadruplet_loss(inputs)
        self.add_loss(loss)
        return loss


def build_vgg7_embedding_model(IMG_SHAPE):
    ''' 
    Return and embedding model using the fist 7 layer of VGG16 w/ 2 dense layers
    This architechture is from the VGG7 implementaion of Velez
    '''
    ## Note that VGG16 was trained on 3 channels. We can't use the weights w/ 1 channel so set to None
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

def build_siamese_model(IMG_SHAPE):
    ''' Build a siamese vgg7 model that computes the distance between two images '''
    embedding_model = build_vgg7_embedding_model(IMG_SHAPE)
    
    imgA = Input(shape=IMG_SHAPE)
    imgB = Input(shape=IMG_SHAPE)

    featsA = embedding_model(imgA)
    featsB = embedding_model(imgB)
    
    distance = Lambda(euclidean_distance)([featsA, featsB])
    siamese_vgg7_model = Model(inputs=[imgA, imgB], outputs=distance)
    return siamese_vgg7_model


def build_triplet_model(IMG_SHAPE, PARAMS):
    ''' Build a triplet vgg7 model that computes the distance between
    an anchor image, a positive image, and a negative image '''
    triplet_encoding_model = build_vgg7_embedding_model(IMG_SHAPE)
    triplet_encoding_model.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))
    
    anchor_input = Input(IMG_SHAPE, name="anchor_input")
    positive_input = Input(IMG_SHAPE, name="positive_input")
    negative_input = Input(IMG_SHAPE, name="negative_input") 
    
    # Generate the encodings (feature vectors) for the three images
    encoded_a = triplet_encoding_model(anchor_input)
    encoded_p = triplet_encoding_model(positive_input)
    encoded_n = triplet_encoding_model(negative_input)
    
    #TripletLoss Layer
    loss_layer = TripletLossLayer(margin=PARAMS.TRAINING.MARGIN,name='triplet_loss_layer')([encoded_a, encoded_p, encoded_n])
    
    # Connect the inputs with the outputs
    triplet_model = Model(inputs=[anchor_input,positive_input,negative_input],outputs=loss_layer)
    return triplet_model

def build_quadruplet_model(IMG_SHAPE, PARAMS):
    ''' Build a quadruplet vgg7 model that computes the distance between
    an anchor image, a positive image, and two dissimilar negative images '''
    quadruplet_encoding_model = build_vgg7_embedding_model(IMG_SHAPE)
    quadruplet_encoding_model.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))
    
    anchor_input = Input(IMG_SHAPE, name="anchor_input")
    positive_input = Input(IMG_SHAPE, name="positive_input")
    negative_input1 = Input(IMG_SHAPE, name="negative_input1") 
    negative_input2 = Input(IMG_SHAPE, name="negative_input2") 

    
    # Generate the encodings (feature vectors) for the three images
    encoded_a = quadruplet_encoding_model(anchor_input)
    encoded_p = quadruplet_encoding_model(positive_input)
    encoded_n1 = quadruplet_encoding_model(negative_input1)
    encoded_n2 = quadruplet_encoding_model(negative_input2)
    
    #QuadrupletLoss Layer
    loss_layer = QuadrupletLossLayer(margin=PARAMS.TRAINING.MARGIN,name='quadruplet_loss_layer')([encoded_a, encoded_p, encoded_n1, encoded_n2])
    
    # Connect the inputs with the outputs
    quadruplet_model = Model(inputs=[anchor_input, positive_input, negative_input1, negative_input2],outputs=loss_layer)
    return quadruplet_model