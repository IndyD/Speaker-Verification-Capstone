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

'''
class ContrastiveLossLayer(Layer):
    def __init__(self, margin, **kwargs):
        self.margin = margin
        super(ContrastiveLossLayer, self).__init__(**kwargs)
    
    def contrastive_loss(self, inputs):
        img_l, img_r, y = inputs
        img_dist = K.sum(K.square(img_l-img_r), axis=-1)
        return K.sum(K.maximum(y * img_dist + (self.margin - img_dist) * (1-y), 0), axis=0)
    
    def call(self, inputs):
        loss = self.contrastive_loss(inputs)
        self.add_loss(loss)
        return loss
'''


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

def get_embedding_layers(PARAMS):
    if PARAMS.MODEL.N_DENSE == 1:
        embedding_layers = [
            Flatten(name='flatten'),
            Dense(
                PARAMS.MODEL.DENSE1_NODES, 
                kernel_regularizer=tf.keras.regularizers.l2(PARAMS.MODEL.L2_WEIGHT_DECAY),
                bias_regularizer=tf.keras.regularizers.l2(PARAMS.MODEL.L2_WEIGHT_DECAY)
            ),
            Activation(PARAMS.MODEL.DENSE_ACTIVATION),
        ]
    elif PARAMS.MODEL.N_DENSE == 2:
        embedding_layers = [
            Flatten(name='flatten'),
            Dense(
                PARAMS.MODEL.DENSE1_NODES, 
                kernel_regularizer=tf.keras.regularizers.l2(PARAMS.MODEL.L2_WEIGHT_DECAY),
                bias_regularizer=tf.keras.regularizers.l2(PARAMS.MODEL.L2_WEIGHT_DECAY)
            ),
            Activation(PARAMS.MODEL.DENSE_ACTIVATION),
            Dense(
                PARAMS.MODEL.DENSE2_NODES, 
                kernel_regularizer=tf.keras.regularizers.l2(PARAMS.MODEL.L2_WEIGHT_DECAY),
                bias_regularizer=tf.keras.regularizers.l2(PARAMS.MODEL.L2_WEIGHT_DECAY)
            ),
            Activation(PARAMS.MODEL.DENSE_ACTIVATION),
        ]
    else:
        raise ValueError('ERROR: Invalid PARAMS.MODEL.N_DENSE value!')

    return embedding_layers

def get_vgg7_layers(IMG_SHAPE):
    VGG16 = tf.keras.applications.VGG16(input_shape=IMG_SHAPE, include_top=False, weights=None)
    VGG7 = VGG16.layers[0:7]
    return VGG7

def build_vgg7_embedding_model(IMG_SHAPE, PARAMS):
    ''' 
    Return and embedding model using the fist 7 layer of VGG16 w/ 2 dense layers
    This architechture is from the VGG7 implementaion of Velez
    '''
    ## Note that VGG16 was trained on 3 channels. We can't use the weights w/ 1 channel so set to None
    VGG7 = get_vgg7_layers(IMG_SHAPE)
    embedding_layers = get_embedding_layers(PARAMS)
    embedding_model = Sequential(VGG7 + embedding_layers)
    return embedding_model

def set_embedding_model(embedding_model, IMG_SHAPE, PARAMS):
    if not embedding_model:
        embedding_model = build_vgg7_embedding_model(IMG_SHAPE, PARAMS)
    else:
        if PARAMS.MODEL.N_DENSE == 1:
            embedding_model.add(
                Dense(
                    PARAMS.MODEL.DENSE1_NODES, 
                    kernel_regularizer=tf.keras.regularizers.l2(PARAMS.MODEL.L2_WEIGHT_DECAY),
                    bias_regularizer=tf.keras.regularizers.l2(PARAMS.MODEL.L2_WEIGHT_DECAY)
                )
            )
            embedding_model.add(Activation(PARAMS.MODEL.DENSE_ACTIVATION))
        elif PARAMS.MODEL.N_DENSE == 2:
            embedding_model.add(
                Dense(
                    PARAMS.MODEL.DENSE1_NODES, 
                    kernel_regularizer=tf.keras.regularizers.l2(PARAMS.MODEL.L2_WEIGHT_DECAY),
                    bias_regularizer=tf.keras.regularizers.l2(PARAMS.MODEL.L2_WEIGHT_DECAY)
                )
            )
            embedding_model.add(Activation(PARAMS.MODEL.DENSE_ACTIVATION))
            embedding_model.add(
                Dense(
                    PARAMS.MODEL.DENSE2_NODES, 
                    kernel_regularizer=tf.keras.regularizers.l2(PARAMS.MODEL.L2_WEIGHT_DECAY),
                    bias_regularizer=tf.keras.regularizers.l2(PARAMS.MODEL.L2_WEIGHT_DECAY)
                )
            )
            embedding_model.add(Activation(PARAMS.MODEL.DENSE_ACTIVATION))

    return embedding_model

def build_siamese_model(IMG_SHAPE, PARAMS, embedding_model=None):
    ''' Build a siamese vgg7 model that computes the distance between two images '''
    embedding_model = set_embedding_model(embedding_model, IMG_SHAPE, PARAMS)
    
    imgA = Input(shape=IMG_SHAPE, name="input1")
    imgB = Input(shape=IMG_SHAPE, name="input2")

    featsA = embedding_model(imgA)
    featsB = embedding_model(imgB)
    
    distance = Lambda(euclidean_distance)([featsA, featsB])
    siamese_vgg7_model = Model(inputs=[imgA, imgB], outputs=distance)
    return siamese_vgg7_model


"""
def build_siamese_model_new(IMG_SHAPE, PARAMS, embedding_model=None):
    ''' Build a triplet vgg7 model that computes the distance between
    an anchor image, a positive image, and a negative image '''
    if not embedding_model:
        embedding_model = build_vgg7_embedding_model(IMG_SHAPE, PARAMS)
    embedding_model.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))
    
    input_l = Input(IMG_SHAPE, name="input_l")
    input_r = Input(IMG_SHAPE, name="input_r")
    
    # Generate the encodings (feature vectors) for the three images
    encoded_l = embedding_model(input_l)
    encoded_r = embedding_model(input_r)
    
    #TripletLoss Layer
    loss_layer = ContrastiveLossLayer(margin=PARAMS.MODEL.MARGIN,name='contrastive_loss_layer')([encoded_l, encoded_r, encoded_n])
    
    # Connect the inputs with the outputs
    triplet_model = Model(inputs=[anchor_input,positive_input,negative_input],outputs=loss_layer)
    return triplet_model
"""

def build_triplet_model(IMG_SHAPE, PARAMS, embedding_model=None):
    ''' Build a triplet vgg7 model that computes the distance between
    an anchor image, a positive image, and a negative image '''
    embedding_model = set_embedding_model(embedding_model, IMG_SHAPE, PARAMS)
    embedding_model.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))
    
    anchor_input = Input(IMG_SHAPE, name="anchor_input")
    positive_input = Input(IMG_SHAPE, name="positive_input")
    negative_input = Input(IMG_SHAPE, name="negative_input") 
    
    # Generate the encodings (feature vectors) for the three images
    encoded_a = embedding_model(anchor_input)
    encoded_p = embedding_model(positive_input)
    encoded_n = embedding_model(negative_input)
    
    #TripletLoss Layer
    loss_layer = TripletLossLayer(margin=PARAMS.MODEL.MARGIN,name='triplet_loss_layer')([encoded_a, encoded_p, encoded_n])
    
    # Connect the inputs with the outputs
    triplet_model = Model(inputs=[anchor_input,positive_input,negative_input],outputs=loss_layer)
    return triplet_model

def build_quadruplet_model(IMG_SHAPE, PARAMS, embedding_model=None):
    ''' Build a quadruplet vgg7 model that computes the distance between
    an anchor image, a positive image, and two dissimilar negative images '''
    embedding_model = set_embedding_model(embedding_model, IMG_SHAPE, PARAMS)
    embedding_model.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))
    
    anchor_input = Input(IMG_SHAPE, name="anchor_input")
    positive_input = Input(IMG_SHAPE, name="positive_input")
    negative_input1 = Input(IMG_SHAPE, name="negative_input1") 
    negative_input2 = Input(IMG_SHAPE, name="negative_input2") 

    
    # Generate the encodings (feature vectors) for the three images
    encoded_a = embedding_model(anchor_input)
    encoded_p = embedding_model(positive_input)
    encoded_n1 = embedding_model(negative_input1)
    encoded_n2 = embedding_model(negative_input2)
    
    #QuadrupletLoss Layer
    loss_layer = QuadrupletLossLayer(margin=PARAMS.MODEL.MARGIN,name='quadruplet_loss_layer')([encoded_a, encoded_p, encoded_n1, encoded_n2])
    
    # Connect the inputs with the outputs
    quadruplet_model = Model(inputs=[anchor_input, positive_input, negative_input1, negative_input2],outputs=loss_layer)
    return quadruplet_model

def build_crossentropy_model(n_classes, IMG_SHAPE, PARAMS):
    ''' Build a cross-entropy model for baseline performance and to pre-train
    before fine-tuning with distance metrics '''

    VGG7 = get_vgg7_layers(IMG_SHAPE)
    crossentropy_layers = [
        Flatten(name='flatten'),
        Dense(
            PARAMS.MODEL.CROSSENTROPY_DENSE_NODES, 
            activation=PARAMS.MODEL.CROSSENTROPY_ACTIVATION
        ),
        Dense(
            n_classes,
            activation='softmax'
        ),
    ]
    crossentropy_model = Sequential(VGG7 + crossentropy_layers)
    return crossentropy_model
    
