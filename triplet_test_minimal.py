import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Dropout, Activation, Flatten, Lambda, Layer
import sys 
import pickle
import tf_models
import utils
import pdb

tf.config.run_functions_eagerly(True)


IMG_SHAPE = (130, 300, 1)
with open('output/speaker_spectrograms.pkl', 'rb') as fin:
    speaker_spectrograms = pickle.load(fin)
with open('output/contrastive_triplets.pkl', 'rb') as fin:
    triplet_locs = pickle.load(fin)

triplets = []
for triplet_data in triplet_locs:
    triplets.append( (
        speaker_spectrograms[triplet_data[0][0]][triplet_data[0][1]],
        speaker_spectrograms[triplet_data[1][0]][triplet_data[1][1]],
        speaker_spectrograms[triplet_data[2][0]][triplet_data[2][1]]
    ) )

#############################
######     Model       ######
#############################


class TripletLossLayer(Layer):
    def __init__(self, margin, **kwargs):
        self.margin = margin
        super(TripletLossLayer, self).__init__(**kwargs)
    
    def triplet_loss(self, inputs):
        (anchor, positive, negative) = inputs
        p_dist = K.sum(K.square(anchor-positive), axis=-1)
        n_dist = K.sum(K.square(anchor-negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.margin, 0), axis=0)
    
    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss



def build_triplet_model(IMG_SHAPE):
    VGG16 = tf.keras.applications.VGG16(input_shape=IMG_SHAPE, include_top=False, weights=None)
    embedding_layers = [Flatten(name='flatten'), Dense(1024), Activation('sigmoid')]
    embedding_model = Sequential(VGG16.layers[0:7] + embedding_layers)
    embedding_model.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))
    
    anchor_input = Input(IMG_SHAPE, name="anchor_input")
    positive_input = Input(IMG_SHAPE, name="positive_input")
    negative_input = Input(IMG_SHAPE, name="negative_input") 
    
    # Generate the encodings (feature vectors) for the three images
    encoded_a = embedding_model(anchor_input)
    encoded_p = embedding_model(positive_input)
    encoded_n = embedding_model(negative_input)
    loss_layer = TripletLossLayer(margin=1,name='triplet_loss_layer')((encoded_a, encoded_p, encoded_n))
    
    # Connect the inputs with the outputs
    triplet_model = Model(inputs=(anchor_input,positive_input,negative_input),outputs=loss_layer)
    return triplet_model

train_a = np.array([triplet[0] for triplet in triplets])
train_p = np.array([triplet[1] for triplet in triplets])
train_n = np.array([triplet[2] for triplet in triplets])

model2 = build_triplet_model(IMG_SHAPE)
model2.compile(optimizer='adam') #, run_eagerly=True)
####  compile and fit model  ####
history = model2.fit(
        [train_a, train_p, train_n],
        #batch_size=30,
        validation_split=.1
)
