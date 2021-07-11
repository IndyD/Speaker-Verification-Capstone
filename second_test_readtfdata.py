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
file_path = 'test_contrastive_triplets.tfrecord'
file_pathA = 'test_contrastive_triplets_anchor.tfrecord'
file_pathP = 'test_contrastive_triplets_positive.tfrecord'
file_pathN = 'test_contrastive_triplets_negative.tfrecord'

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
        #pdb.set_trace()
        (anchor, positive, negative) = inputs
        p_dist = K.sum(K.square(anchor-positive), axis=-1)
        n_dist = K.sum(K.square(anchor-negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.margin, 0), axis=0)
    
    def call(self, inputs):
        #pdb.set_trace()
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

''' 
class TripletLossLayer(Layer):
    def __init__(self, margin, **kwargs):
        self.margin = margin
        super(TripletLossLayer, self).__init__(**kwargs)
    
    def triplet_loss(self, anchor, positive, negative):
        p_dist = K.sum(K.square(anchor-positive), axis=-1)
        n_dist = K.sum(K.square(anchor-negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.margin, 0), axis=0)
    
    def call(self, anchor, positive, negative):
        pdb.set_trace()
        loss = self.triplet_loss(anchor, positive, negative)
        self.add_loss(loss)
        return loss

'''


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
'''

def get_base_model(IMG_SHAPE):
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    conv2d_1 = Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu")(inputs)
    conv2d_2 = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu")(conv2d_1)
    maxpool_1 = MaxPool2D(pool_size=(2,2),strides=(2,2))(conv2d_2)
    conv2d_3 = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(maxpool_1)
    conv2d_4 = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(conv2d_3)
    maxpool_2 = MaxPool2D(pool_size=(2,2),strides=(2,2))(conv2d_4)
    flatten = Flatten(name='flatten')(maxpool_2)
    outputs = Dense(1024)(flatten)

    model = Model(inputs=inputs, outputs=outputs)
    return model



def build_triplet_model(IMG_SHAPE):
    anchor_input = Input(shape=IMG_SHAPE, name="anchor_input")
    positive_input = Input(shape=IMG_SHAPE, name="positive_input")
    negative_input = Input(shape=IMG_SHAPE, name="negative_input") 

    base_model = get_base_model(IMG_SHAPE)

    encoded_a  = base_model(anchor_input)
    encoded_p = base_model(positive_input)
    encoded_n = base_model(negative_input)

    # build distance measuring layer
    l1_lambda = Lambda(lambda x: K.l2_normalize(x,axis=-1))
    l1_dist   = l1_lambda([encoded_a, encoded_p, encoded_n])

    pred = Dense(1,activation='sigmoid')(l1_dist)
    return Model(inputs=[anchor_input, positive_input, negative_input], outputs=pred)
'''



'''
def triplet_generator():
    for triplet_data in triplet_locs:
        #yield speaker_spectrograms[triplet_data[0][0]][triplet_data[0][1]], speaker_spectrograms[triplet_data[1][0]][triplet_data[1][1]], speaker_spectrograms[triplet_data[2][0]][triplet_data[2][1]]
        anch = speaker_spectrograms[triplet_data[0][0]][triplet_data[0][1]]
        pos = speaker_spectrograms[triplet_data[1][0]][triplet_data[1][1]]
        neg = speaker_spectrograms[triplet_data[2][0]][triplet_data[2][1]]
        
        output = {
            "anchor_input": anch,
            "positive_input": pos,
            "negative_input": neg
        }

        yield output

#pdb.set_trace()

types = (np.float32,np.float32,np.float32)
types_dict = {
            "anchor_input": np.float32,
            "positive_input": np.float32,
            "negative_input": np.float32
        }

shapes = ([None, 130,300,1],[None, 130,300,1],[None, 130,300,1])
parsed_dataset = tf.data.Dataset.from_generator(triplet_generator,
                                      output_types=types_dict,
                                      #output_shapes=shapes
                                     )
                                     

'''




####################################################
###############   Write Dataset  ################
####################################################

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


with tf.io.TFRecordWriter(file_pathA) as writerA:
    with tf.io.TFRecordWriter(file_pathP) as writerP:
        with tf.io.TFRecordWriter(file_pathN) as writerN:
            for i, triplet_data in enumerate(triplet_locs):
                spectA_b = speaker_spectrograms[triplet_data[0][0]][triplet_data[0][1]].tobytes()
                spectP_b = speaker_spectrograms[triplet_data[1][0]][triplet_data[1][1]].tobytes()
                spectN_b = speaker_spectrograms[triplet_data[2][0]][triplet_data[2][1]].tobytes()

                exampleA = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'spectA': _bytes_feature(spectA_b),
                        }
                    )
                )
                writerA.write(exampleA.SerializeToString())

                exampleP = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'spectP': _bytes_feature(spectP_b),
                        }
                    )
                )
                writerP.write(exampleP.SerializeToString())

                exampleN = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'spectN': _bytes_feature(spectN_b),
                        }
                    )
                )
                writerN.write(exampleN.SerializeToString())


##################################################################
########   READ DATASET   ###################
##################################################################


anchor_dataset = tf.data.TFRecordDataset([file_pathA])
positive_dataset = tf.data.TFRecordDataset([file_pathP])
negative_dataset = tf.data.TFRecordDataset([file_pathN])


def _decode_img(img_bytes, IMG_SHAPE):
    img = tf.io.decode_raw(img_bytes, tf.float32)
    img.set_shape([IMG_SHAPE[0] * IMG_SHAPE[1] * IMG_SHAPE[2]])
    img = tf.reshape(img, IMG_SHAPE)
    return img

def _read_contrastive_tfrecord_A(serialized_example):
    feature_description = { 'spectA': tf.io.FixedLenFeature((), tf.string) }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    spectA = _decode_img(example['spectA'], IMG_SHAPE)
    return spectA

def _read_contrastive_tfrecord_P(serialized_example):
    feature_description = { 'spectP': tf.io.FixedLenFeature((), tf.string) }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    spectA = _decode_img(example['spectP'], IMG_SHAPE)
    return spectA

def _read_contrastive_tfrecord_N(serialized_example):
    feature_description = { 'spectN': tf.io.FixedLenFeature((), tf.string) }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    spectA = _decode_img(example['spectN'], IMG_SHAPE)
    return spectA

anchor_dataset = anchor_dataset.map(_read_contrastive_tfrecord_A)
positive_dataset = positive_dataset.map(_read_contrastive_tfrecord_P)
negative_dataset = negative_dataset.map(_read_contrastive_tfrecord_N)

anchor_dataset = anchor_dataset.batch(30).prefetch(1)
positive_dataset = positive_dataset.batch(30).prefetch(1)
negative_dataset = negative_dataset.batch(30).prefetch(1)

parsed_dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
#parsed_dataset = parsed_dataset.batch(30).prefetch(1)


#pdb.set_trace()

model = build_triplet_model(IMG_SHAPE)
model.compile(optimizer='adam') #, run_eagerly=True)
history = model.fit( parsed_dataset, batch_size=30, )

train_a = np.array([triplet[0] for triplet in triplets])
train_p = np.array([triplet[1] for triplet in triplets])
train_n = np.array([triplet[2] for triplet in triplets])

model2 = build_triplet_model(IMG_SHAPE)
model2.compile(optimizer='adam') #, run_eagerly=True)
####  compile and fit model  ####
history = model2.fit(
        [train_a, train_p, train_n],
        batch_size=30,)
