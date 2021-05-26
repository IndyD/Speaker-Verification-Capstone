import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, Lambda, Conv2D, MaxPooling2D
from tensorflow_addons.losses import contrastive_loss

import pdb
import utils
import sys


'''
def make_dummy_data():
    X1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([100, 50], maxval=1000, dtype=tf.int32))
    X2 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([100, 1], dtype=tf.float32))
    X = tf.data.Dataset.zip((X1, X2)).map(lambda x1, x2: {'x1': x1, 'x2': x2})
    y_true = tf.data.Dataset.from_tensor_slices(tf.random.uniform([100, 1], dtype=tf.float32))
    return X, y_true

pdb.set_trace()
'''


PARAMS = utils.config_init(sys.argv)

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

def build_embedding_model(IMG_SHAPE):
    ''' 
    Return and embedding model using the fist 7 layer of VGG16 w/ 2 dense layers
    This architechture is from the VGG7 implementaion of Velez
    '''
    ## Note that VGG16 was trained on 3 channels. We can't use the weights w/ 1 channel so set to None
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SHAPE))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1))

    return model

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




IMG_SHAPE = (
    PARAMS.DATA_GENERATOR.N_MELS,
    PARAMS.DATA_GENERATOR.MAX_FRAMES,
    1
)
model = build_siamese_model(IMG_SHAPE)
(pairs, labels) = utils.load(PARAMS.PATHS.PAIRS_PATH)
pairs_train, pairs_test, label_train, label_test = train_test_split(
    pairs, labels, test_size=PARAMS.DATA_GENERATOR.TEST_SPLIT, random_state=1
)
pairs_train_l = np.array([pair[0] for pair in pairs_train])
pairs_train_r = np.array([pair[1] for pair in pairs_train])
pairs_test_l = np.array([pair[0] for pair in pairs_test])
pairs_test_r = np.array([pair[1] for pair in pairs_test])
label_train = tf.cast(np.array(label_train), tf.float32)
label_test = tf.cast(np.array(label_test), tf.float32)

#pdb.set_trace()
#model.compile(loss=siamese_model.contrastive_loss, optimizer="adam")
model.compile(loss=contrastive_loss_with_margin(margin=1), optimizer="adam")
print("Training model...")
pdb.set_trace()
history = model.fit(
    [pairs_train_l, pairs_train_r], label_train,
    validation_data=([pairs_test_l, pairs_test_r], label_test),
    epochs=PARAMS.TRAINING.EPOCHS,
    verbose=1,
    )
pdb.set_trace()