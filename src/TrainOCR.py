from src.ImageDataGenerator import TextImageGenerator
from src.Models import get_Model
import fnmatch
from random import random

import numpy as np
import cv2
import os
import pandas as pd
import string
import matplotlib.pyplot as plt

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

# ignore warnings in the output
from tensorflow.python.keras import *
from tensorflow.python.keras.backend import ctc_batch_cost, reverse, concatenate
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Activation, MaxPooling2D, Conv2D, MaxPool2D, Reshape, Dense, LSTM, Lambda, \
    add, BatchNormalization
from tensorflow.python.keras.optimizer_v2.adadelta import Adadelta
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD

from src.Augment import *
from src.ImageDataGenerator import TextImageGenerator

from src.Preprocessor import preprocessor

img_w, img_h = 800, 64
batch_size = 50
val_batch_size = 10
epochs = 100
e = str(epochs)

train_file_path = './DB/train/'
tiger_train = TextImageGenerator(train_file_path, img_w, img_h, batch_size, 4)
tiger_train.LoadTrain()
tiger_train.build_data()

model = get_Model(training=True)

try:
    model.load_weights('C:/Users/giorgos/PycharmProjects/ParagraphOCR/model/' + 'LSTM+BN5--03--15.740.hdf5')
    print("...Previous weight data...")
except:
    print("...New weight data...")
    pass

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=2400,
    decay_rate=0.96,
)

ada = tf.keras.optimizers.Adam(lr_schedule)

# sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

early_stop = EarlyStopping(monitor='val_loss', patience=4, mode='auto', verbose=1)
# early_stop = EarlyStopping(monitor='loss', patience=4,  mode='auto', verbose=1)

checkpoint = ModelCheckpoint(filepath='../model/' + 'LSTM+BN5--{epoch:02d}--{val_loss:.3f}.hdf5', monitor='val_loss',
                             verbose=1, mode='min', period=1, save_best_only=True)
callbacks_list = [checkpoint]
# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)

# captures output of softmax so we can decode the output during visualization
model.fit_generator(generator=tiger_train.next_batch(True),
                    steps_per_epoch=int(tiger_train.n / batch_size),
                    epochs=1000,
                    callbacks=callbacks_list,
                    validation_data=tiger_train.next_batchval(False),
                    validation_steps=int(tiger_train.nval / batch_size))
