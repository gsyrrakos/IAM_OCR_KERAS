
import tensorflow as tf

# ignore warnings in the output
from tensorflow.python.keras import *
from tensorflow.python.keras.backend import ctc_batch_cost, reverse, concatenate, squeeze
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Activation, MaxPooling2D, Conv2D, MaxPool2D, Reshape, Dense, LSTM, Lambda, \
    add, BatchNormalization, Bidirectional






tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
char_list = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    return ctc_batch_cost(labels, y_pred, input_length, label_length)
# input with shape of height=32 and width=128

# inputs = Input(shape=(32, 128))
# inputs = tf.expand_dims(input=inputs, axis=3)
def get_Model(training):
    input_shape = (800, 64, 1)  # (128, 64, 1)

    # Make Networkw
    inputs = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 128, 64, 1)
    with tf.name_scope('Conv_Pool_1'):
        # convolution layer with kernel size (3,3)
        conv_1 = Conv2D(64, (5, 5), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding='SAME', strides=1)(inputs)
        # poolig layer with kernel size (2,2)
        pool_1 = MaxPool2D(pool_size=(2, 2))(conv_1)

    with tf.name_scope('Conv_Pool_2'):
        conv_2 = Conv2D(128, (5, 5), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding='SAME', strides=1)(pool_1)
        pool_2 = MaxPool2D(pool_size=(1, 2))(conv_2)

    with tf.name_scope('Conv_Pool_BN_3'):
        conv_3 = Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding='SAME', strides=1)(pool_2)
        batch_norm_3 = BatchNormalization()(conv_3)
        pool_3 = MaxPool2D(pool_size=(2, 2))(batch_norm_3)

    with tf.name_scope('Conv_4'):
        conv_4 = Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding='SAME', strides=1)(pool_3)
    # poolig layer with kernel size (2,1)

    with tf.name_scope('Conv_Pool_5'):
        conv_5 = Conv2D(256, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding='SAME', strides=1)(conv_4)
        # Batch normalization layer
        pool_5 = MaxPool2D(pool_size=(2, 2))(conv_5)

    with tf.name_scope('Conv_Pool_BN_6'):
        conv_6 = Conv2D(256, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding='SAME', strides=1)(pool_5)
        batch_norm_6 = BatchNormalization()(conv_6)
        pool_6 = MaxPool2D(pool_size=(1, 2))(batch_norm_6)

    with tf.name_scope('Conv_Pool_7'):
        conv_7 = Conv2D(512, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding='SAME', strides=1)(pool_6)
        pool_7 = MaxPool2D(pool_size=(1, 2))(conv_7)

    # CNN to RNN
    inner = Reshape(target_shape=((100, 512)), name='reshape')(pool_7)  # (None, 32, 2048)


    squeezed = inner
    blstm_1 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.2))(squeezed)
    blstm_2 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.2))(blstm_1)
    outputs = Dense(len(char_list) + 1)(blstm_2)
    y_pred = Activation('softmax', name='softmax')(outputs)

    the_labels = Input(name='the_labels', shape=[100], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, the_labels, input_length, label_length])
    if training:
        return Model(inputs=[inputs, the_labels, input_length, label_length], outputs=loss_out)
    else:
        return Model(inputs=[inputs], outputs=y_pred)


act_model = get_Model(training=False)

act_model.summary()






