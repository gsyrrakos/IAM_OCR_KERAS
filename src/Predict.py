import codecs

import cv2

import numpy as np

from keras import backend as K

from src.Models import get_Model
from src.Preprocessor import preprocessor


class FilePaths:
    """ Filenames and paths to data """
    fnCharList = '../data/charList.txt'
    fnWordCharList = '../data/wordCharList.txt'
    fnCorpus = '../data/corpus.txt'
    fnAccuracy = '../model/accuracy.txt'
    fnTrain = 'C:/Users/giorgos/Desktop/data/'
    fnInfer = '../data/self.png'  ## path to recognize the single image


char_list = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


def getdecoder():
    print('WordBeamCtc')

    chars = char_list
    wordChars = codecs.open(
        FilePaths.fnWordCharList, 'r').read()
    corpus = codecs.open(FilePaths.fnCorpus, 'r').read()

    # Decoder using the "NGramsForecastAndSample": restrict number of (possible) next words to at most 20 words: O(W) mode of word beam search
    # decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(ctcIn3dTBC, dim=2), 25, 'NGramsForecastAndSample', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))

    # Decoder using the "Words": only use dictionary, no scoring: O(1) mode of word beam search
    from word_beam_search import WordBeamSearch
    decoder = WordBeamSearch(25, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'),
                             wordChars.encode('utf8'))
    return decoder


# Get CRNN model
model = get_Model(training=False)

try:
    model.load_weights('C:/Users/giorgos/PycharmProjects/ParagraphOCR/model/' + 'LSTM+BN5--54--10.734.hdf5')
    print("...Previous weight data...")
except:
    raise Exception("No weight file!")

test_dir = 'C:/Users/giorgos/PycharmProjects/ParagraphOCR/data/' + 'testImage.png'
test_imgs = 'Capture.png'


def Predict(img):
    img_pred = preprocessor(cv2.imread(img, cv2.IMREAD_GRAYSCALE), imgSize=(800, 64))
    img_pred = np.expand_dims(img_pred, -1)
    img_pred = np.expand_dims(img_pred, axis=0)
    # predict outputs on validation images
    prediction = model.predict(img_pred)

    # use CTC decoder
    # out = getdecoder().compute(prediction)
    # decoderOutputToText(out)

    # see the results
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                                   greedy=True)[0][0])
    i = 0
    pred = []
    for x in out:

        print("predicted text = ", end='')

        for p in x:
            if int(p) != -1:
                print(char_list[int(p)], end='')
                pred.append(char_list[int(p)])

        print('\n')

        i += 1
        string = ''.join([str(elem) for elem in pred])
    # print(string)
    return string


def PredictLine(img):
    img_pred = preprocessor(img, imgSize=(800, 64))
    img_pred = np.expand_dims(img_pred, -1)
    img_pred = np.expand_dims(img_pred, axis=0)
    # predict outputs on validation images
    prediction = model.predict(img_pred)

    # use CTC decoder
    # out = getdecoder().compute(prediction)
    # decoderOutputToText(out)

    # see the results
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                                   greedy=True)[0][0])
    i = 0
    pred = []
    for x in out:

        # print("predicted text = ", end='')

        for p in x:
            if int(p) != -1:
                # print(char_list[int(p)], end='')
                pred.append(char_list[int(p)])

        print('\n')

        i += 1
        string = ''.join([str(elem) for elem in pred])

    return string


Predict(test_dir)
