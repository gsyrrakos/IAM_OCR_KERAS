import cv2
import os, random
import numpy as np

# # Input data generator
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from src.Preprocessor import preprocessor


char_list = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
letters = [letter for letter in char_list]


def labels_to_text(labels):  # letters의 index -> text (string)
    return ''.join(list(map(lambda x: letters[int(x)], labels)))


def text_to_labels(text):  # text를 letters 배열에서의 인덱스 값으로 변환
    return list(map(lambda x: letters.index(x), text))


def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)

    return dig_lst


def truncateLabel(text, maxTextLen):
    cost = 0
    for i in range(len(text)):
        if i != 0 and text[i] == text[i - 1]:
            cost += 2
        else:
            cost += 1
        if cost > maxTextLen:
            return text[:i]
    return text


class TextImageGenerator:
    def __init__(self, img_dirpath, img_w, img_h,
                 batch_size, downsample_factor, max_text_len=100):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor
        self.img_dirpath = img_dirpath  # image dir path
        self.img_dir = 0  # os.listdir(self.img_dirpath)  # images list
        self.n = 0  # len(self.img_dir)  # number of images
        self.nval = 0;
        self.indexes = list(range(self.n))
        self.indexesval = list(range(self.nval))
        self.cur_index = 0
        self.imgs = []  # np.zeros((self.n, self.img_h, self.img_w))
        self.imgsval = []
        self.texts = []
        self.textsval = []
        self.train_images = []
        self.train_labels = []
        self.train_input_length = []
        self.train_label_length = []
        self.train_original_text = []

        self.valid_images = []
        self.valid_labels = []
        self.valid_input_length = []
        self.valid_label_length = []
        self.valid_original_text = []
        self.f = open("C:/Users/giorgos/Desktop/data/" + 'lines.txt')
        self.fnTrain = 'C:/Users/giorgos/Desktop/data/'
        self.index = 0
        self.max_label_len = 0
        self.DataAug = False

    def LoadTrain(self):
        for line in self.f:
            # ignore comment line
            self.index += 1

            # process image
            if not line or line[0] == '#':
                continue

            lineSplit = line.strip().split(' ')  ## remove the space and split with ' '
            assert len(lineSplit) >= 9

            # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
            fileNameSplit = lineSplit[0].split('-')
            # print(fileNameSplit)
            fileName = self.fnTrain + 'lines/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[
                1] + '/' + \
                       lineSplit[0] + '.png'

            # GT text are columns starting at 10
            # see the lines.txt and check where the GT text starts, in this case it is 10
            gtText_list = lineSplit[8].split('|')
            word = ' '.join(gtText_list)
            # print(index)
            img = fileName  # preprocessor(cv2.imread(fileName, cv2.IMREAD_GRAYSCALE), (800, 64), False, False)

            # process label

            label = encode_to_labels(word)

            if self.index % 10 == 0:
                self.valid_images.append(img)
                self.valid_labels.append(label)
                self.valid_input_length.append(100)
                self.valid_label_length.append(len(word))
                self.valid_original_text.append(word)
            else:
                self.train_images.append(img)
                self.train_labels.append(label)
                self.train_input_length.append(100)
                self.train_label_length.append(len(word))
                self.train_original_text.append(word)

            if len(word) > self.max_label_len:
                self.max_label_len = len(word)
            self.n = len(self.train_labels)
            self.nval = len(self.valid_labels)
            self.indexes = list(range(self.n))
            self.indexesval = list(range(self.nval))

    ## samples의 이미지 목록들을 opencv로 읽어 저장하기, texts에는 label 저장
    def build_data(self):
        print(self.n, " Image Loading start...")
        train_padded_label = pad_sequences(self.train_labels,
                                           maxlen=self.max_label_len,
                                           padding='post',
                                           value=len(char_list))
        for i, imgfile in enumerate(self.train_images):
            self.imgs.append(imgfile)
            self.texts.append((self.train_original_text[i]))
        print(len(self.texts) == self.n)

        # self.texts=pad_sequences( self.texts, maxlen=self.max_label_len, padding='post', value=len(char_list))
        print(self.n, " Image  Loading finish...")

        for i, imgfile in enumerate(self.valid_images):
            self.imgsval.append(imgfile)
            self.textsval.append((self.valid_original_text[i]))
        print(len(self.textsval) == self.nval)

        # self.textsval = pad_sequences(self.textsval, maxlen=self.max_label_len, padding='post', value=len(char_list))
        print(self.nval, " Image val Loading finish...")

    def next_sample(self, bool):  ## index max -> 0 으로 만들기
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]], \
               self.train_input_length[self.indexes[self.cur_index]], self.train_label_length[
                   self.indexes[self.cur_index]]

    def next_batch(self, bool):  ## batch size만큼 가져오기

        while True:

            X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])  # (bs, 128, 64, 1)
            Y_data = np.zeros([self.batch_size, self.max_label_len])  # (bs, 9)
            input_length = np.ones((self.batch_size, 1), dtype=np.float32) * (self.img_w // 8)
            label_length = np.zeros((self.batch_size, 1), dtype=np.float32)

            for i in range(self.batch_size):
                img, text, train_input_length, train_label_length = self.next_sample(bool)
                # img = img.T
                img = preprocessor(cv2.imread(img, cv2.IMREAD_GRAYSCALE), (800, 64), False,
                                   bool)
                # img = np.array(img)
                img = np.expand_dims(img, -1)

                X_data[i] = img
                text = truncateLabel(text, self.max_label_len)
                len_text = len(text)
                # print(text)
                # print(text_to_labels(text))
                # print(truncateLabel(text, self.max_label_len))

                Y_data[i, :len_text] = text_to_labels(text)
                # print(Y_data[i, :len_text])

                label_length[i] = len(text)

            # dict 형태로 복사

            inputs = {
                'the_input': X_data,  # (bs, 128, 64, 1)
                'the_labels': Y_data,  # (bs, 8)
                'input_length': input_length,  # (bs, 1) -> 모든 원소 value = 30
                'label_length': label_length  # (bs, 1) -> 모든 원소 value = 8
            }
            outputs = {'ctc': np.zeros([self.batch_size])}  # (bs, 1) -> 모든 원소 0
            yield (inputs, outputs)

    def next_sampleval(self, bool):  ## index max -> 0 으로 만들기
        self.cur_index += 1
        if self.cur_index >= self.nval:
            self.cur_index = 0
            random.shuffle(self.indexesval)
        return self.imgsval[self.indexesval[self.cur_index]], self.textsval[self.indexesval[self.cur_index]], \
               self.valid_input_length[self.indexesval[self.cur_index]], self.valid_label_length[
                   self.indexesval[self.cur_index]]

    def next_batchval(self, bool):  ## batch size만큼 가져오기

        while True:

            X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])  # (bs, 128, 64, 1)
            Y_data = np.zeros([self.batch_size, self.max_label_len])  # (bs, 9)
            input_length = np.ones((self.batch_size, 1), dtype=np.float32) * (self.img_w // 8)
            label_length = np.zeros((self.batch_size, 1), dtype=np.float32)

            for i in range(self.batch_size):
                img, text, val_input_length, val_label_length = self.next_sampleval(bool)
                # img = img.T
                img = preprocessor(cv2.imread(img, cv2.IMREAD_GRAYSCALE), (800, 64), False,
                                   bool)
                # img = np.array(img)
                img = np.expand_dims(img, -1)

                X_data[i] = img
                text = truncateLabel(text, self.max_label_len)
                len_text = len(text)

                Y_data[i, :len_text] = text_to_labels(text)  # encode_to_labels(text)

                label_length[i] = len(text)

            # dict 형태로 복사

            inputs = {
                'the_input': X_data,  # (bs, 128, 64, 1)
                'the_labels': Y_data,  # (bs, 8)
                'input_length': input_length,  # (bs, 1) -> 모든 원소 value = 30
                'label_length': label_length  # (bs, 1) -> 모든 원소 value = 8
            }
            outputs = {'ctc': np.zeros([self.batch_size])}  # (bs, 1) -> 모든 원소 0
            yield (inputs, outputs)
