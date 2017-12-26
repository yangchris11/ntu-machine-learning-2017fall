import os 
import sys
import csv
import config
import pickle
import logging
import numpy as np
import pandas as pd 
import random as rd 
from termcolor import colored,cprint

import time

import gensim
from gensim.models import Word2Vec

import jieba
jieba.set_dictionary(config.jieba_dict_path)

import keras
from keras.optimizers import SGD, Adam, Adagrad, RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,  load_model, Model
from keras.layers import Activation, Flatten, Dropout, Dense, Embedding, LSTM, GRU, Merge, Input 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adadelta

import keras.backend as K


from sklearn.model_selection import train_test_split



word_embedding_model = Word2Vec.load(config.word_embedding_model_path)

with open(config.sentences_pickle_path,'rb') as handle:
    sentences = pickle.load(handle)




max_length = config.max_length

padding_sequence = []
x_train_raw = []
x_train = []
x_train_q = []
x_train_a = []
x_val_q = []
x_val_a = []
y_train = []
y_val = []

jieba_tokenizer_dict = {}

print(word_embedding_model)

# for i in range(len(sentences)):
#     if(len(sentences[i])<4):
#         sentences[i] += sentences[i+1]

sen = sentences 
idx = 1 
for i in range(len(sen)):
    tmp = [] 
    for j in range(len(sen[i])):
        if sen[i][j] not in jieba_tokenizer_dict :
            jieba_tokenizer_dict[sen[i][j]] = idx
            idx += 1
        tmp.append(jieba_tokenizer_dict[sen[i][j]])
    x_train_raw.append(tmp)

for i in range(len(x_train_raw)-1):
    x_train.append([x_train_raw[i], x_train_raw[i+1]])
    y_train.append(1)
    x_train.append([x_train_raw[i],x_train_raw[rd.randint(0,len(x_train_raw)-1)]])
    y_train.append(0)
    x_train.append([x_train_raw[i],x_train_raw[rd.randint(0,len(x_train_raw)-1)]])
    y_train.append(0)

for i in range(len(x_train)):
    x_train[i] = pad_sequences(x_train[i],maxlen=max_length,padding='post')



print('pre-done')


X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.1)

for i in range(len(X_train)):
    x_train_q.append(X_train[i][0])
    x_train_a.append(X_train[i][1])


for i in range(len(X_val)):
    x_val_q.append(X_val[i][0])
    x_val_a.append(X_val[i][1])

x_train_q = np.array(x_train_q)
x_train_a = np.array(x_train_a)
x_val_q = np.array(x_val_q)
x_val_a = np.array(x_val_a)
Y_train = np.array(Y_train)
Y_val = np.array(Y_val)

print('split-done')


embedding_matrix = np.zeros((53597, 500))
for key, value in jieba_tokenizer_dict.items(): # key = chinese value = idx
    if key in word_embedding_model:
        embedding_vector = word_embedding_model[key]
        if embedding_vector is not None:
            embedding_matrix[value] = embedding_vector


print('embedding-done')

n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 64
n_epoch = 25

def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

# The visible layer
left_input = Input(shape=(max_length,), dtype='int32')
right_input = Input(shape=(max_length,), dtype='int32')

embedding_layer = Embedding(53597,500, weights=[embedding_matrix], input_length=max_length, trainable=False)

# Embedded version of the inputs
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

# Since this is a siamese network, both sides share the same LSTM
shared_lstm = LSTM(n_hidden)

left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)

# Calculates the distance as defined by the MaLSTM model
malstm_distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

# Pack it all up into a model
malstm = Model([left_input, right_input], [malstm_distance])

# Adadelta optimizer, with gradient clipping by norm
optimizer = Adadelta(clipnorm=gradient_clipping_norm)

malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

# Start training

malstm_trained = malstm.fit([x_train_q,x_train_a], Y_train, batch_size=batch_size, nb_epoch=n_epoch,
                            validation_data=([x_val_q,x_val_a], Y_val))




'''padding sense
if len() < 4 :
    average_len = 5.96
    max_len = 15
if len() < 5 :
    averge_len = 6.74
    max_len = 17
if len() < 6 :
    average)len = 7.51
    max_len = 18
'''