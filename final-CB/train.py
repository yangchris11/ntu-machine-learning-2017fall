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
from keras.models import Sequential,  load_model
from keras.layers import Activation, Flatten, Dropout, Dense, Embedding, LSTM, GRU, Merge
from keras.callbacks import EarlyStopping, ModelCheckpoint



word_embedding_model = Word2Vec.load(config.word_embedding_model_path)

with open(config.sentences_pickle_path,'rb') as handle:
    sentences = pickle.load(handle)



batch_size = 512
epoch = 500
adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
max_length = config.max_length

padding_sequence = []
x_train_raw = []
x_train = []
x_train_q = []
x_train_a = []
y_train = []

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

for i in range(len(x_train)):
    x_train[i] = pad_sequences(x_train[i],maxlen=max_length,padding='post')



print(x_train[0][0])
print(x_train[0][1])
print(y_train[0])

print(x_train[5][0])
print(x_train[5][1])
print(y_train[5])





embedding_matrix = np.zeros((53597, 500))
for key, value in jieba_tokenizer_dict.items(): # key = chinese value = idx
    if key in word_embedding_model:
        embedding_vector = word_embedding_model[key]
        if embedding_vector is not None:
            embedding_matrix[value] = embedding_vector


embedding_layer = Embedding( 53597 , 500 , weights = [embedding_matrix] , input_length = max_length , trainable = False )



left_branch = Sequential()
left_branch.add(embedding_layer)
left_branch.add(GRU(512, recurrent_dropout = 0.4, dropout=0.4, return_sequences=True, activation='relu'))
left_branch.add(GRU(256, recurrent_dropout = 0.4, dropout=0.4, activation='relu'))
left_branch.add(Dense(256, activation='relu'))
left_branch.add(Dropout(0.3))
left_branch.add(Dense(1, activation='sigmoid'))


right_branch = Sequential()
right_branch.add(embedding_layer)
right_branch.add(GRU(512, recurrent_dropout = 0.4, dropout=0.4, return_sequences=True, activation='relu'))
right_branch.add(GRU(256, recurrent_dropout = 0.4, dropout=0.4, activation='relu'))
right_branch.add(Dense(256, activation='relu'))
right_branch.add(Dropout(0.3))
right_branch.add(Dense(1, activation='sigmoid'))

merged = Merge([left_branch, right_branch], mode='concat')
final_model = Sequential()
final_model.add(merged)
final_model.add(Dropout(0.3))
final_model.add(Dense(1, activation='sigmoid'))


final_model.compile(optimizer='Adam', loss='binary_crossentropy')

callbacks = [EarlyStopping(monitor='val_acc',patience=5 , verbose=1),
			 ModelCheckpoint('model/checkpoint_whole_model.h5', monitor='val_acc', save_best_only=True, verbose=1)]

final_model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epoch,
          validation_data=(x_val,y_val),
          callbacks=callbacks)



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