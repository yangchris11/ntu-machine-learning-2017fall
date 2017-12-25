import os 
import sys
import csv
import time
import config
import pickle
import logging
import itertools
import numpy as np
import pandas as pd 
from termcolor import colored,cprint

import gensim
from gensim.models import Word2Vec

import jieba

from scipy import spatial

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


def cut(s):
    return list(jieba.cut(s))

def process(s):
  rtns = s.split('\t')
  rtns = [cut(rtn[rtn.find(':')+1 : ]) for rtn in rtns]
  return rtns

jieba.set_dictionary(config.jieba_dict_path)
word_embedding_model = Word2Vec.load(config.word_embedding_model_path)

with open(config.sentences_pickle_path,'rb') as handle:
    sentences = pickle.load(handle)

max_length = config.max_length

jieba_tokenizer_dict = {}

sen = sentences 
idx = 1 
for i in range(len(sen)):
    for j in range(len(sen[i])):
        if sen[i][j] not in jieba_tokenizer_dict :
            jieba_tokenizer_dict[sen[i][j]] = idx
            idx += 1

test_num = 1

compare = []
with open('data/self_ans.csv','r') as f:
    csvf = csv.reader(f)
    next(csvf)
    for row in itertools.islice(csvf, test_num):
    # for row in csvf:
        compare.append(row[1])
f.close()

total_test_case = 5060
with open('data/testing_data.csv', 'r') as f:
  csvf = csv.reader(f)
  next(csvf)
  predict = [] 
  for row in itertools.islice(csvf, test_num):
#   for row in csvf:
    choice_similiar = []

    U = process(row[1].replace(" ",""))
    Us = []
    for i in range(len(U)):
        Us += U[i]
    # print(Us)

    Rs = process(row[2].replace(" ",""))
    # print(Rs)

f.close()

embedding_matrix = np.zeros((53597, 500))
for key, value in jieba_tokenizer_dict.items(): # key = chinese value = idx
    if key in word_embedding_model:
        embedding_vector = word_embedding_model[key]
        if embedding_vector is not None:
            embedding_matrix[value] = embedding_vector

print(Us[0])

print(Rs[0])
print(Rs[1])
print(Rs[2])
print(Rs[3])
print(Rs[4])
print(Rs[5])




# model = load_model( 'first_model.h5' )
# preds = model.predict([])
# print(preds)
