import os 
import sys
import csv
import logging
import pickle
import numpy as np
import pandas as pd 
from termcolor import colored,cprint

import gensim
from gensim.models import Word2Vec

import keras
from keras.optimizers import SGD, Adam, Adagrad, RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,  load_model
from keras.layers import Activation, Flatten, Dropout, Dense, Embedding, LSTM, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint


max_length = 30
word_embedding_model_dim = 200
batch_size = 512
adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
epoch = 50
semi_threshold_p = 0.9
semi_threshold_n = 0.1



testFilename = sys.argv[1]
predictFilename = sys.argv[2]
sentences = []
x_test = []

f = open(testFilename,'r',encoding='utf8')
next(f)
for row in f :
	tmp , tmp_d = row.split(',',1)
	tmp_c = tmp_d.split()
	i = 0 
	while i < len(tmp_c) :
		if ( tmp_c[i] == "'" ) and (i+1<len(tmp_c)) :
			tmp_c[i] = str(tmp_c[i-1])+"'"+str(tmp_c[i+1])
			tmp_c.pop(i+1)
			tmp_c.pop(i-1)
		else :
			i += 1
	sentences.append(tmp_c)
	tmp_str = ""
	for i in range(len(tmp_c)):
		tmp_str += str(tmp_c[i])+" "
	x_test.append(tmp_str)
f.close()
# print(colored("Loaded text from {}".format('./data/testing_data.txt', 'yellow')))

word_embedding_model = Word2Vec.load('./model/word_embedding_model.bin')
rnn_model = load_model('./model/checkpoint_whole_model.h5')
with open('./model/tokenizer.pickle', 'rb') as handle:
    t = pickle.load(handle)
vocab_size = len(t.word_index) + 1
x_test = t.texts_to_sequences(x_test)
x_test = pad_sequences(x_test,maxlen=max_length,padding='post')

embedding_matrix = np.zeros((vocab_size, word_embedding_model_dim))
for word, i in t.word_index.items():
	if word in word_embedding_model :
		embedding_vector = word_embedding_model[word]
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
	else : 
		embedding_matrix[i] = embedding_matrix[0]

preds = rnn_model.predict(x_test)

ans = []
for i in range(len(preds)) :
	if preds[i] >= 0.5 : 
		ans.append(1) 
	else :
		ans.append(0)
        
csvFile = open( predictFilename , 'w' )
csvFile.write('id,label\n')
for i in range( len(ans) ):
    csvFile.write(str(i) + "," + str(ans[i]) + "\n")
