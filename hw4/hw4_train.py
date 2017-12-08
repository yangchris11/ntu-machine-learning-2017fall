import os 
import sys
import csv
import pickle
import logging
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

# model config
max_length = 30
word_embedding_model_dim = 200
batch_size = 512
adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
epoch = 500
labelFilename = sys.argv[1]
nolabelFilename = sys.argv[2]


# data preprocessing 

sentences = []
labels = []
x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
x_nolabel = []

f = open(nolabelFilename,'r',encoding='utf8')
for row in f :
	tmp_c = row.split()
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
	x_nolabel.append(tmp_str)
f.close()
print(colored("Loaded text from {}".format('./data/training_nolabel.txt', 'yellow')))


f = open(labelFilename,'r',encoding='utf8')
for row in f :
	tmp_l , tmp , tmp_d = row.split(' ',2)
	y_train.append(tmp_l)
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
	x_train.append(tmp_str)
f.close()
print(colored("Loaded text from {}".format('./data/training_label.txt', 'yellow')))

	
f = open('./testing_data.txt','r',encoding='utf8')
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
print(colored("Loaded text from {}".format('./data/testing_data.txt', 'yellow')))



# train model
word_embedding_model = Word2Vec(sentences,
				size=word_embedding_model_dim,
				min_count=3)
print(word_embedding_model)
word_embedding_model.save('./model/word_embedding_model.bin')
print(colored("Saved pretrained word-embedding model to {}".format('word_embedding_model.bin', 'red')))



t = Tokenizer(filters='\t\n')
t.fit_on_texts(x_train+x_test)
vocab_size = len(t.word_index) + 1

with open('./model/tokenizer.pickle', 'wb') as handle:
    pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)
x_train = t.texts_to_sequences(x_train)
x_test = t.texts_to_sequences(x_test)

x_train = pad_sequences(x_train,maxlen=max_length,padding='post')
x_test = pad_sequences(x_test,maxlen=max_length,padding='post')

# split
x_tmp = []
y_tmp = []
for i in range(len(x_train)):
	if i%100 == 0 :
		x_val.append(x_train[i])
		y_val.append(y_train[i])
	else :
		x_tmp.append(x_train[i])
		y_tmp.append(y_train[i])

x_train = x_tmp
y_train = y_tmp 

x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)

embedding_matrix = np.zeros((vocab_size, word_embedding_model_dim))
for word, i in t.word_index.items():
	if word in word_embedding_model :
		embedding_vector = word_embedding_model[word]
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
	else : 
		embedding_matrix[i] = embedding_matrix[0]



# rnn_model 

rnn_model = Sequential()
embedding_layer = Embedding( vocab_size , word_embedding_model_dim , weights = [embedding_matrix] , input_length = max_length , trainable = False )
rnn_model.add(embedding_layer)
rnn_model.add(GRU(512, recurrent_dropout = 0.4, dropout=0.4, return_sequences=True, activation='relu'))
rnn_model.add(GRU(256, recurrent_dropout = 0.4, dropout=0.4))
rnn_model.add(Dense(256, activation='relu'))
rnn_model.add(Dropout(0.3))
rnn_model.add(Dense(1, activation='sigmoid'))

rnn_model.compile(loss='binary_crossentropy', optimizer='Adam',metrics=['accuracy'])


print(rnn_model.summary())

callbacks = [EarlyStopping(monitor='val_acc',patience=5 , verbose=1),
			 ModelCheckpoint('./model/checkpoint_whole_model.h5', monitor='val_acc', save_best_only=True, verbose=1)]
    

rnn_model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epoch,
          validation_data=(x_val,y_val),
          callbacks=callbacks)
score, acc = rnn_model.evaluate(x_val, y_val,
                            batch_size=batch_size)


rnn_model.save('rnn_model.h5')

# rnn_model = load_model('checkpoint_whole_model.h5')

# preds = rnn_model.predict(x_test)

# ans = []
# for i in range(len(preds)) :
#     if preds[i] >= 0.5 :
#         ans.append(1) 
#     else :
#         ans.append(0) 

# csvFile = open( 'predict.csv' , 'w' )
# csvFile.write('id,label\n')
# for i in range( len(ans) ):
#     csvFile.write(str(i) + "," + str(ans[i]) + "\n")
