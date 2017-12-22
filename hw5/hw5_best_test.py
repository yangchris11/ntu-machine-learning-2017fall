import os 
import sys 
import csv
import pickle
import argparse
import numpy as np
import pandas as pd 

import keras
import keras.models as kmodels
import keras.layers as klayers
from keras.optimizers import SGD, Adam, Adagrad
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

import keras.backend as K

from sklearn import dummy, metrics, ensemble, cross_validation

test_csv_path = sys.argv[1]
prediction_csv_path = sys.argv[2]
movies_csv_path = sys.argv[3]
users_csv_path = sys.argv[4]

dnn_model_path = 'model/best_dnn_weight.h5'

test_data = pd.read_csv( test_csv_path , dtype = int )
test_data.MovieID = test_data.MovieID.astype('category')
test_data.UserID = test_data.UserID.astype('category')

# train_data = pd.read_csv( 'data/train.csv' ,dtype = int )
# train_data.MovieID = train_data.MovieID.astype('category')
# train_data.UserID = train_data.UserID.astype('category')

# movieid = np.array(train_data.MovieID.values)
# userid = np.array(train_data.UserID.values)
# y = np.zeros((train_data.shape[0], 5))
# y[np.arange(train_data.shape[0]), train_data.Rating - 1] = 1
# y = np.array(train_data.Rating)

n_movies = 3952
n_users = 6040

movie_input = keras.layers.Input(shape=[1])
movie_vec = keras.layers.Flatten()(keras.layers.Embedding(n_movies + 1, 50, trainable=True)(movie_input))
movie_vec = keras.layers.Dropout(0.4)(movie_vec)

user_input = keras.layers.Input(shape=[1])
user_vec = keras.layers.Flatten()(keras.layers.Embedding(n_users + 1, 50, trainable=True)(user_input))
user_vec = keras.layers.Dropout(0.4)(user_vec)

input_vecs = keras.layers.merge([movie_vec, user_vec], mode='concat')
nn = keras.layers.Dropout(0.35)(keras.layers.Dense(512, activation='relu')(input_vecs))
nn = keras.layers.normalization.BatchNormalization()(nn)
nn = keras.layers.Dropout(0.35)(keras.layers.Dense(256, activation='relu')(nn))
nn = keras.layers.normalization.BatchNormalization()(nn)
nn = keras.layers.Dropout(0.35)(keras.layers.Dense(128, activation='relu')(nn))
nn = keras.layers.normalization.BatchNormalization()(nn)
nn = keras.layers.Dropout(0.35)(keras.layers.Dense(64, activation='relu')(nn))
result = keras.layers.Dense(1, activation='relu')(nn)

callbacks = [EarlyStopping('val_loss', patience=10, verbose=1), 
              ModelCheckpoint(dnn_model_path, save_best_only=True, verbose=1)]
             
model = kmodels.Model([movie_input, user_input], result)
model.compile(Adam(lr=0.001), loss='mean_squared_error')

# print(model.summary())

# a_movieid, b_movieid, a_userid, b_userid, a_y, b_y = cross_validation.train_test_split(movieid, userid, y,test_size=0.05)
# model.fit([a_movieid, a_userid], a_y, epochs=200, batch_size=512, validation_data=([b_movieid, b_userid], b_y), callbacks=callbacks)

model.load_weights( dnn_model_path )

test_movieid = np.array(test_data.MovieID.values)
test_userid = np.array(test_data.UserID.values)

prediction = model.predict([test_movieid,test_userid])

with open(prediction_csv_path, 'w') as outfile:
    print('TestDataID,Rating',file=outfile)
    for idx, pred in enumerate(prediction):
        rating = pred[0]
        if rating > 5:
            rating = 5
        elif rating < 1:
            rating = 1
        print('{},{}'.format(idx+1, rating),file=outfile)
