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
from keras.models import Sequential, load_model
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

n_movies = 3952
n_users = 6040

model = load_model( dnn_model_path )

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
