import os 
import sys 
import csv
import pickle
import argparse
import numpy as np
import pandas as pd 

test_csv_path = sys.argv[1]
prediction_csv_path = sys.argv[2]
movies_csv_path = sys.argv[3]
users_csv_path = sys.argv[4]

users_matrix_path = 'model/nP.pickle' 
movies_matrix_path = 'model/nQ.pickle'

with open(users_matrix_path,'rb') as handle:
    U = pickle.load(handle)
with open(movies_matrix_path,'rb') as handle:
    M = pickle.load(handle)

R = np.dot(U,M.T)

with open(prediction_csv_path,'w') as outfile:
    test_file = pd.read_csv(test_csv_path) 
    print('TestDataID,Rating',file=outfile) 
    for idx, row in test_file.iterrows():
        rating = R[row['UserID']-1][row['MovieID']-1]
        if rating > 5:
            rating = 5
        if rating < 1:
            rating = 1
        print('{},{}'.format(idx+1, rating),file=outfile)

