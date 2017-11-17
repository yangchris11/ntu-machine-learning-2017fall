import sys 
import csv
import numpy as np 
import pandas as pd 

import keras
from keras import backend as K
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, Adagrad
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def read( _testFile ):

    raw_data = pd.read_csv( _testFile ) 
    print ('Testing set = ',len(raw_data))
    x_test = []
    for i in range(len(raw_data)):
        feat = np.fromstring(raw_data['feature'][i],dtype=int,sep=' ')
        feat = np.reshape(feat,(1,48,48))
        x_test.append(feat) 
    x_test = np.array(x_test,dtype=float) / 255

    return x_test

if __name__ == '__main__':

# system argv
    _testFilename   = sys.argv[1]
    _predictFilename = sys.argv[2]

# load model
    model = load_model("checkpoint_whole_model.h5")

# write output file
    x_test = read(_testFilename) 

# pridiction 
    prediction = model.predict(x_test)
    csvFile = open( _predictFilename , 'w' )
    csvFile.write('id,label\n')
    for i in range( len(prediction) ):
        csvFile.write(str(i) + "," + str(np.argmax(prediction[i])) + "\n")
