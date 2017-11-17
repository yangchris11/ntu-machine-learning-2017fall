import sys 
import csv
import numpy as np 
import pandas as pd 
# import matplotlib.pyplot as plt

import keras
from keras import backend as K
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping,LearningRateScheduler

from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, Adagrad
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

batch_size = 512
num_classes = 7
epoch = 200
opt = Adam( lr = 0.001 )

# input image dimensions
img_rows, img_cols = 48 , 48

# create and compile training model
def train_model():

    model = Sequential() 
    model.add(Dense(1024, input_dim=2304, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model 

def read( _trainfile ):

    raw_data = pd.read_csv( _trainfile ) 

    print ('Data set = ',len(raw_data))

    data = [] 
    for i in range(len(raw_data)):
        noise_ct = 0 
        label = raw_data['label'][i]
        feat = np.fromstring(raw_data['feature'][i],dtype=int,sep=' ')
        for j in range(2304):
            if feat[j] == 0 :
                noise_ct += 1
        if noise_ct < 1100 :
            feat = np.reshape(feat,(2304))
            data.append([label,feat])

    return data 

def spilt( data ):

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in range(len(data)):
        if i % 20 == 1 :
            y_test.append(data[i][0])
            x_test.append(data[i][1])
        else :
            y_train.append(data[i][0])
            x_train.append(data[i][1])

    x_train = np.array(x_train,dtype=float) / 255
    y_train = np_utils.to_categorical(np.array(y_train,dtype=int))
    x_test = np.array(x_test,dtype=float) / 255
    y_test = np_utils.to_categorical(np.array(y_test,dtype=int))
    
    print ('Training set = ',len(x_train))
    print ('Testing set = ',len(x_test))    

    return x_train , y_train , x_test , y_test

if __name__ == '__main__':

# system argv
    _trainFilename  = sys.argv[1] 

# read file
    input_data = read(_trainFilename) 

# spilt training/testing set 
    x_train , y_train , x_test , y_test = spilt(input_data) 

# model building
    K.set_image_dim_ordering('th')
    dnn_model = train_model() 
    print( dnn_model.summary() )

# Without ImageDataGenerator Training 
    callbacks = [
        ModelCheckpoint('dnn_whole_model.h5', monitor='val_acc', save_best_only=True, verbose=1)]
    history = dnn_model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=epoch,batch_size=batch_size,callbacks=callbacks)

# Testing
    scores = dnn_model.evaluate(x_test,y_test,verbose=0)
    print ("DNN Accuracy: %.2f%%" % (scores[1]*100))

# Model Saving
    dnn_model.save_weights("dnn_model_weight.h5")
    dnn_model.save('dnn_whole_model.h5')

# Plotting 
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('DNN Model Accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('# of epoch')
    # plt.legend(['training','validation'],loc='upper left')
    # plt.savefig('dnn_training_history.png')
