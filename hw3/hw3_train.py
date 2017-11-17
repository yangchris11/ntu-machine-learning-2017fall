
import sys 
import csv
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

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
epoch = 400
opt = Adam( lr = 0.00027 )

# input image dimensions
img_rows, img_cols = 48 , 48

# create and compile training model
def train_model():

    model = Sequential() 

    model.add(Conv2D(32, (3, 3), input_shape=(1,48,48), activation='relu',padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.35))
    
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.45))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model 

noise_idx = []

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
            noise_idx.append(i)
            feat = np.reshape(feat,(1,48,48))
            data.append([label,feat])

    return data 

def spilt( data ):

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in range(len(data)):
        if i % 100 == 1 :
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
    cnn_model = train_model() 
    print( cnn_model.summary() )

# ImageDataGenerator Training
    datagenerator = ImageDataGenerator(
        featurewise_center = False, 
        samplewise_center = False, 
        featurewise_std_normalization = False,  
        samplewise_std_normalization = False,  
        zca_whitening = False,  
        rotation_range = 20,  
        width_shift_range = 0.1, 
        height_shift_range = 0.1, 
        horizontal_flip = True,  
        vertical_flip = False) 
    datagenerator.fit(x_train)
    callbacks = [
        ModelCheckpoint('checkpoint_whole_model.h5', monitor='val_acc', save_best_only=True, verbose=1),]
    history = cnn_model.fit_generator(
            datagenerator.flow(x_train, y_train,batch_size=batch_size,seed=7),
            steps_per_epoch = x_train.shape[0] // batch_size,
            epochs = epoch,
            validation_data = (x_test, y_test),
            callbacks = callbacks)

# Without ImageDataGenerator Training 
    # cnn_model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=0,batch_size=batch_size)

# Testing
    scores = cnn_model.evaluate(x_test,y_test,verbose=0)
    print ("CNN Error: %.2f%%" % (100-scores[1]*100))

# Model Saving
    cnn_model.save_weights("cnn_model_weight.h5")
    cnn_model.save('cnn_whole_model.h5')

# Print
    print(history.history['val_acc'])
    print(history.history['acc'])
