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

import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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
            feat = np.reshape(feat,(1,48,48))
            data.append([label,feat])

    return data 

def spilt( data ):

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in range(len(data)):
        if i % 50 == 1 :
            y_test.append(data[i][0])
            x_test.append(data[i][1])
        else :
            y_train.append(data[i][0])
            x_train.append(data[i][1])

    y_ans = y_test

    x_train = np.array(x_train,dtype=float) / 255
    y_train = np_utils.to_categorical(np.array(y_train,dtype=int))
    x_test = np.array(x_test,dtype=float) / 255
    y_test = np_utils.to_categorical(np.array(y_test,dtype=int))
    
    print ('Training set = ',len(x_train))
    print ('Testing set = ',len(x_test))    

    return x_train , y_train , x_test , y_test , y_ans

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == '__main__':

# system argv 
    _trainFilename = sys.argv[1]
    _modelPath = sys.argv[2]

# read file
    input_data = read(_trainFilename) 

# load model
    model = load_model(_modelPath)

# spilt training/testing set 
    x_train , y_train , x_test , y_test ,y_ans = spilt(input_data)    

# print confusion matrix 
    prediction = model.predict(x_test)
    y_preds = []
    for i in range(len(prediction)):
        y_preds.append(np.argmax(prediction[i]))

    cnf_matrix = confusion_matrix(y_ans,y_preds)
    np.set_printoptions(precision=2)

    class_names = ['Anger','Disgust','Fear','Joy','Sadness','Surprised','Neutral']

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                        title='Normalized confusion matrix')
    plt.show()