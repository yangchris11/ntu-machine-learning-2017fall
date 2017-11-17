import os 
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras
from keras import backend as K
from keras.models import load_model
from keras.utils import np_utils
from termcolor import colored,cprint

K.set_image_dim_ordering('th')

base_dir = './'
exp_dir = 'exp'
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
vis_dir = os.path.join('image','vis_layer')
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)
filter_dir = os.path.join('image','vis_filter')
if not os.path.exists(filter_dir):
    os.makedirs(filter_dir)


def read( filename ):

    raw_data = pd.read_csv( filename ) 
    x_test = []
    for i in range(len(raw_data)):
        feat = np.fromstring(raw_data['feature'][i],dtype=int,sep=' ')
        feat = np.reshape(feat,(1,48,48))
        x_test.append(feat) 
    x_test = np.array(x_test,dtype=float) / 255

    return x_test

def main():
    parser = argparse.ArgumentParser(prog='visFilter.py',
            description='ML-Assignment3 Visualize CNN filter.')
    parser.add_argument('--model',type=str,default='./model1/whole_model.h5',metavar='<model>')
    parser.add_argument('--data', type=str,default='./model1/train.csv', metavar='<#data>')
    parser.add_argument('--epoch',type=int,metavar='<#epoch>',default=20)
    parser.add_argument('--mode',type=int,metavar='<visMode>',default=1,choices=[1,2])
    parser.add_argument('--batch',type=int,metavar='<batch_size>',default=64)
    args = parser.parse_args()
    data_path = args.data
    model_path = args.model

    x = read( data_path )

    emotion_classifier = load_model(model_path)
    print(colored("Loaded model from {}".format(model_path), 'yellow'))

    input_img = emotion_classifier.input

    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])

    layer_name = [ 'conv2d_2']
    collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in layer_name]
    
    img_ids = 5688
    photo = x[img_ids].reshape(1,1,48,48)


    for cnt,fn in enumerate(collect_layers):
        im = fn([photo,0])
        fig = plt.figure(figsize=(14,8))
        nb_filter = 64
        print(layer_name[cnt],im[0].shape[3])
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/16,16,i+1)
            ax.imshow(im[0][0,i, :,:], cmap='GnBu')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
        img_path = os.path.join(filter_dir, 'vis')
        if not os.path.isdir(img_path):
            os.mkdir(img_path)
        fig.savefig(os.path.join(img_path, '{}-{}'.format(layer_name[cnt],img_ids)))


    
if __name__ == "__main__":
    main()