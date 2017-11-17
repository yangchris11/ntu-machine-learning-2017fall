import os 
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
img_dir = os.path.join(base_dir, 'image')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
cmap_dir = os.path.join(img_dir, 'cmap')
if not os.path.exists(cmap_dir):
    os.makedirs(cmap_dir)
partial_see_dir = os.path.join(img_dir,'partial_see')
if not os.path.exists(partial_see_dir):
    os.makedirs(partial_see_dir)
origin_dir = os.path.join(img_dir,'origin')
if not os.path.exists(origin_dir):
    os.makedirs(origin_dir)

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

    parser = argparse.ArgumentParser(prog='plot_saliency.py',
            description='ML-Assignment3 visualize attention heat map.')
    parser.add_argument('--epoch', type=int, metavar='<#epoch>', default=1)
    parser.add_argument('--model', type=str, metavar='<#model>', default='./model1/whole_model.h5')
    parser.add_argument('--data', type=str, metavar='<#data>', default='./model1/train.csv')
    args = parser.parse_args()
    model_path = args.model
    data_path = args.data

    emotion_classifier = load_model(model_path)
    print(colored("Loaded model from {}".format(model_path), 'yellow'))

    x = read( data_path )

    input_img = emotion_classifier.input 
    img_ids = [1]

    for idx in img_ids:

        val_proba = emotion_classifier.predict(x[idx].reshape(-1, 1, 48, 48))
        pred = val_proba.argmax(axis=-1)
        target = K.mean(emotion_classifier.output[:,pred])
        grads = K.gradients(target, input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [grads])

        val_grads = fn([x[idx].reshape(-1, 1, 48, 48), 0])[0].reshape(48, 48, -1)

#  gradient normalize 
        val_grads *= -1
        val_grads = np.max(np.abs(val_grads), axis=-1, keepdims=True)
        val_grads = (val_grads - np.mean(val_grads)) / (np.std(val_grads) + 1e-5)
        val_grads *= 0.1
        val_grads += 0.5
        val_grads = np.clip(val_grads, 0, 1)
        val_grads /= np.max(val_grads)

        heatmap = val_grads.reshape(48, 48)

#  original figure
        plt.figure()
        plt.imshow(x[idx].reshape(48, 48), cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(origin_dir, '{}.png'.format(idx)), dpi=100)
        print(colored("Saving image {}".format('{}.png'.format(idx)), 'red'))


#  heatmap figure
        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(cmap_dir, 'cmap{}.png'.format(idx)), dpi=100)
        print(colored("Saving image {}".format('cmap{}.png'.format(idx)), 'red'))

# masked figure 
        thres = 0.55
        see = x[idx].reshape(48, 48)
        see[np.where(heatmap <= thres)] = np.mean(see)
        plt.figure()
        plt.imshow(see,cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(partial_see_dir, 'masked{}.png'.format(idx)), dpi=100)
        print(colored("Saving image {}".format('masked{}.png'.format(idx)), 'red'))

if __name__ == "__main__":
    main()
    