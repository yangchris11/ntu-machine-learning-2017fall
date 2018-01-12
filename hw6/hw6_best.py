import os
import sys
import math
import pickle
import numpy as np
import pandas as pd

from sklearn import *

import keras 
from keras.models import Model
from keras.layers import Dense, Input

image_npy_path = sys.argv[1]
test_case_path = sys.argv[2]
prediction_csv_path = sys.argv[3]

training_image = np.load(image_npy_path).astype('float64')

num_data, num_dim = training_image.shape
mean = training_image.mean(axis=0)
data_norm = training_image.copy()
for i in range(num_data):
    data_norm[i] -= mean
data_norm = data_norm.astype('float32')/255

# Training Params
latent_dim = 128 
epoch_time = 80
batch_size = 512

# Encoder 
input_img = Input(shape=(28*28,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
encoded_output = Dense(latent_dim)(encoded)

# Decoder
decoded = Dense(32, activation='relu')(encoded_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(28*28, activation='tanh')(decoded)

# Build Encoder
encoder = Model(input=input_img, output=encoded_output)

# Build Autoencoder
autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(data_norm, data_norm, epochs=40, batch_size=512, shuffle=True)
autoencoder.summary()

# Save Model
autoencoder.save('autoencoder.h5')
encoder.save('encoder.h5')


# KMeans
reduced_data = encoder.predict(data_norm)
clf = cluster.KMeans(init='k-means++', n_clusters=2, random_state=32)
clf.fit(reduced_data)

predict = clf.predict(reduced_data)

test = pd.read_csv(test_case_path)
img_1 = np.array(test.image1_index)
img_2 = np.array(test.image2_index)
ID = np.array(test.ID)

Ans = []
for i in range(len(ID)):
    if predict[img_1[i]] == predict[img_2[i]]:
        Ans.append(1)
    else:
        Ans.append(0)

result = open(prediction_csv_path, 'w')
result.write("ID,Ans\n")
for i in range(len(ID)):
    result.write(str(i)+','+str(Ans[i])+'\n')
result.close()