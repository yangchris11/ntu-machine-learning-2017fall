import sys
import csv
import math
import random as rd
import numpy as np 
import pandas as pd 
from numpy.linalg import inv,pinv

np.set_printoptions(suppress=True)

FEATURE_NUM = 107

def sigmoid(w,x,b):
    return 1.0/(1.0+np.exp(-1*(np.dot(w,x)+b)))

def read(_filename):
    data = pd.read_csv(_filename)
    for column in data:
        mean = data[column].mean()
        std = data[column].std()
        data[column] = data[column].apply(lambda x:(x-mean)/std)
    return data

def readtest(_filename):
    data = pd.read_csv(_filename)
    for column in data:
        mean = data[column].mean()
        std = data[column].std()
        if std != 0:
            data[column] = data[column].apply(lambda x:(x-mean)/std)
    return data

def spiltClass(y):
    c0 = []
    c1 = []
    for i in range(len(y)): 
        if y[i] == 0 :
            c0.append(x[i])
        else :
            c1.append(x[i])
    return np.array(c0),np.array(c1) 


if __name__ == '__main__' : 


    _rawtrainfileX = sys.argv[1]
    _rawtrainfileY = sys.argv[2]
    _trainfileX    = sys.argv[3]
    _trainfileY    = sys.argv[4]
    _testfile      = sys.argv[5]
    _ansfile       = sys.argv[6]

    x = read(_trainfileX)
    test_x = readtest(_testfile)
    y = pd.read_csv(_trainfileY,header=0)  
    x = np.array(x)
    y = np.array(y)
    test_x = np.array(test_x)

    [type0,type1] = spiltClass(y) 

    w0 = np.zeros(106)
    w1 = np.zeros(106)
    w0 = np.average(type0, axis=0)
    w1 = np.average(type1, axis=0)

    sigma0 = np.zeros(shape=(106,106))
    sigma1 = np.zeros(shape=(106,106))

    for i in range(len(type0)):
        tmp = type0[i] - w0 
        sigma0 += tmp.reshape(106,1) * tmp.reshape(1,106)
    for i in range(len(type1)):
        tmp = type1[i] - w1
        sigma1 += tmp.reshape(106,1) * tmp.reshape(1,106)

    sigma0 /= len(type0)
    sigma1 /= len(type1)
    p = len(type0)/len(y) 
    sigma = p*sigma0 + (1-p)*sigma1

    w = ((w0-w1).reshape(1,106)).dot(inv(sigma))
    b = (-1/2*(w0.reshape(1,106).dot(inv(sigma))).dot(w0.reshape(106,1)) + 1/2*(w1.reshape(1,106).dot(inv(sigma))).dot(w1.reshape(106,1)) + np.log(len(type0)/len(type1)))

    ans = []

    for n in range(len(test_x)):
        tmp = sigmoid(w,test_x[n],b)
        ans.append([str(n+1)])
        if tmp>0.5 :
            ans[n].append(int(0))
        else:
            ans[n].append(int(1))

    text = open(_ansfile, "w+")
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    s.writerow(["id","label"])
    for i in range(len(ans)):
        s.writerow(ans[i]) 
    text.close()