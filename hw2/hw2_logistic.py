import sys
import csv
import math
import random as rd
import numpy as np 
import pandas as pd 

np.set_printoptions(suppress=True)

FEATURE_NUM = 107

def sigmoid(x,w):
    return 1.0/(1.0+np.exp(-1*np.dot(x,w)))

def read(_filename):
    data = pd.read_csv(_filename)
    for column in data:
        mean = data[column].mean()
        std = data[column].std()
        if std != 0:
            data[column] = data[column].apply(lambda x:(x-mean)/std)
    data.insert(106,'Bias',1)    # add bais term
    return data

if __name__ == '__main__' : 

    _rawtrainfileX = sys.argv[1]
    _rawtrainfileY = sys.argv[2]
    _trainfileX    = sys.argv[3]
    _trainfileY    = sys.argv[4]
    _testfile      = sys.argv[5]
    _ansfile       = sys.argv[6]

    x = read(_trainfileX)
    y = pd.read_csv(_trainfileY,header=0)   

    x = np.array(x)     # get x_trainingSet in ndarray x
    y = np.array(y)     # gey y_trainingSet in ndarray y

    w = np.zeros(FEATURE_NUM)
    sigma = np.ones(FEATURE_NUM)

    learning_rate = 0.02
    lamda = 0
    iteration = 400

    rd.seed(666)
    val_x = []
    val_y = []
    tmp_x = []
    tmp_y = []
    for n in range(len(x)):
        seed = rd.randint(1,10)
        if seed == 0:
            val_x.append(x[n])
            val_y.append(y[n])
        else:
            tmp_x.append(x[n])
            tmp_y.append(y[n])

    val_x = np.array(val_x)
    val_y = np.array(val_y)
    x = np.array(tmp_x)
    y = np.array(tmp_y)

    # start training
    for i in range(iteration+1):      
        for n in range(len(x)):
            tmp = sigmoid(x[n],w)
            loss = y[n]-tmp
            grad = -1.0*loss*x[n] + lamda*w
            sigma = sigma + grad**2
            w = w - (learning_rate/np.sqrt(sigma))*grad

        if i%5 == 0:
            ct = 0 
            vct = 0
            for n in range(len(x)):
                tmp = sigmoid(x[n],w)
                if tmp>0.5 and y[n] == 1:
                    ct += 1
                elif y[n] == 0:
                    ct += 1
            for n in range(len(val_x)):
                vtmp = sigmoid(val_x[n],w)
                if vtmp>0.5 and val_y[n]==1 :
                    vct += 1
                elif y[n] == 0:
                    vct += 1
            print ("Iteration {0} : {1}/{2}={3:.2f}%".format(i,ct,len(x),100*ct/len(x)))

    test_x = read(_testfile)
    test_x = np.array(test_x)

    ans = []

    for n in range(len(test_x)):
        tmp = sigmoid(test_x[n],w)
        ans.append([str(n+1)])
        if tmp>0.5 :
            ans[n].append(int(1))
        else:
            ans[n].append(int(0))

    text = open(_ansfile, "w+")
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    s.writerow(["id","label"])
    for i in range(len(ans)):
        s.writerow(ans[i]) 
    text.close()
