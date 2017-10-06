import csv 
import sys
import math
import random as rd
import numpy as np
from numpy.linalg import inv

np.set_printoptions(suppress=True)


if __name__ == '__main__' : 

    # _trainingFilename = sys.argv[1]
    # _testingFilename  = sys.argv[2]
    _ansFilename      = sys.argv[1]
    _trainingFilename = 'train.csv'
    _testingFilename = 'test.csv'

    data = []

    for i in range(18):
        data.append([])

    n_row = 0
    text = open(_trainingFilename,'r',encoding='big5') 
    row = csv.reader(text,delimiter=",")
    for r in row:
        if n_row != 0:
            for i in range(3,27):
                if r[i] != "NR":
                    data[(n_row-1)%18].append(float(r[i]))
                else:
                    data[(n_row-1)%18].append(float(0))	
        n_row = n_row+1
    text.close()


    x = []
    y = []


    for i in range(12):
        # 一個月取連續10小時的data可以有471筆
        for j in range(471):
            x.append([])
            # 18種污染物
            for t in range(18):
                # 連續9小時
                for s in range(9):
                    x[471*i+j].append(data[t][480*i+j+s] )
                    y.append(data[9][480*i+j+9])
    x = np.array(x)
    y = np.array(y)

    # add square term
    # x = np.concatenate((x,x**2), axis=1)


    # add bias
    # x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

    print (x[0])
    print (x[0][89])
    print (y[1])
    print ("feature num : ",len(x[0]))
    print ("training num : ",len(x))

    trainingSet = x 

    iteration = 10000

    b = 1.0 
    w = np.full(len(x[0]),1.0)
    w[89] = 1.0
    lr_b = 10
    lr_w = np.full(len(x[0]),10)
    r = 2.5

    sigma_b = 0.0
    sigma_w = np.zeros(len(x[0]))

    for i in range(iteration) :

        if i%100 == 0 and i != 0 :
            print ("iter",i,"............done   error=",error) 

        grad_b = 0.0
        grad_w = np.zeros(len(x[0]))
         
        error = 0 

        for n in range(len(x)): 
            
            L = y[n] - b - trainingSet[n].dot(w)
            error += L

            grad_b =  2*r*b - 2*L 
            grad_w =  2*r*w - 2*L*trainingSet[n]

            sigma_b += grad_b**2 
            sigma_w += grad_w**2

            b = b - lr_b/np.sqrt(sigma_b)*grad_b
            w = w - lr_w/np.sqrt(sigma_w)*grad_w
    