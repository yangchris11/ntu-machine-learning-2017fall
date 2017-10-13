import csv 
import sys
import math
import random as rd
import numpy as np
# import matplotlib.pyplot as plt
from numpy.linalg import inv


TYPE = 18  
[AMB_TEMP,CH4,CO,NMHC,NO,NO2,NOx,O3,PM10,PM25,RAINFALL,RH,SO2,THC,WD_HR,WIND_DIREC,WIND_SPEED,WS_HR] = range(18)
wanted_features = [ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 ]
wanted_features = [ int(i) for i in wanted_features ]
# O3 = 7
# PM25 = 9
# SO2 = 12
# WD_HR = 14
# WD_DIREC = 15
# WIND_SPEED = 16
# WS_HR = 17  

np.set_printoptions(suppress=True)



if __name__ == '__main__' : 

    _testingFilename  = sys.argv[1]
    _ansFilename      = sys.argv[2]
    _trainingFilename = 'train.csv'
    # _testingFilename = 'test.csv'

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
                if t in wanted_features :
                    for s in range(5):
                        if t == 9 and data[t][480*i+j+s] == -1 :
                            data[t][480*i+j+s] = data[t][480*i+j+s-1]
                        x[471*i+j].append(data[t][480*i+j+s] )
                    if t == 9:
                        for s in range(5):
                            x[471*i+j].append(data[t][480*i+j+s]**2)
            y.append(data[9][480*i+j+5])
    x = np.array(x)
    y = np.array(y)

    # add square term
    # x = np.concatenate((x,x**2), axis=1)

    # add bias
    x = np.concatenate((x,np.ones((x.shape[0],1))), axis=1)

    # print ("feature num : ",len(x[0]))
    # print ("training num : ",len(x))

    trainingSet = x 


    iteration = 10000

    # b = 1.0 
    w = np.zeros(len(x[0]))
    w[8] = 1.0
    r = 0.1

    # sigma_b = 0.0
    sigma_w = np.zeros(len(x[0]))

    # plot error 
    history_err = []

    x_t = x.transpose()
    s_gra = np.zeros(len(x[0]))
    l_rate = 10

    for i in range(iteration):
        hypo = np.dot(x,w)
        loss = hypo - y
        cost = np.sum(loss**2) / len(x)
        cost_a  = math.sqrt(cost)
        gra = np.dot(x_t,loss) + 2*r*w
        s_gra += gra**2
        ada = np.sqrt(s_gra)
        w = w - l_rate * gra/ada
        # print ('iteration: %d | Cost: %f  ' % ( i,cost_a))
        history_err.append(cost_a)

    test_x = []
    n_row = 0
    f_row = 0 
    text = open(_testingFilename,"r")
    row = csv.reader(text , delimiter= ",")

    for r in row:
        if n_row % 18 == 0:
            test_x.append([])
            f_row = 0
            # for i in range(2,11):
            #     test_x[n_row//18].append(float(r[i]) )
        else :
            if f_row in wanted_features :
                for i in range(6,11):
                    if f_row == 9 and r[i] == -1:
                        r[i] = r[i+1]
                    if r[i] !="NR":
                        test_x[n_row//18].append(float(r[i]))
                    else:
                        test_x[n_row//18].append(0)
                if f_row == 9 :
                    for i in range(6,11):
                        test_x[n_row//18].append(int(r[i])**2)
        n_row += 1 
        f_row += 1 
    text.close()
    test_x = np.array(test_x)

    # test_x = np.concatenate((test_x,test_x**2), axis=1)

    test_x = np.concatenate((test_x,np.ones((test_x.shape[0],1))), axis=1)


    ans = []
    for i in range(len(test_x)):
        ans.append(["id_"+str(i)])
        a = np.dot(w,test_x[i])
        ans[i].append(a)


    filename = _ansFilename
    text = open(filename, "w+")
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    s.writerow(["id","value"])
    for i in range(len(ans)):
        s.writerow(ans[i]) 
    text.close()