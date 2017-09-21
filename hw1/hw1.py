import sys
import time 
import numpy as np 
import pandas as pd 

TYPE = 18  # {AMB_TEMP,CH4,CO,NMHC,NO,NO2,NOx,O3,PM10,PM2.5,RAINFALL,RH,SO2,THC,WD_HR,WIND_DIREC,WIND_SPEED,WS_HR}
PM25 = 9 # PM2.5 = no.9 element 
MONTH = 12 
HOUR = 24 
TRAININGDATE = 20 
KEY = [0,0,0,0]

def sliceData( rawData ):
    
    slicedData = []
    trainingSet = []
    for mm in range(MONTH):
        for hh in range(mm*TRAININGDATE*HOUR-9):
            for i in range(9):
                slicedData.extend(rawData[hh+i])
            ans = rawData[hh+PM25][PM25]
            trainingSet.append([ans,slicedData])
            slicedData = []

    return trainingSet



def readTestingData( filename ):
    
    originalcsvData = pd.read_csv(filename)
    rawData = []

    for mm in range(MONTH) :
        for dd in range(TRAININGDATE) :
            for hh in range(HOUR) :
                tmp = originalcsvData.ix[dd*18:dd*18+TYPE-1,str(hh)].values
                for i in range(TYPE) :
                    if tmp[i] == 'NR' : 
                        tmp[i] = 0 
                    else :
                        tmp[i] = float(tmp[i])
                rawData.append(tmp)

    trainingSet = sliceData(rawData)

    return trainingSet





if __name__ == '__main__' : 

    tStart = time.time()
    
    _trainingFilename = "olddata/train.csv"
    trainingSet = readTestingData(_trainingFilename)


    print "FUCK"
    

    tEnd = time.time() 

    print "Run time :" , tEnd-tStart , "second(s)" 

'''

def AdaGrad(f, gf, n, trainSet, theta,T):
    gd_sq_sum = np.zeros(n, dtype=float)
    eta = 1
    e = 1e-8
    for t in range(1, T):
        g = gf(trainSet, theta)
        gd_sq_sum += g*g
        for i in range(0, n):
            theta[i] -= eta * g[i] / np.sqrt(gd_sq_sum[i] + e)
        grad_norm = np.linalg.norm(gf(trainSet, theta))
        #print "Itr = %d" % t
        #print "f(theta) =", f(trainSet, theta)
        #print "norm(grad) =", grad_norm
        if grad_norm < 1e-3:
            return theta
return theta

if __name__== '__main__':
  trainSet = train_data_parser("data/train.csv")
  testSet = test_data_parser("data/test_X.csv")
  
  w_init = np.zeros(163)
  w = AdaGrad(quadratic_loss, grad_f, 163, trainSet, w_init, 100000)
  
  labels = [getTestLabel(testData, w) for testData in testSet]
  ids = ['id_'+str(i) for i in range(len(labels))]
  
  output = pd.DataFrame({'id': ids, 'value': labels})
output.to_csv("linear_regression.csv", index=False) 

'''