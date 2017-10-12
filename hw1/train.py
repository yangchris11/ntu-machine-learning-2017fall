import sys
import csv 
import time 
import random as rd 
import numpy as np 
import pandas as pd 

np.set_printoptions(suppress=True)

TYPE = 18  
[AMB_TEMP,CH4,CO,NMHC,NO,NO2,NOx,O3,PM10,PM25,RAINFALL,RH,SO2,THC,WD_HR,WIND_DIREC,WIND_SPEED,WS_HR] = range(18)
wanted_features = [ 7,9,12,14,15,16,17 ]
wanted_features = [ int(i) for i in wanted_features ]
# O3 = 7
# PM25 = 9
# SO2 = 12
# WD_HR = 14
# WD_DIREC = 15
# WIND_SPEED = 16
# WS_HR = 17  
MONTH = 12 
HOUR = 24 
TRAININGDATE = 20 
TESTCASE = 240 
TESTHOUR =  9 
FEATURENUM = len(wanted_features)+1

def sliceData( rawData ):
    
    slicedData = []
    trainingSet = []
    num = 0 

    print (len(rawData))
  
    for mm in range(MONTH):
        for hh in range(TRAININGDATE*HOUR-9):
            for i in range(9):
                for j in wanted_features :
                    if j == 9 :
                        if rawData[mm*480+hh+i][j] == -1 :
                            rawData[mm*480+hh+i][j] = rawData[mm*480+hh+i-1][j]
                    slicedData.append(float(rawData[mm*480+hh+i][j]))
            for i in range(9):
                slicedData.append(float(slicedData[i*7+1]**2))
            ans = rawData[mm*480+hh+9][PM25]
            trainingSet.append([ans,np.array(slicedData)])
            slicedData = []
            num += 1 

    return trainingSet

def readTrainingData( filename ):
    
    originalcsvData = pd.read_csv(filename,encoding='big5')
    rawData = []

    for mm in range(MONTH) :
        for dd in range(TRAININGDATE) :
            for hh in range(HOUR) :
                tmp = originalcsvData.ix[mm*360+dd*18:mm*360+dd*18+TYPE-1,str(hh)].values
                for i in range(TYPE) :
                    if tmp[i] == 'NR' : 
                        tmp[i] = 0 
                    else :
                        tmp[i] = float(tmp[i])
                rawData.append(tmp)

    trainingSet = sliceData(rawData)

    return trainingSet

def readTestingData( filename ):

    testingSet = []
    
    with open(filename,'r',encoding='big5') as testingFile :
        count = 0
        ftmp = 0 
        feat = np.zeros(FEATURENUM*TESTHOUR)
        for row in csv.reader(testingFile):
            if count%18 != 17 :
                if count in wanted_features :
                    # print (count,row)
                    for hh in range(TESTHOUR):
                        feat[ftmp+hh*7] = row[2+hh]
                    ftmp += 1
                count += 1
            elif count%18 == 17 : 
                # print (count,row)
                for hh in range(TESTHOUR):
                    feat[ftmp+hh*7] = row[2+hh]
                ftmp = 0
                count = 0
                for i in range(9):
                    if feat[i*7+1] == -1 :
                        feat[i*7+1] = feat[i*7+8]
                    feat[63+i] = feat[i*7+1]**2
                testingSet.append(np.array(feat))

    return testingSet 
        
def shuffletrainingData( trainingSet ):
    
    num = len(trainingSet)
    rd.seed(666)
    _trainSet = []
    _valSet = []
    _testSet = []
    for i in range(num):
        sample = rd.randint(1,3) 
        if sample == 1 :
            _trainSet.append(trainingSet[i])
        elif sample == 3 :
            _valSet.append(trainingSet[i])
        else :
            _testSet.append(trainingSet[i])

    return _trainSet,_valSet,_testSet


if __name__ == '__main__' : 

    tStart = time.time()

    # _trainingFilename = sys.argv[1]
    _testingFilename  = sys.argv[1]
    _ansFilename      = sys.argv[2]
    
    _trainingFilename = "train.csv"
    # _testingFilename = "test.csv"

    trainingSet = readTrainingData(_trainingFilename)
    testingSet = readTestingData(_testingFilename)

    _trainSet,_valSet,_testSet = shuffletrainingData( trainingSet )

    trainingSet = _trainSet

    rd.shuffle(_trainSet)


    history_b = []

    iteration = 10000

    b = 0.0 
    w = np.zeros(FEATURENUM*TESTHOUR)
    w[57] = 1.0
    lr_b = 0.05
    lr_w = np.full(FEATURENUM*TESTHOUR,0.03)
    r = 2.5

    sigma_b = 0.0
    sigma_w = np.zeros(FEATURENUM*TESTHOUR)

    for i in range(iteration) :

        if i%100 == 0 and i != 0 :
            print ("iter",i,"............done   error=",np.sqrt(error/len(_trainSet)))

        grad_b = 0.0
        grad_w = np.zeros(FEATURENUM*TESTHOUR)
         
        error = 0 

        for n in range(len(_trainSet)) : 
            
            L = trainingSet[n][0] - b - trainingSet[n][1].dot(w)
            error += L**2


            grad_b =  2*r*b - 2*L 
            grad_w =  2*r*w - 2*L*trainingSet[n][1] 

            sigma_b += grad_b**2 
            sigma_w += grad_w**2

            b = b - lr_b/np.sqrt(sigma_b)*grad_b
            w = w - lr_w/np.sqrt(sigma_w)*grad_w

        history_b.append(b)

    print (b) 
    print (w) 
    
    np.save('model.npy',w)

    tEnd = time.time() 

    print ("Run time :" , tEnd-tStart , "second(s)")

    error = 0

    for i in range(len(_testSet)):
        L = _testSet[i][0] - b - _testSet[i][1].dot(w)
        error += L**2
 
    error /= len(_testSet)

    print ("Estimated error:" , error)

    with open(_ansFilename,'w') as ansFile:
        ansFile.write('id,value')
        answer = []
        for i in range(240):
            tmpans = b + testingSet[i].dot(w)
            answer.append(tmpans)
            ansFile.write('\nid_'+str(i)+','+str(tmpans))
