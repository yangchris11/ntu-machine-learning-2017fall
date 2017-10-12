import sys
import csv
import random as rd 
import numpy as np 

# np.set_printoptions(suppress=True)

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

if __name__ == '__main__' : 

    _testingFilename  = sys.argv[1]
    _ansFilename      = sys.argv[2]
    
    testingSet = readTestingData(_testingFilename)

    b = 0.00997487759548
    w = np.load('model.npy')

    with open(_ansFilename,'w') as ansFile:
        ansFile.write('id,value')
        answer = []
        for i in range(240):
            tmpans = b + testingSet[i].dot(w)
            answer.append(tmpans)
            ansFile.write('\nid_'+str(i)+','+str(tmpans))
