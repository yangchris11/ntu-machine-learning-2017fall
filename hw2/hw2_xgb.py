import xgboost as xgb
import sys
import csv
import math
import random as rd
import numpy as np 
import pandas as pd 

np.set_printoptions(suppress=True)

FEATURE_NUM = 107

_rawtrainfileX = sys.argv[1]
_rawtrainfileY = sys.argv[2]
_trainfileX    = sys.argv[3]
_trainfileY    = sys.argv[4]
_testfile      = sys.argv[5]
_ansfile       = sys.argv[6]


def read(_filename):
    data = pd.read_csv(_filename)
    # for column in data:
    #     mean = data[column].mean()
    #     std = data[column].std()
    #     if std != 0:
    #         data[column] = data[column].apply(lambda x:(x-mean)/std)
    return data

# read in data
x = read(_trainfileX)
test_x = read(_testfile)
y = pd.read_csv(_trainfileY,header=0)  
x = np.array(x)
y = np.array(y)
test_x = np.array(test_x)


# train_x = []
# train_y = []
# val_x = []
# val_y = []

# =rd.seed(777)
# for i in range(len(x)):
#     i = rd.randint(1,8)
#     if i=1 :
#         val_x.append(x[i])
#         val_y.append(y[i])
#     else:
#         train_x.append(x[i])
#         train_y.append(y[i])

# print ("Data spilt into train/val : {}/{}".format(len(train_x),len(val_x)))

# x = np.array(train_x)
# y = np.array(train_y)
# val_x = np.array(val_x)
# val_y = np.array(val_y)

dtrain = xgb.DMatrix(x,label=y)
dtest = xgb.DMatrix(test_x)


#specify parameters via map
param = { 'max_depth':5, 'eta':0.1, 'silent':1, 'objective':'binary:logistic' }
num_round = 500
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)

# dval = xgb.DMatrix(val_x)
# val_preds = bst.predict(dval)
# ct = 0
# for n in range(len(val_x)):
#     tmp = val_preds[n]
#     print (tmp)
#     if tmp>0.5 and val_y[n]==1:
#         ct += 1
#     elif tmp<=0.5 and val_y[n]==0:
#         ct += 1
# print ("====================parameters====================")
# print ("iter =",num_round)
# print ("lr =",param['eta'])
# print ("max_depth=",param['max_depth'])
# print ("Acc =",100*ct/len(val_x),"%",ct,len(val_x),len(val_y))





ans = []
a1 = 0
a0 = 0

for n in range(len(test_x)):
    ans.append([str(n+1)])
    tmp = preds[n]
    if tmp>0.5 :
        ans[n].append(int(1))
        a1 += 1
    else:
        ans[n].append(int(0))
        a0 += 1

text = open(_ansfile, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()

print ("0 =",a0,"1 =",a1)


