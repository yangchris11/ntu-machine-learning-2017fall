import sys
import csv
import math
import random as rd
import numpy as np 
import pandas as pd 
import xgboost as xgb
from sklearn.cross_validation import train_test_split

np.set_printoptions(suppress=True)

FEATURE_NUM = 107

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

dtrain = xgb.DMatrix(x,y)
watchlist = [(dtrain,'train')]
params = {
    'booster': 'gbtree',
    'max_depth':4, 
    'min_child_weight':1,
    'gamma':0.6,
    'eta':0.1, 
    'subsample': 1,
    'colsample_bytree': 0.9,
    'silent':1, 
    'eval_metric': 'error',
    'objective':'binary:logistic'
}
num_round = 360
clf = xgb.train(params,dtrain,num_round,watchlist,early_stopping_rounds=200,verbose_eval=50)
preds = clf.predict(xgb.DMatrix(test_x))

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