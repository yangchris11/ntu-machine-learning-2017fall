import os 
import sys
import csv
import time
import config
import pickle
import logging
import itertools
import numpy as np
import pandas as pd 
from termcolor import colored,cprint

import gensim
from gensim.models import Word2Vec

import jieba

from scipy import spatial

def cut(s):
    return list(jieba.cut(s))

def process(s):
  rtns = s.split('\t')
  rtns = [cut(rtn[rtn.find(':')+1 : ]) for rtn in rtns]
  return rtns

jieba.set_dictionary(config.jieba_dict_path)
word_embedding_model = Word2Vec.load(config.word_embedding_model_path)

test_num = 5060

compare = []
with open('data/self_ans.csv','r') as f:
    csvf = csv.reader(f)
    next(csvf)
    # for row in itertools.islice(csvf, test_num):
    for row in csvf:
        compare.append(row[1])
f.close()

total_test_case = 5060
with open('data/testing_data.csv', 'r') as f:
  csvf = csv.reader(f)
  next(csvf)
  predict = [] 
  for row in itertools.islice(csvf, test_num):
  #for row in csvf:
    choice_similiar = []

    U = process(row[1].replace(" ",""))
    Us = []
    for i in range(len(U)):
        Us += U[i]
    # print(Us)

    Rs = process(row[2].replace(" ",""))
    # print(Rs)

    for k in range(config.option_num):
        sim = 0 
        for i in range(len(Us)):
            for j in range(len(Rs[k])):
                if Us[i] in word_embedding_model and Rs[k][j] in word_embedding_model :
                    u = word_embedding_model[Us[i]]
                    v = word_embedding_model[[Rs[k][j]]]
                    sim += (1 - spatial.distance.cosine(u,v))
        sim /= (len(Us)*len(Rs[k]))
        choice_similiar.append(sim)
    predict.append(np.argmax(choice_similiar))

    # print('{}/{}'.format(csvf.line_num-1,total_test_case)) 
    # sys.stdout.flush()
f.close()

'''
ct = 0 
for i in range(test_num):
    if int(compare[i]) == predict[i]:
        ct += 1

print("wor_embedding_moedel_dim =",config.word_embedding_model_dim)
print("wor_embedding_moedel_win =",config.word_embedding_model_win)
print("wor_embedding_moedel_min =",config.word_embedding_model_min)
print(colored("{}/{}={}%".format(ct,test_num,ct/test_num),'red'))
'''

if config.do_predict:
    pred_file_name = config.pred_file \
                    + '_{}'.format(config.word_embedding_model_dim) \
                    + '_{}'.format(config.word_embedding_model_win) \
                    + '_{}'.format(config.word_embedding_model_min) \
                    + '.csv' 
    with open(pred_file_name, 'w') as outfile:
        print('id,ans',file=outfile)
        for i in range(len(predict)):
            print('{},{}'.format(i+1, predict[i]),file=outfile)
    print(colored("Predicted testing file to {}".format(pred_file_name),'red'))
