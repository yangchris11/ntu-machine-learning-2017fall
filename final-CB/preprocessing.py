import os 
import sys
import csv
import config
import pickle
import logging
import numpy as np
import pandas as pd 
from termcolor import colored,cprint

import time

import gensim
from gensim.models import Word2Vec

import jieba
jieba.set_dictionary(config.jieba_dict_path)

def cut(s):
    return list(jieba.cut(s))

def process(s):
    rtns = s.split('\t')
    rtns = [cut(rtn[rtn.find(':')+1 : ]) for rtn in rtns]
    return rtns

start_time = time.time()

training_txt_file = ['data/training_data/1.txt',
                        'data/training_data/2.txt',
                        'data/training_data/3.txt',
                        'data/training_data/4.txt',
                        'data/training_data/5.txt',]
sentences = []

for i in range(len(training_txt_file)):
    f = open(training_txt_file[i],'r',encoding='utf8')
    for row in f :
        tmp = [ x for x in cut(row) if x != '\n' ]
        sentences.append(tmp)

print("Sentences number used for training data =",len(sentences))
print("Parsing raw training data cost {} seconds".format(round(time.time()-start_time,3)))

pickle.dump( sentences, open( "data/sentencess.pickle", "wb" ))
print(colored("Saved raw sentences to {}".format('data/sentencess.pickle'),'yellow'))

