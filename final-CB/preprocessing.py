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

tmp_time = time.time()

print("Sentences number used for training data =",len(sentences))
print("Parsing raw training data cost {} seconds".format(round(tmp_time-start_time,3)))
pickle.dump( sentences, open( config.sentences_path , "wb" ))
print(colored("Saved raw sentences to {}".format(config.sentences_path),'yellow'))


jieba_tokenizer_dict = {}
idx = 1 
for i in range(len(sentences)):
    tmp = [] 
    for j in range(len(sentences[i])):
        if sentences[i][j] not in jieba_tokenizer_dict :
            jieba_tokenizer_dict[sentences[i][j]] = idx
            idx += 1
        tmp.append(jieba_tokenizer_dict[sentences[i][j]])

print("Tokenizer index numbers =",idx)
print("Tokenizing training data cost {} seconds".format(round(time.time()-tmp_time,3)))
pickle.dump( sentences, open( config.jieba_tokenizer_path , "wb" ))
print(colored("Saved jieba-tokenizer to {}".format(config.jieba_tokenizer_path),'yellow'))

tmp_time = time.time()
