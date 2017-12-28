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
jieba.set_dictionary(config.jieba_dict_path)



from scipy import spatial

def cut(s):
    return list(jieba.cut(s))

def process(s):
    rtns = s.split('\t')
    rtns = [cut(rtn[rtn.find(':')+1 : ]) for rtn in rtns]
    return rtns

training_txt_file = ['data/training_data/1.txt',
                        'data/training_data/2.txt',
                        'data/training_data/3.txt',
                        'data/training_data/4.txt',
                        'data/training_data/5.txt']
sentences = []

for i in range(len(training_txt_file)):
    f = open(training_txt_file[i],'r',encoding='utf8')
    for row in f :
        tmp = [ x for x in cut(row) if x != '\n' ]
        sentences.append(tmp)

for i in range(len(sentences)-5):
    sentences[i] += sentences[i+1]
    sentences[i] += sentences[i+2]
    sentences[i] += sentences[i+3]
    sentences[i] += sentences[i+4]

print("Sentences number used to pretrained =",len(sentences))
print(colored("Finish preprocessing".format(),'yellow'))

tmp_time = time.time()

word_embedding_model = Word2Vec(sentences,
				size=config.word_embedding_model_dim,
                window=config.word_embedding_model_win,
				min_count=config.word_embedding_model_min)
word_embedding_model.save('model/word_embedding_model.bin')
print("Training word2Vec word embedding model cost {} seconds".format(round(time.time()-tmp_time,3)))
print(colored("Saved pretrained word-embedding model to {}".format('word_embedding_model.bin'),'yellow'))
