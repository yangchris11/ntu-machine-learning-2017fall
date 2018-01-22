import re
import csv
import argparse
import numpy as np
from termcolor import colored

import gensim
from gensim.models import Word2Vec

import jieba

from scipy import spatial

parser = argparse.ArgumentParser(prog='pretest.py', description='ML2017-final TV-conversation')
parser.add_argument('--suffix', action='store_true', default=False)
parser.add_argument('--stopword', action='store_true', default=False)
parser.add_argument('--punctuation', action='store_true', default=False)
parser.add_argument('--dim', type=int, default='200')
parser.add_argument('--jieba', type=str, default='tw')
parser.add_argument('--jieba_dict', type=str, default='../data/dict.txt.big')
parser.add_argument('--model', type=str, default='../data/word_embedding_model.bin')
parser.add_argument('--test_file', type=str, default='../data/testing_data.csv')
parser.add_argument('--result_file', type=str, default=None)
args = parser.parse_args()

def cut(s):
    if args.stopword:
        s = re.sub(r'的', '', s)
        s = re.sub(r'了', '', s)
        s = re.sub(r'就', '', s)
        s = re.sub(r'都', '', s)
        s = re.sub(r'著', '', s)
        
    if args.suffix:    
        s = re.sub(r'阿', '', s)
        s = re.sub(r'啊', '', s)
        s = re.sub(r'巴', '', s)
        s = re.sub(r'吧', '', s)
        s = re.sub(r'嗎', '', s)
        s = re.sub(r'呀', '', s)
        s = re.sub(r'呦', '', s)
        s = re.sub(r'嗚', '', s)
   
    if args.punctuation:
        s = re.sub(r' ', '', s)
        s = re.sub(r'!', '', s)
        s = re.sub(r'\.', '', s)
        s = re.sub(r'\?', '', s)
        s = re.sub(r'，', '', s)
    
    s = s.strip()
    return list(jieba.cut(s))

def process(s):
  rtns = s.split('\t')
  rtns = [cut(rtn[rtn.find(':')+1 : ]) for rtn in rtns]
  return rtns

if args.jieba == 'tw':
    jieba.set_dictionary(args.jieba_dict)
word_embedding_model = Word2Vec.load(args.model)


with open(args.test_file, 'r') as f:
  csvf = csv.reader(f)
  header = next(csvf)
  predict = []
  for row in csvf:
    choice_similiar = []

    U = process(row[1])
    Us = []
    for i in range(len(U)):
        Us += U[i]
    Rs = process(row[2])
        
    
    q = np.zeros(args.dim)
    qct = 0 
    for i in range(len(Us)):
        if Us[i] in word_embedding_model:
            q += word_embedding_model[Us[i]]
            qct += 1
    if qct != 0:
        q /= qct
    preds = []
    for k in range(6):
        sim = 0
        a = np.zeros(args.dim)
        act = 0
        for i in range(len(Rs[k])):
            if Rs[k][i] in word_embedding_model :
                a += word_embedding_model[Rs[k][i]]
                act += 1
        if act != 0:
            a /= act
        if q.all() != np.zeros(args.dim).all() and a.all() != np.zeros(args.dim).all():
            sim = spatial.distance.cosine(q,a)
        preds.append(sim)
    predict.append(np.argmin(preds)) 
f.close()

with open(args.result_file, 'w') as outfile:
    print('id,ans',file=outfile)
    for i in range(len(predict)):
        print('{},{}'.format(i+1, predict[i]),file=outfile)
print(colored("Predicted testing file to {}".format(args.result_file),'red'))
