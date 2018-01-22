import os
import re
import time
import argparse
import numpy as np
from termcolor import colored

import gensim
from gensim.models import Word2Vec

import jieba

parser = argparse.ArgumentParser(prog='pretrain.py', description='ML2017-final TV-conversation')
parser.add_argument('--suffix', action='store_true', default=False)
parser.add_argument('--stopword', action='store_true', default=False)
parser.add_argument('--punctuation', action='store_true', default=False)
parser.add_argument('--dim', type=int, default='200')
parser.add_argument('--mincnt', type=int, default='3')
parser.add_argument('--negative', type=int, default='5')
parser.add_argument('--word2vec_win', type=int, default='5')
parser.add_argument('--sentence_win', type=int, default='5')
parser.add_argument('--jieba', type=str, default='tw')
parser.add_argument('--jieba_dict', type=str, default='../data/dict.txt.big')
parser.add_argument('--model', type=str, default='../data/word_embedding_model.bin')
parser.add_argument('--train_dir', type=str, default='../data/training_data/')
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
    
if args.jieba == 'tw':
    jieba.set_dictionary(args.jieba_dict)

training_txt_file = [os.path.join(args.train_dir, f) for f in sorted(os.listdir(args.train_dir))]

sentences = []
for i in range(len(training_txt_file)):
    f = open(training_txt_file[i], 'r', encoding='utf8')
    for row in f :
        tmp = [ x for x in cut(row) ]
        sentences.append(tmp)

for i in range(len(sentences)-args.sentence_win):
    for j in range(1, args.sentence_win):
        sentences[i] += sentences[i+j]

print("Sentences number used to pretrained =",len(sentences))
print(colored("Finish preprocessing".format(),'yellow'))

tStart = time.time()

word_embedding_model = Word2Vec(sentences, sg=1, seed=1, negative=args.negative,
				size=args.dim, window=args.word2vec_win, min_count=args.mincnt)
word_embedding_model.save(args.model)

tEnd = time.time()
print("Training word2Vec word embedding model cost {} seconds".format(round(tEnd-tStart,3)))
print(colored("Saved pretrained word-embedding model to {}".format(args.model),'yellow'))
