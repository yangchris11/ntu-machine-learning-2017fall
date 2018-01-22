import os
import sys
import csv
import numpy as np
from keras.utils import np_utils


filenames = [ os.path.join(sys.argv[1], f) for f in os.listdir(sys.argv[1]) if f.startswith('pred') ]

answers = []
for filename in filenames:
    f = open(filename, 'r')
    answer = np.asarray([row[1] for i, row in enumerate(csv.reader(f, delimiter=',')) if i != 0], dtype='int')
    answer = np_utils.to_categorical(answer, 6)
    answers.append(answer)
pred = np.sum(answers ,axis=0)
pred = np.argmax(pred, axis=1)

outfile = open(sys.argv[2], 'w')
print('id,ans', file=outfile)
for i, p in enumerate(pred):
    print(i+1, p, sep=',', file=outfile)
