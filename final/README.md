# TV  Conversations Prediction (ML2017 Final Project)

## Task Description

Given lines of “murmur of one person” or “conversation between multiple people”,  select a sentence that is most likely to appear following the lines.

## File Description
- config.py : configuration file for `pretrain.py` and `pretest.py`
- pretrain.py : generate files for training (word-embedding model) 
- pretest.py : generate prediction based on word-embedding model
- pre.sh : run pretraining and pretesting
- preprocessing.py : gernerate files for `train.py` including `data/sentences.pickle`, `data/jieba_tokenizer.pickle`


## Directory 

```
   root
   +-- src
   |   +-- pretrain.py
   |   +-- pretest.py
   |   +-- train.sh
   |   +-- test.sh
   |   +-- pretrain.py
   +-- data
   |   +-- training_data
   |   |   +-- 1.txt
   |   |   +-- 2.txt
   |   |   +-- 3.txt
   |   |   +-- 4.txt
   |   |   +-- 5.txt   
   |   +-- testing_data.csv
   |   +-- dic.txt.big
   |   +-- sentences.pickle
   |   +-- jieba_tokenizer.pickle
   +-- submission
   +--
```

## Reference

- Mueller, J and Thyagarajan, A. Siamese Recurrent Architectures for Learning Sentence Similarity. Proceedings of the 30th AAAI Conference on Artificial Intelligence (AAAI 2016).
http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12195
