# Conversations Prediction (ML2017 Final Project)

## Task Description

Given lines of “murmur of one person” or “conversation between multiple people”,  select a sentence that is most likely to appear following the lines.

## File Description
- config.py : configuration file for `pretrain.py` and `pretest.py`
- pretrain.py : generate files for training (word-embedding model) 
- pretest.py : generate prediction based on word-embedding model
- pre.sh : run pretraining and pretesting

## Directory 

```
   root
   +-- config.py
   +-- pretrain.py
   +-- pretest.py
   +-- pre.sh
   +-- model
   |   +-- word_embedding_model_bin
   +-- data
   |   +-- training_data
   |   |   +-- 1.txt
   |   |   +-- 2.txt
   |   |   +-- 3.txt
   |   |   +-- 4.txt
   |   |   +-- 5.txt   
   |   +-- testing_data.csv
   |   +-- dic.txt.big
   +-- submission
```

## Log 
