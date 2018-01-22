# TV  Conversations Prediction (ML2017 Final Project)

## Team Member

* b03901086 電機四 楊正彥
* b03901124 電機四 李昂軒
* b03901130 電機四 林奕廷
* b03901145 電機四 郭恆成


## Required Toolkits

* python 3.5+
* keras 2.0.8
* numpy 1.13.3
* gensim 3.1.0
* jieba 0.39
* scipy 0.19.1
* termcolor 1.1.0

## File Description

* `pretrain.py` is used for training word embedding model.
* `pretest.py` is used for predicting testing file.
* `vote.py` is used for models ensembling.
* `test.sh` is used for reproducing best score on kaggle.

## How to Reproduce

1. Run `mkdir data/training_data` and download the training file into the `data/training_data/`
2. Download the testing file into the `data/` and rename it into `testing_data.csv`
3. Run `bash test.sh ../data/testing_data.csv ../save/pred.csv` under `src/` directory.
4. `save/pred.csv` is the reproducing prediction file.

## Directory 

```
   root
   +-- src
   |   +-- train.sh
   |   +-- test.sh
   |   +-- pretrain.py
   |   +-- pretest.py
   |   +-- vote.py
   +-- data
   |   +-- training_data
   |   |    +-- train_1.txt (to be placed) 
   |   |    +-- train_2.txt (to be placed) 
   |   |    +-- train_3.txt (to be placed) 
   |   |    +-- train_4.txt (to be placed) 
   |   |    +-- train_5.txt (to be placed) 
   |   +-- testing_data.csv (to be placed)
   |   +-- dic.txt.big
   |   +-- word_embedding_model.bin
   +-- save
   +-- Report.pdf
   +-- README.md
```
