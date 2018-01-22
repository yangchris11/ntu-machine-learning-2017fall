#!/bin/bash
python pretest.py --dim 200 --jieba ch --model ../data/word_embedding_model_5.bin  --test_file $1 --result_file ../save/pred_1.csv --stopword
python pretest.py --dim 200 --jieba tw --model ../data/word_embedding_model_8.bin  --test_file $1 --result_file ../save/pred_2.csv --stopword --suffix --punctuation
python pretest.py --dim 300 --jieba tw --model ../data/word_embedding_model_9.bin  --test_file $1 --result_file ../save/pred_3.csv --stopword --suffix --punctuation
python pretest.py --dim 32  --jieba tw --model ../data/word_embedding_model_12.bin --test_file $1 --result_file ../save/pred_4.csv
python pretest.py --dim 64  --jieba tw --model ../data/word_embedding_model_13.bin --test_file $1 --result_file ../save/pred_5.csv
python pretest.py --dim 128 --jieba tw --model ../data/word_embedding_model_14.bin --test_file $1 --result_file ../save/pred_6.csv
python vote.py ../save/ $2
