python pretrain.py --dim 200 --mincnt 3 --negative 5 --word2vec_win 5  --sentence_win 5 --jieba ch --train_dir $1 --model ../data/word_embedding_model_5.bin  --stopword
python pretrain.py --dim 200 --mincnt 3 --negative 1 --word2vec_win 5  --sentence_win 5 --jieba tw --train_dir $1 --model ../data/word_embedding_model_8.bin  --stopword --suffix --punctuation
python pretrain.py --dim 300 --mincnt 2 --negative 5 --word2vec_win 5  --sentence_win 3 --jieba tw --train_dir $1 --model ../data/word_embedding_model_9.bin  --stopword --suffix --punctuation
python pretrain.py --dim 32  --mincnt 1 --negative 7 --word2vec_win 20 --sentence_win 3 --jieba tw --train_dir $1 --model ../data/word_embedding_model_12.bin
python pretrain.py --dim 64  --mincnt 1 --negative 7 --word2vec_win 20 --sentence_win 3 --jieba tw --train_dir $1 --model ../data/word_embedding_model_13.bin
python pretrain.py --dim 128 --mincnt 1 --negative 7 --word2vec_win 20 --sentence_win 3 --jieba tw --train_dir $1 --model ../data/word_embedding_model_14.bin
