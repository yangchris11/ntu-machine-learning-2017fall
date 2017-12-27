import argparse



option_num = 6 
sentences_path = 'data/sentences.pickle'

# word embedding model config
word_embedding_model_dim = 500
word_embedding_model_win = 16
word_embedding_model_min = 3
jieba_dict_path = 'data/dict.txt.big'
word_embedding_model_path = 'model/word_embedding_model.bin'

# jieba-tokenizer config
jieba_tokenizer_path = 'data/jieba_tokenizer.pickle'

# input sequence config 
max_length = 12






pred_file = 'submission/pred'

do_predict = True

