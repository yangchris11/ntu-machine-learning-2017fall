#!/bin/bash 
wget -O ./model/checkpoint_whole_model.h5 "https://www.dropbox.com/s/7w0kmk9slbr7h1x/checkpoint_whole_model.h5?dl=1"
python3 hw4_test.py $1
