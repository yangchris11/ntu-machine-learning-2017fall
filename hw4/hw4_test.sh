#!/bin/bash 
wget -O ./model/checkpoint_whole_model.h5 "https://www.dropbox.com/s/sh9khpn3v6zhtny/hw4_checkpoint_whole_model.h5?dl=1"
python3 hw4_test.py $1
