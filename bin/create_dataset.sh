#!/bin/bash

# Split training set into train and validation sets
shuf data/ucfTrainTestlist/trainlist01.txt > shuffled.txt
head -8500 shuffled.txt > data/ucfTrainTestlist/train.txt
tail shuffled.txt --lines=+8501 > data/ucfTrainTestlist/val.txt
rm shuffled.txt

python -m src.preprocessing.convert2lmdb data/ucfTrainTestlist/train.txt data/UCF-101 data/traindb.lmdb --batchsize 25 --mapsize 300000000000

python -m src.preprocessing.convert2lmdb data/ucfTrainTestlist/val.txt data/UCF-101 data/valdb.lmdb --batchsize 25 --mapsize 100000000000
