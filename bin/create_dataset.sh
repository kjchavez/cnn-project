#!/bin/bash

# Split training set into train and validation sets
shuf data/ucfTrainTestlist/trainlist01.txt > shuffled.txt
head -8000 shuffled.txt > train.txt
tail shuffled.txt --lines=+8001 > val.txt

python -m src.preprocessing.convert2lmdb data/ucfTrainTestlist/train.txt data/UCF-101 data/traindb.lmdb --batchsize 25 --mapsize 300000000000

python -m src.preprocessing.convert2lmdb data/ucfTrainTestlist/val.txt data/UCF-101 data/valdb.lmdb --batchsize 25 --mapsize 150000000000
