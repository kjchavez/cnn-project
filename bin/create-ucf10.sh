#!/bin/bash
python -m src.preprocessing.trainval_split data/ucfTrainTestlist/trainlist01.txt -c 10

# Split training set into train and validation sets
shuf data/ucfTrainTestlist/train.txt --output data/ucfTrainTestlist/train.txt
shuf data/ucfTrainTestlist/val.txt --output data/ucfTrainTestlist/val.txt

python -m src.preprocessing.convert2lmdb data/ucfTrainTestlist/train.txt data/UCF-101 data/train-10class.lmdb --batchsize 25 --mapsize 25000000000 --height 224 --width 224 --subsample 2 --cuts 4

python -m src.preprocessing.convert2lmdb data/ucfTrainTestlist/val.txt data/UCF-101 data/val-10class.lmdb --batchsize 25 --mapsize 7500000000 --height 224 --width 224 --subsample 2 --cuts 4
