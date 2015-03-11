#!/bin/bash
head -985 data/ucfTrainTestlist/trainlist01.txt > data/ucfTrainTestlist/train10class.txt
head -389 data/ucfTrainTestlist/testlist01.txt > data/ucfTrainTestlist/test10class.txt

# Split training set into train and validation sets
shuf data/ucfTrainTestlist/train10class.txt > shuffled.txt
head -850 shuffled.txt > data/ucfTrainTestlist/train.txt
tail shuffled.txt --lines=+851 > data/ucfTrainTestlist/val.txt
rm shuffled.txt

python -m src.preprocessing.convert2lmdb data/ucfTrainTestlist/train.txt data/UCF-101 data/train-10class.lmdb --batchsize 25 --mapsize 25000000000 --height 224 --width 224 --subsample 2 --cuts 4

python -m src.preprocessing.convert2lmdb data/ucfTrainTestlist/val.txt data/UCF-101 data/val-10class.lmdb --batchsize 25 --mapsize 7500000000 --height 224 --width 224 --subsample 2 --cuts 4
