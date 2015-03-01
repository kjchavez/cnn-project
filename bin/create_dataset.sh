python -m src.preprocessing.convert2lmdb data/ucfTrainTestlist/trainlist01.txt data/UCF-101 data/traindb.lmdb --randomize --batchsize 25 --mapsize 300000000000

python -m src.preprocessing.convert2lmdb data/ucfTrainTestlist/testlist01.txt data/UCF-101 data/testdb.lmdb --randomize --batchsize 25 --mapsize 150000000000
