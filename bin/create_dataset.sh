cd data
wget http://crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
unzip UCF101TrainTestSplits-RecognitionTask.zip
rm *.zip

python -m src.preprocessing.convert2lmdb data/ucfTrainTestlist/trainlist01.txt data/UCF-101 data/traindb.lmdb --randomize --batchsize 25 --mapsize 300000000000
