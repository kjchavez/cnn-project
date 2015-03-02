shuf data/ucfTrainTestlist/trainlist01.txt > shuffled.txt
head -20 shuffled.txt > data/ucfTrainTestlist/tinytrain.txt
head -30 shuffled.txt | tail -10 > data/ucfTrainTestlist/tinyval.txt
rm shuffled.txt

python -m src.preprocessing.convert2lmdb data/ucfTrainTestlist/tinytrain.txt data/UCF-101 data/tinytraindb.lmdb --randomize --batchsize 25 --mapsize 1000000000
python -m src.preprocessing.convert2lmdb data/ucfTrainTestlist/tinyval.txt data/UCF-101 data/tinyvaldb.lmdb --randomize --batchsize 25 --mapsize 1000000000
