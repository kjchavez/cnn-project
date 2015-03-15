# -*- coding: utf-8 -*-
"""
Hacky way to compute training accuracies at each of the saved snapshots.
Created on Sun Mar 15 13:08:01 2015

@author: Kevin Chavez
"""
import os
import shutil
import argparse
from train import train

parser = argparse.ArgumentParser()
parser.add_argument("directory")

args = parser.parse_args()

with open(os.path.join(args.directory,"architecture.txt")) as fp:
    test_arch = open(os.path.join(args.directory,"test-architecture.txt"),'w')  
    for line in fp:
        line = line.strip()
        if not line:
            continue
        left, right = line.split(":")
        if left == "train":
            traindb = right
            print >> test_arch, left + ":" + right
        elif left == "val":
            print >> test_arch, left + ":" + traindb
        else:
            print >> test_arch, line
            
    test_arch.close()

trial_id = int(args.directory.rsplit('-',1)[1]) 
print trial_id
snaps = os.listdir(os.path.join(args.directory,'snapshots'))
snapshots = [sorted([int(x.rsplit('-',1)[1]) for x in snaps])[-1]]

# Move validation history to temporary file
shutil.move(os.path.join(args.directory,'validation-history.txt'),
            os.path.join(args.directory,'validation-history.txt.tmp'))

accuracies = []
for snapshot in snapshots:
    acc, _ = train(os.path.join(args.directory,"test-architecture.txt"),trial_id,
                   resume=snapshot, validate_rate=1, num_iter=1, lr=0.0)
    os.remove(os.path.join(args.directory,'validation-history.txt'))
    accuracies.append(acc)
         
shutil.move(os.path.join(args.directory,'validation-history.txt.tmp'),
            os.path.join(args.directory,'validation-history.txt'))

with open(os.path.join(args.directory,"training-accuracy.txt"),'w') as fp:
    for iteration, accuracy in zip(snapshots,accuracies):
        print >> fp, iteration, accuracy
        
print snapshots
print accuracies