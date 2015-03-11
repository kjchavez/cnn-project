# -*- coding: utf-8 -*-
"""
Create training and validation splits, making sure to keep groups separate so 
the validation set will still be representative of test set.
Created on Wed Feb 11 22:46:41 2015

@author: Kevin Chavez
"""
import os
import random
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('list',
                        help="Train list given by UCF.")
    parser.add_argument('--num-classes','-c',dest="num_classes",type=int,
                        help="Number of classes to keep in the output lists",
                        default=101)
    parser.add_argument('--output-dir','-o',dest="output_dir",
                        default="data/ucfTrainTestlist")

    args = parser.parse_args()
    
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    

    holdout_groups = random.sample(range(8,26),2)

    train_list = open(os.path.join(args.output_dir,"train.txt"),'w')    
    val_list = open(os.path.join(args.output_dir,"val.txt"),'w')
    with open(args.list) as fp:
        for line in fp:
            filename, label = line.split()
            if int(label) <= args.num_classes:
                group = int(filename[-10:-8])
                if group in holdout_groups:
                    # Write to validation list
                    val_list.write(line)
                else:
                    # Write to train list
                    train_list.write(line)
    
    train_list.close()
    val_list.close()

if __name__ == "__main__":
    main()