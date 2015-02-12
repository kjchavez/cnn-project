# -*- coding: utf-8 -*-
"""
Expand training lists to contain all frames of videos or at least a random
smattering of them.
Created on Wed Feb 11 22:46:41 2015

@author: kevin
"""
import os
import glob
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list','-l',
                        default="../../data/ucfTrainTestlist/trainlist01.txt")
    parser.add_argument('--frames-dir','-d',dest="frames_dir",
                        default="../../data/frames")
    parser.add_argument('--output-dir','-o',dest="output_dir",
                        default="../../data/framesTrainTestlist")


    args = parser.parse_args()
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
    output_list_file = open(os.path.join(args.output_dir,
                                         args.list.split("/")[-1]),'w')
    
    with open(args.list) as fp:
        os.chdir(args.frames_dir)
        for line in fp:
            video, label = line.split()
            video = video.split('.')[0]
    
            for filename in glob.glob(os.path.join(video,"*.jpg")):
                print >> output_list_file, filename, label
                
    output_list_file.close()

if __name__ == "__main__":
    main()