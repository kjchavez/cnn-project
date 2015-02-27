# -*- coding: utf-8 -*-
"""
Extract Frames
Created on Wed Feb 11 22:06:20 2015

@author: kevin
"""
import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("root")
parser.add_argument("output")
parser.add_argument("--fps",type=int,default=25)
parser.add_argument("--sublist",default=None)

args = parser.parse_args()

if not os.path.isdir(args.output):
    os.makedirs(args.output)

avconv_log = open('avconv-log.txt','w')
if args.sublist is None:
    classes = os.listdir(args.root)
    num_videos_processed = 0
    for c in classes:
        path = os.path.join(args.root,c)
        output_path = os.path.join(args.output,c)
        os.makedirs(output_path)
        videos = os.listdir(path)
        for video in videos:
            video_path = os.path.join(output_path,video.split(".")[0])
            os.makedirs(video_path)

            input_file = os.path.join(path,video)
            output_file = os.path.join(video_path,video.split(".")[0]+"-%04d.jpg")

            #avconv -i file.avi -f image2 Out%00d.jpg
            subprocess.call(['avconv', '-i', input_file, '-f', 'image2', 
                             '-s', 'qvga', '-r', str(args.fps), output_file],
                             stdout=avconv_log,stderr=avconv_log)

            num_videos_processed += 1
            if num_videos_processed % 100 == 0:
                print "Processed %d videos" % num_videos_processed

else:
    with open(args.sublist) as fp:
        videos = [line.split()[0] for line in fp]

    for v in videos:
        classname,video = v.split('/')
        output_path = os.path.join(args.output,v.split('.')[0])
        file_pattern = video.split('.')[0]+"-%04d.jpg"
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        output_pattern = os.path.join(output_path,file_pattern)
        input_file = os.path.join(args.root,v)
        subprocess.call(['avconv', '-i', input_file, '-f', 'image2', 
                         '-s', 'qvga', '-r', str(args.fps), output_pattern],
                         stdout=avconv_log,stderr=avconv_log)

avconv_log.close()
