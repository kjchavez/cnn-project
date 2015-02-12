# -*- coding: utf-8 -*-
"""
Extract Frames
Created on Wed Feb 11 22:06:20 2015

@author: kevin
"""
import os
import glob
import subprocess

ROOT = "../../data/UCF-101"
OUTPUT_DIR = "../../data/frames-10"
FPS = 10

if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

classes = os.listdir(ROOT)

avconv_log = open('avconv-log.txt','w')
num_videos_processed = 0
for c in classes:
    path = os.path.join(ROOT,c)
    output_path = os.path.join(OUTPUT_DIR,c)
    os.makedirs(output_path)
    videos = os.listdir(path)
    for video in videos:
        video_path = os.path.join(output_path,video.split(".")[0])
        os.makedirs(video_path)

        input_file = os.path.join(path,video)
        output_file = os.path.join(video_path,video.split(".")[0]+"-%03d.jpg")

        #avconv -i file.avi -f image2 Out%00d.jpg
        subprocess.call(['avconv', '-i', input_file, '-f', 'image2', 
                         '-s', 'qvga', '-r', str(FPS), output_file],
                         stdout=avconv_log,stderr=avconv_log)

        num_videos_processed += 1
        if num_videos_processed % 100 == 0:
            print "Processed %d videos" % num_videos_processed

avconv_log.close()

