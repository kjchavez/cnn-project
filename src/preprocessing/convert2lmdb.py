# -*- coding: utf-8 -*-
"""
Converts AVI files to fixed length raw pixel data stored in an LMDB database.
Created on Sun Feb 22 12:16:17 2015

@author: Kevin Chavez
"""
import os
import lmdb
import cv2
from cv2.cv import * #Constants of the form CV_*
import argparse
import numpy as np
import itertools
from src.dataio.datum import Datum4D

APPROXIMATE_MEAN = 127.0
MAX_ATTEMPTS = 3

def read_clip(capture, video_filename, num_frames, 
              height=-1, width=-1, start_frame=0, mean_subtract=True,
              subsample=1,num_cuts=1):
    success = capture.open(video_filename)
    
    if not success:
        print "Couldn't open video"
        print "Crashed at index %d." % index
        return None
        
    if width < 0:
        width = capture.get(CV_CAP_PROP_FRAME_WIDTH)
    if height < 0:
        height = capture.get(CV_CAP_PROP_FRAME_HEIGHT)
    
    top = (capture.get(CV_CAP_PROP_FRAME_HEIGHT) - height)/2
    bottom = top + height
    left = (capture.get(CV_CAP_PROP_FRAME_WIDTH) - width)/2
    right = left + width
    
    frame_count = int(capture.get(CV_CAP_PROP_FRAME_COUNT))
    if num_frames < 0:
        num_frames = int(capture.get(CV_CAP_PROP_FRAME_COUNT))

    start_frames = [start_frame + int((frame_count - start_frame - num_frames)
                        / float(num_cuts)*i) for i in range(num_cuts)]
    print "Frame count =", frame_count
    print "Starting at", start_frames

    clips = []
    for n in range(num_cuts):
        clip = np.empty((3,num_frames,height/subsample,width/subsample),
                        dtype=np.int16)
        
        capture.set(CV_CAP_PROP_POS_FRAMES,start_frames[n])
            
        for i in xrange(num_frames):
            frame_available, frame = capture.read()
            if not frame_available:
                print "Ran out of frames when reading", video_filename
                print "Padding with %d empty frames." % (num_frames-i)
                clip[:,i:] = 0
                break
                
            if mean_subtract:
                clip[:,i] = frame[top:bottom:subsample, left:right:subsample, :] \
                                .transpose(2,0,1) - APPROXIMATE_MEAN
            else:
                clip[:,i] = frame[top:bottom:subsample, left:right:subsample, :].transpose(2,0,1)

        clips.append(clip)
        
    capture.release() # Shouldn't have to do this explicitly, but otherwise
                      # it crashes on certain machines after a number of videos
    return clips
    
def create_datum(clip,label):
    """ Creates a Datum structure from a 4D clip and label. """
    datum = Datum4D.array_to_datum(clip,label)
    return datum
    
def write_to_lmdb(env, keys, data):
    """ Adds a 4D video clip to an LMDB database using Caffe's Datum structure.
    
    Args:
        env (lmdb.Environment): database environment
        data ([Datum]): list of datum object to be written to database

    Returns:
        Boolean flag indicating success or failure.
    """
    try:
        with env.begin(write=True) as txn:
            cursor = txn.cursor()
            for key,datum in itertools.izip(keys,data):
                cursor.put(key,datum.value)
        return True
    except:
        return False
        
        
def convert_list(list_file,database_name,root_directory,
                 num_frames=16,width=-1,height=-1,start_frame=0,batch_size=10,
                 map_size=100e6,randomize=False,subsample=1,num_cuts=1):

    capture = cv2.VideoCapture()
    env = lmdb.open(database_name,map_size=map_size,writemap=True)
    data = []
    keys = []
    done = False
    batch_num = 1
    
    curr_key = 0
    with open(list_file) as video_list:
        while not done:
            for k in xrange(batch_size):
                try:
                    line = next(video_list)
                except StopIteration:
                    done = True
                    break
                
                filename, label = line.split()
                print "Processing", filename, "..."                
                
                label = int(label) - 1 # So its zero-indexed
                full_filename = os.path.join(root_directory,filename)                
                clips = read_clip(capture,full_filename,num_frames,height=height,
                                  width=width,start_frame=start_frame,
                                  subsample=subsample,num_cuts=num_cuts)
                datums = [create_datum(clip,label) for clip in clips]
                data += datums

                new_keys = ["%08d_%s" % (k,filename) 
                            for k in range(curr_key,curr_key+num_cuts)]
                keys += new_keys
                curr_key += num_cuts

            for n in range(MAX_ATTEMPTS):
                success = write_to_lmdb(env,keys,data)
                if success:
                    break
            else:
                print "Warning: could not write batch %d to database" % batch_num
            
            data = []
            keys = []
            batch_num += 1
            
    env.close()
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("root",help="Root directory of video files specified in"
                                     " input file")
    parser.add_argument("database_name")
    parser.add_argument("--length", "-l", type=int, default=16,
                        help="Number of frames per video")
    parser.add_argument("--width", "-w", type=int, default=-1, 
                        help="Fixed width to crop to (default is no cropping)")
    parser.add_argument("--height", "-g", type=int, default=-1,
                        help="Fixed height to crop to (default is no cropping)")
    parser.add_argument("--batchsize","-b",type=int,default=10,
                        help="Number of videos to write to lmdb at a time"
                             " (note: all of these will have to be in memory,"
                             " so make sure you have space for it).")
    parser.add_argument("--mapsize","-m",type=int,default=100e6,
                        help="Maximum size of lmdb database")
    parser.add_argument("--subsample", type=int, default=1,
                        help="ratio by which to subsample. Note width/height "
                             "must be divisible by this")
    parser.add_argument("--cuts",'-c',type=int,default=1,
                        help="Number of temporal cuts to take from video. The"
                             " first cut will start from the first frame, then"
                             " staggered, possibly overlapping.")

    args = parser.parse_args()

    if args.width > 0:
        assert args.width % args.subsample == 0
    if args.height > 0:
        assert args.height % args.subsample == 0
    
    convert_list(args.input_file,args.database_name,args.root,
                 num_frames=args.length,width=args.width,height=args.height,
                 start_frame=0,batch_size=args.batchsize,map_size=args.mapsize,
                 num_cuts=args.cuts,subsample=args.subsample)

if __name__ == "__main__":
    main()