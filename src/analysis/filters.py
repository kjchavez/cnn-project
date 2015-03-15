# -*- coding: utf-8 -*-
"""
CS 231N Project:
Analysis toolbox for interpreting output of training
Created on Mon Mar  2 07:33:06 2015

@author: Kevin Chavez
"""
import os
import glob
import cPickle
import shutil
import subprocess
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import cv2
from cv2.cv import * # constants
from src.constants import *

from src.preprocessing.convert2lmdb import read_clip

def load_first_layer_params(directory,iteration='best'):
    param_names = []
    params = []
    
    with open(os.path.join(directory,"parameter-names.txt")) as fp:
        param_names.append(next(fp).rstrip())
        param_names.append(next(fp).rstrip())
        
    if iteration == "best":
        model_file = os.path.join(directory,"best-model")
    else:
        model_file = glob.glob(os.path.join(directory,"*.iter-%06d" % iteration))[0]
    
    with open(model_file,'rb') as fp:
        params.append(cPickle.load(fp))
        params.append(cPickle.load(fp))
        
    assert param_names[0][-1] == "W"
    assert param_names[1][-1] == "b"
    W = params[0]
    b = params[1]

    return W, b    

def load_clips(file_list,mean_subtract=False):
    capture = cv2.VideoCapture()
    clips = []
    for filename in file_list:
        clip = read_clip(capture, filename, -1,
                          start_frame=0, mean_subtract=mean_subtract,
                          subsample=2,height=224,width=224)[0]
        clips.append(clip.transpose(1,2,3,0))
        
    return clips

def play_clip(clip,delay=24,window="window",loops=np.inf):
    """ Play a clip for |loops| times. Assumes clip is in proper data format.
    """
    i = 0
    while i < clip.shape[0]*loops:
        frame = clip[i % clip.shape[0]]     
        cv2.imshow(window,frame)
        k = cv2.waitKey(delay)
        i += 1
        if k == 1048689: # this is the 'q' key
            cv2.destroyAllWindows()
            
            
def plot_val_accuracy(directory):
    filename = os.path.join(directory,"validation-history.txt")
    iterations = []
    accuracies = []
    with open(filename) as fp:
        for line in fp:
            iteration, accuracy = line.split()
            iterations.append(int(iteration))
            accuracies.append(float(accuracy))
            
    plt.plot(iterations,accuracies)
    plt.title(directory + " Validation Accuracy History")
    plt.xlabel("iteration")
    plt.ylabel("accuracy")

def visualize_filters(directory,savepath="figs",mean=0.3,std=0.1):
    W, b = load_first_layer_params(directory)
    TT = W.shape[2]
    mu = np.mean(W)
    sigma = np.std(W)
        
    # Apply a linear stretch and clip to visualize nicely
    W = (W - mu)/sigma * std + mean
    W = np.minimum(np.maximum(W,0),1.0)
    
    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    for i in xrange(W.shape[0]):
        Wi = W[i]
        Wi = Wi.transpose(1,2,3,0)
        Wi = Wi[:,:,:,[2,1,0]] # bgr to rgb
        fig = plt.figure()
        for t in xrange(TT):
            plt.subplot(1,TT,t+1)
            plt.imshow(Wi[t])#,interpolation='none')
            plt.axis('off')
            
        fig.suptitle("Filter #%d Weights" % i)
        plt.savefig(os.path.join(savepath,'filter-%03d.png' % i))
        plt.close(fig)
            
def apply_filters(directory,videos,output_filename,iteration="best",level=0,num_loops=5):
    if level > 0:
        raise NotImplementedError("Sorry, not implemented yet for higher level"
                                  " filters yet.")
    param_names = []
    params = []
    
    with open(os.path.join(directory,"parameter-names.txt")) as fp:
        param_names.append(next(fp).rstrip())
        param_names.append(next(fp).rstrip())
        
    if iteration == "best":
        model_file = os.path.join(directory,"best-model")
    else:
        model_file = glob.glob(os.path.join(directory,"*.iter-%06d" % iteration))[0]
    
    with open(model_file,'rb') as fp:
        params.append(cPickle.load(fp))
        params.append(cPickle.load(fp))
        
    assert param_names[0][-1] == "W"
    W = params[0]
    b = params[1]
    
    # Load sample videos
    clips = load_clips(videos)
    clip = clips[0].astype(np.float64)
    filters = W   
    
    activations = []
    for idx in xrange(filters.shape[0]):
        print "-"*20
        print "Filter #%d activation" % idx
        print "-"*20
        W = filters[idx]
        # We do convolution with scipy here, but its exactly the same as during
        # training. In particular: stride = 1, and kernel is flipped, just like
        # conv2d3d's implementation of conv3d
        activation = sum(scipy.ndimage.filters.convolve(
                            clip[:,:,:,ch],W[ch],mode='constant')
                         for ch in range(3)) + b[idx]
                             
        # Apply relu
        activation = np.maximum(activation,0)/2

        plt.hist(activation.ravel(),bins=50)
            
        print "Max:", np.max(activation)
        print "Mean:", np.mean(activation)
        print "Std:", np.std(activation)        
        
        # For display, we'll also clip at 255
        activation = np.minimum(activation,255)
        activations.append(activation.astype(np.uint8))

    #plt.show()  

    # Play all activation videos on big screen
    TT,HH,WW = clip.shape[0:3]
    N = int(np.ceil(np.sqrt(filters.shape[0])))
    frames = np.zeros((TT,(WW+5)*N,(HH+5)*N),dtype=np.uint8)
    for i in range(N):
        for j in range(N):
            if (i*N + j) < len(activations):
                frames[:,i*(HH+5):i*(HH+5)+HH,j*(HH+5):j*(WW+5)+WW] = \
                    activations[i*N + j]
      
    folder = "video"
    shutil.rmtree(folder)
    os.makedirs(folder)
    for i in range(frames.shape[0]):
        filename = os.path.join(folder,'frame-%04d.png' % i)
        cv2.imwrite(filename,frames[i])
        
    subprocess.call(['avconv', '-framerate', '25', '-f', 'image2',
                     '-i', os.path.join(folder,'frame-%04d.png'), 
                     '-c:v', 'h264', '-crf', '1', output_filename])
    #play_clip(frames,loops=num_loops)
            
def test():
    plt.close('all')
    #visualize_filters("../important-data/baby-regnet-0013")
    #plot_val_accuracy("../tmp/project-data/baby-regnet-0002")
    samples = [
        "ApplyLipstick/v_ApplyLipstick_g08_c01",
        "BalanceBeam/v_BalanceBeam_g17_c01",
        "BasketballDunk/v_BasketballDunk_g18_c05",
        "BabyCrawling/v_BabyCrawling_g16_c05",
        "Archery/v_Archery_g11_c04",
        "BaseballPitch/v_BaseballPitch_g22_c04",
        "BenchPress/v_BenchPress_g19_c01",
        "ApplyEyeMakeup/v_ApplyEyeMakeup_g12_c02",
        "BandMarching/v_BandMarching_g14_c04",
        "Basketball/v_Basketball_g10_c04"]
    
    if not os.path.isdir("video-samples"):
        os.makedirs("video-samples")
    
    for s in samples:
        apply_filters("../important-data/baby-regnet-0013",
                      ["data/UCF-101/%s.avi" % s],
                      "video-samples/%s.mp4" % s.split('/')[1])
    

if __name__ == "__main__":
    test()
        