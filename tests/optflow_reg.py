# -*- coding: utf-8 -*-
"""
Optical Flow Regularization Test
Created on Sun Feb 22 18:39:02 2015

@author: kevin
"""
from src.convnet3d.convnet3d import optical_flow_regularizer
import theano
import theano.tensor as T
import numpy as np
import time

import cv2
from src.constants import *
APPROXIMATE_MEAN = 127

theano.config.optimizer='fast_run'
theano.config.exception_verbosity="high"


def bgr_to_gray(X):
    gray = 0.2989*X[:,:,2] + 0.5870*X[:,:,1] + 0.1140*X[:,:,0]
    return gray
    
def clip(X,low,high):
    X = np.minimum(X,high)
    X = np.maximum(X,low)
    return X
    
def play_clip(frames,gray=False):
    """ 
    frames has shape (C, TT, HH, WW)
    """
    frames = [frames[:,i].transpose(1,2,0)
              for i in range(frames.shape[1])]
    for k in range(100):
        if not gray:
            frame = frames[k % len(frames)]
        else:
            frame = bgr_to_gray(frames[k % len(frames)])
            
        cv2.imshow("Window",frame.astype('uint8'))    
        k = cv2.waitKey(400)
        if k == 1048689: # apparently this is 'q'
            cv2.destroyAllWindows()


def display(filt, Vx, Vy):
    plt.figure()
    TT = filt.shape[1]
    HH, WW = filt.shape[2:]
    frames = [filt[:,i].transpose(1,2,0) + APPROXIMATE_MEAN for i in xrange(filt.shape[1])]
    for t, frame in zip(range(TT),frames):
        frame = clip(frame,0,255)
        if Vx is not None and Vy is not None:
            plt.subplot(2,TT,TT+t+1)
            I,J = np.meshgrid(range(HH),range(WW),indexing='ij')
    
            #print np.mean(Vx), np.mean(Vy)
            #testy = np.zeros_like(Vy[t])
            #testy[0,:] = 1 
            plt.streamplot(I[:,0], -J[0], Vx[t], -Vy[t])            
            plt.axis('image')
            plt.tick_params(\
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')
            plt.tick_params(\
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                left='off',      # ticks along the bottom edge are off
                right='off',         # ticks along the top edge are off
                labelleft='off')
        plt.subplot(2,TT,t+1)
        plt.imshow(frame.astype('uint8'),interpolation='none')
    
    plt.show()


def test_random_filter():
    # Create random set of filters
    num_filters = 1
    N = num_filters
    C, TT, H, W = 3, 9, 11, 11
    random_gray = np.repeat(40*np.random.randn(num_filters,1,TT,H,W),3,axis=1)
    kernel = theano.shared(
                random_gray.astype(theano.config.floatX),
                borrow=True)
    
    original_frames = [kernel.get_value().copy()[0,:,i].transpose(1,2,0) \
                        + APPROXIMATE_MEAN for i in range(TT)]
    
    loss, updates, grad = optical_flow_regularizer(kernel,(N,C,TT,H,W))
    
    lr = 0.01
    loss_fn = theano.function(
                inputs=[],
                outputs=[loss,grad],
                updates=updates+[(kernel, kernel - lr*grad)])
    
    #loss_fn2 = theano.function(
    #            inputs=[],
    #            outputs=[loss, grad2])
                
    tic = time.time()
    l, g = loss_fn()
    toc = time.time()
    print "Evaluated loss and gradient in %0.3f milliseconds" % (1000*(toc-tic))
    
    
    ## Gradient descent on kernel
    for n in range(40):
        loss, grad = loss_fn()
        print "Loss:", loss
        print "Norm grad:", np.linalg.norm(grad)
        print "Norm W:", np.linalg.norm(kernel.get_value())
        
        
    frames = [kernel.get_value()[0,:,i].transpose(1,2,0) + APPROXIMATE_MEAN for i in range(TT)]
    for k in range(100):
        frame = bgr_to_gray(frames[k % len(frames)])
        original = bgr_to_gray(original_frames[k % len(frames)])   
        diff = frame - original + APPROXIMATE_MEAN    
        cv2.imshow("Regularized",frame.astype('uint8'))
        cv2.imshow("Original",original.astype('uint8'))    
        cv2.imshow("Difference",diff.astype('uint8'))    
        k = cv2.waitKey(400)
        if k == 1048689: # apparently this is 'q'
            cv2.destroyAllWindows()
            
                # Show in plot
    import matplotlib.pyplot as plt
    for k,original, frame in zip(range(1,TT+1),original_frames,frames):
        frame = bgr_to_gray(frame).astype('uint8')
        original = bgr_to_gray(original).astype('uint8')
        print np.mean(frame)
        print np.mean(original)
        plt.subplot(2,TT,k)
        plt.imshow(original,cmap='gray',vmin=0,vmax=255,interpolation='none')
        plt.subplot(2,TT,TT+k)
        plt.imshow(frame,cmap='gray',vmin=0,vmax=255,interpolation='none')
    
    plt.show()


#%% Part 2: Hand-crafted filters
# -------------------------------
def moving_edge_test():
    plt.close('all')
    
    # Create random set of filters
    num_filters = 1
    N = num_filters
    C, TT, H, W = 3, 9, 11, 11
    
    # Create filter that is 'natural' looking: an edge entering the screen
    moving_edge = 0*np.ones((num_filters,1,TT,H,W))
    for t in range(TT):
        moving_edge[:,0,t,t:t+1,:] = 100
        
        
    moving_edge = np.repeat(moving_edge,3,axis=1)
    noise = 5*np.random.randn(*moving_edge.shape)
    moving_edge += noise
    original = np.copy(moving_edge)
    display(original[0],None,None)

    kernel = theano.shared(
                moving_edge.astype(theano.config.floatX),
                borrow=True)
    
    #play_clip(moving_edge[0])
    loss, updates, grad, Vx, Vy = \
        optical_flow_regularizer(kernel,(N,C,TT,H,W), gamma=3.)
        
    lr = 0.5
    train = theano.function(
                inputs=[],
                outputs=[loss,grad, Vx, Vy],
                updates=updates+[(kernel, kernel - lr*grad)])
                
    for k in xrange(20):
        l, g, vx, vy = train()
        print "loss %0.2f, ||grad|| = %0.3f, ||W|| = %0.3f" % \
                (l, np.linalg.norm(g),np.linalg.norm(kernel.get_value()))
        
    display(kernel.get_value()[0],vx[0], vy[0])
    #play_clip(original[0] + 127,gray=True)    
    #play_clip(kernel.get_value()[0] + 127,gray=True)

if __name__ == "__main__":
    moving_edge_test()