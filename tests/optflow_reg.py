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

theano.config.optimizer='fast_run'
theano.config.exception_verbosity="high"

# Create random set of filters
num_filters = 4
N = num_filters
C, TT, H, W = 3, 5, 7, 7

random_gray = np.repeat(40*np.random.randn(num_filters,1,TT,H,W),3,axis=1)

kernel = theano.shared(
            random_gray.astype(theano.config.floatX),
            borrow=True)

original_frames = [kernel.get_value().copy()[0,:,i].transpose(1,2,0) \
                    + APPROXIMATE_MEAN for i in range(TT)]

            
loss, updates, grad = optical_flow_regularizer(kernel,(N,C,TT,H,W))

lr = 100
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

for n in range(100):
    loss, grad = loss_fn()
    print "Loss:", loss
    print "Norm W:", np.linalg.norm(kernel.get_value())
    
#%%
frames = [kernel.get_value()[0,:,i].transpose(1,2,0) + APPROXIMATE_MEAN for i in range(TT)]
for k in range(100):
    frame = frames[k % len(frames)].sum(axis=2)
    original = original_frames[k % len(frames)].sum(axis=2)    
    cv2.imshow("Regularized",frame.astype('uint8'))
    cv2.imshow("Original",original.astype('uint8'))    
    k = cv2.waitKey(100)
    if k == 1048689: # apparently this is 'q'
        cv2.destroyAllWindows()