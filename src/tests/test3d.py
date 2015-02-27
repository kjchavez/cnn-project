# -*- coding: utf-8 -*-
"""
3D Conv Test
Created on Sat Feb 21 16:29:21 2015

@author: kevin
"""
import sys
import time
import numpy as np
import theano
import theano.tensor as T

from src.convnet3d import *
from src.dataio.fetcher import DataFetcher

dtensor5 = T.TensorType('float32', (False,)*5)

def evaluate_3d_conv():
    theano.config.exception_verbosity = "high"
    theano.config.optimizer = 'None'
    rng = np.random.RandomState(234)
    TT, HH, WW = 16,240,320
    N = 10
    num_classes = 5
    batch_size = 1
    num_filters = 4
    num_channels = 3
    
    if len(sys.argv) > 1:
        fetcher = DataFetcher("data/tinyvideodb.lmdb")
        X, y = fetcher.load_data(10,(16,240,320))
        y /= 21
    else:
        X = np.random.randint(-127,127,size=(N,3,16,240,320)).astype(theano.config.floatX)
        y = np.random.randint(0,num_classes,size=(N,))
        
    X_train = theano.shared(X.astype('float32'), borrow=True)
    y_train = theano.shared(y.astype('int32'), borrow=True)
    print y_train.get_value()
    
    params = []

    x = dtensor5('x')
    y = T.ivector('y')
    FT, FH, FW = 5, 5, 5

    ###########################################################################
    # CONV-RELU-POOL (Layer 1)
    ###########################################################################
    conv1 = ConvLayer(x,num_channels,num_filters,(FT,FH,FW),(TT,HH,WW),batch_size,relu,
                      layer_name="Conv1")
    params += conv1.params
    pool1 = PoolLayer(conv1.output,(2,2,2))

    ###########################################################################
    # CONV-RELU-POOL (Layer 2)
    ###########################################################################
    conv2 = ConvLayer(pool1.output,num_filters,num_filters,
                      (FT,FH,FW),
                      (TT/2,HH/2,WW/2),
                      batch_size,
                      relu,
                      layer_name="Conv2")
    params += conv2.params    
    pool2 = PoolLayer(conv2.output,(2,2,2))

    ###########################################################################
    # FULLY-CONNECTED (Layer 3)
    ###########################################################################
    out_dim = num_filters*TT*HH*WW/64            
    num_hidden = 64
    fc3 = HiddenLayer(pool2.output.flatten(ndim=2),out_dim,num_hidden,relu)
    params += fc3.params    
    
    ###########################################################################
    # SOFTMAX (Layer 4)
    ###########################################################################
    softmax = LogRegr(fc3.output,num_hidden,num_classes,relu,rng)
    params += softmax.params
    
    reg = 0.01
    cost = softmax.negative_log_likelihood(y) + reg*T.sum(softmax.W*softmax.W)
    

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in params]

    learning_rate = 1e-5
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(params, gparams)
    ]    
    
    index = T.lscalar()
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: X_train[index * batch_size: (index + 1) * batch_size],
            y: y_train[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    for k in range(10):
        tic = time.time()
        cost = train_model(k % (N/batch_size))
        toc = time.time()
        print cost, "(%0.4f seconds)" % (toc - tic)
    
                      

if __name__ == "__main__":
    evaluate_3d_conv()
    
                      