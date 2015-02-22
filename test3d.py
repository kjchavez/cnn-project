# -*- coding: utf-8 -*-
"""
3D Conv Test
Created on Sat Feb 21 16:29:21 2015

@author: kevin
"""
import time
import lmdb
from convnet3d import *
import numpy as np
import theano
import theano.tensor as T

dtensor5 = T.TensorType('float64', (False,)*5)

class DataFetcher(object):
    def __init__(self,database_name):
        self.db_name = database_name
        self.env = lmdb.open(database_name)
        
    def load_data(self,batch_size,video_shape,out):
        """Fetches a set of videos from an LMDB database.
        
        Args:
            batch_size - number of videos to load into memory
            video_shape - 3 tuple (frames, height, width) for videos
            out - storage container of shape... 
                    (num videos, frames, channels, height,width)
        """
        TT, HH, WW = video_shape
        with self.env.begin() as txn:
            cursor = txn.cursor()
            it = iter(cursor)
            for n in xrange(batch_size):
                try:
                    out[n] = next(it)
                except:
                    cursor.first() # reset to beginning
                    it = iter(cursor)
                    print "completed epoch"
                    out[n] = next(it)
                
        return out
        
    def __del__(self):
        self.env.close()


def evaluate_3d_conv():
    theano.config.exception_verbosity = "high"
    rng = np.random.RandomState(234)
    TT, HH, WW = 20,60,80
    N = 10
    num_classes = 5
    batch_size = 10
    num_filters = 2
    num_channels = 3
    
    
    X_train = theano.shared(rng.randint(0,255,size=(N,num_channels,TT,HH,WW))
                            .astype(theano.config.floatX),borrow=True)
    y_train = theano.shared(rng.randint(0,num_classes,size=(N,)).astype('int32'))
    print y_train

    params = []

    x = dtensor5('x')
    y = T.ivector('y')

    # Apply padding to x  
    FT, FH, FW = 5, 5, 5
    conv1 = ConvLayer(x,num_channels,num_filters,(FT,FH,FW),(TT,HH,WW),batch_size,relu,
                      layer_name="Conv1")
    params += conv1.params

    pool1 = PoolLayer(conv1.output,(2,2,2))     
                 
    #fc2 = HiddenLayer(conv1.output,T*H*W*2,num_classes,relu)
    out_dim = num_filters * np.prod([TT - FT +1,HH - FH + 1, WW - FW + 1])/8
    out_dim = num_filters*TT*HH*WW/8
    softmax = LogRegr(pool1.output.flatten(ndim=2),out_dim,num_classes,relu,rng)
    params += softmax.params
    
    reg = 0.01
    cost = softmax.negative_log_likelihood(y) + reg*T.sum(softmax.W*softmax.W)
    

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in params]

    learning_rate = 0.001
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
    
    for k in range(1):
        tic = time.time()
        cost = train_model(k)
        toc = time.time()
        print cost, "(%0.4f seconds)" % (toc - tic)
    
                      

if __name__ == "__main__":
    evaluate_3d_conv()
    
                      