# -*- coding: utf-8 -*-
"""
CNN3D Wrapper class to make architecture generation simple
Created on Fri Feb 27 15:35:01 2015

@author: Kevin Chavez
"""
import theano
import theano.tensor as T
import numpy as np

from src.convnet3d.convnet3d import PoolLayer,ConvLayer, dtensor5
from src.convnet3d.mlp import LogRegr,DropoutHiddenLayer
from src.convnet3d.activations import relu

class ConvNet3D(object):
    def __init__(self,video_shape,batch_size=10,
                 seed=1234):
        """ Initializes a bare-bones ConvNet for video data.
        
        Args:
            (tuple) video_shape - triple of (num frames, height, width) of data
            (int)   mem_batch_size  - number of data points to have IN MEMORY 
                                      at a time, NOT training batch size.
            (int)   train_batch_size - number of data points per batch at train
        """
        self.layers = []
        self.data_loss = None        
        self.y_pred = None
        self.parameters = []
        self.batch_size = batch_size # TODO: make this actually do
                                                 # something
        
        # Store data in Theano shared variables, so it can be schlepped to GPU
        data_shape = (batch_size,3) + video_shape
        self.X = dtensor5() #theano.shared(np.random.randn(*data_shape))
        self.y = T.ivector() #theano.shared(np.zeros(mem_batch_size,dtype='int32'))
        
        self.top = self.X
        self.top_shape = data_shape[1:] # 4D tensor indicating shape of SINGLE
                                        # training example's output at top of 
                                        # net so far
        
        # Random number generator for replicable results
        self.rng = np.random.RandomState(seed)
        
    def add_conv_layer(self,name,filter_shape,num_filters):
        """ Add convolutional layer. Preserves input volume.
        
            :dafdag
        """
        n_input_maps = self.top_shape[0]
        input_volume = self.top_shape[1:]
        print input_volume
        conv = ConvLayer(self.top,n_input_maps, num_filters, filter_shape,
                         input_volume, self.batch_size,
                         relu, self.rng, layer_name=name)
                      
        self.parameters += conv.params
        self.layers.append(conv)
        
        self.top = conv.output
        self.top_shape = (num_filters,) + input_volume
        
    def add_pool_layer(self,name,pool_shape):
        """ Add pool layer, reducing input volume.
        """
        # Make sure it's divisible by the pool shape
        assert self.top_shape[1] % pool_shape[0] == 0
        assert self.top_shape[2] % pool_shape[1] == 0
        assert self.top_shape[3] % pool_shape[2] == 0
        
        pool = PoolLayer(self.top,pool_shape)
        pool.name = name
        self.layers.append(pool)
        
        self.top = pool.output
        self.top_shape = (self.top_shape[0],) + \
                         tuple(self.top_shape[i+1]/pool_shape[i] 
                               for i in range(3))
                         
    def add_fc_layer(self,name,num_units,dropout_rate):
        """ Add fully connected layer with dropout.
        """
        flat_top = self.top.flatten(ndim=2)
        input_size = np.prod(self.top_shape)
        fc = DropoutHiddenLayer(flat_top,input_size,num_units,relu, dropout_rate,
                                 self.rng, layer_name=name)
                         
         #self, input, n_in, n_out, activation, dropout_rate, use_bias, rng, W=None, b=None):
        self.parameters += fc.params
        self.layers.append(fc)
        
        self.top = fc.output
        self.top_shape = (num_units,)
        
        
    def add_softmax_layer(self,name,num_classes):
        """ Add a softmax loss layer.
        """
        flat_top = self.top.flatten(ndim=2)
        input_size = np.prod(self.top_shape)
        
        softmax = LogRegr(flat_top,input_size,num_classes,None,self.rng,
                          layer_name=name)

        self.parameters += softmax.params
        self.layers.append(softmax)
        
        self.top = softmax.p_y_given_x
        self.top_shape = (num_classes,)
        
        # Expose some top level expressions
        self.y_pred = softmax.y_pred
        self.data_loss = softmax.negative_log_likelihood(self.y)
        
    
def test():
    video_shape = (4,16,16)
    net = ConvNet3D(video_shape)
    num_classes = 5
    
    net.add_conv_layer("conv1",(3,3,3),2)
    net.add_pool_layer("pool1",(2,2,2))
    net.add_conv_layer("conv2",(1,3,3),4)
    net.add_fc_layer("fc1",10,0.9)
    net.add_softmax_layer("softmax",num_classes)
    
    N = 100
    X = theano.shared(np.random.randn(N,3,*video_shape).astype(theano.config.floatX),
                      borrow=True)
    y = theano.shared(np.random.randint(0,2,size=(N,)).astype('int32'),
                      borrow=True)
                      
    batch_size = 10
    index = T.lscalar()
    
    # Build function to evaluate data loss
    data_loss = theano.function(
                    inputs=[index],
                    outputs=[net.data_loss],
                    givens={
                        net.X: X[index * batch_size:(index + 1) * batch_size],
                        net.y: y[index * batch_size:(index + 1) * batch_size]},
                    on_unused_input='warn')
    
    return net, data_loss
    
    
if __name__ == "__main__":
    net, loss = test()

        
        