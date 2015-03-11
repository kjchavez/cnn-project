# -*- coding: utf-8 -*-
"""
Data layer
Created on Fri Feb 27 19:14:35 2015

@author: kevin
"""
import theano
import theano.tensor as T
import numpy as np
from src.dataio.fetcher import DataFetcher

def fetcher_loop(fetcher,shmem):
    """ Worker process that fills the queue as needed and as fast as it can.
    """
    while True:
        X, y, crossed_epoch = fetcher.load_data()
        # Put into shared memory array

class DataLayer:
    """ Wrapper around lmdb database for creating/updating GPU-sized chunks of
    training data. Multiprocessed.
    """
    def __init__(self,db_name,video_shape,mem_batch_size):
        self.fetcher = DataFetcher(db_name,video_shape,mem_batch_size,dtype=theano.config.floatX)
        self.batch_size = mem_batch_size
        self.video_shape = video_shape
        self.current_batch = 0
        
        X = np.empty((mem_batch_size,3)+video_shape,dtype=theano.config.floatX)
        y = np.empty((mem_batch_size,),dtype=theano.config.floatX)
        self.shared_data = theano.shared(X,borrow=True)
        self.shared_label = theano.shared(y,borrow=True)
        self.X = self.shared_data
        self.y = T.cast(self.shared_label,'int32')
        
    def load_batch(self):
        """ Updates the Theano shared variables (will incur CPU-GPU communication)
        """
        X, y, epoch = self.fetcher.load_data(self.batch_size,self.video_shape)
        self.shared_data.set_value(X,borrow=True)
        self.shared_label.set_value(y,borrow=True)
        
        if epoch:
            return True
        else:
            return False
            
            
def test():
    import numpy as np
    video_shape = (16,240,320)
    data = DataLayer("data/traindb.lmdb",video_shape,20)
    
    for i in range(1):
        data.load_batch()
        print np.linalg.norm(data.X.get_value(borrow=True))
        print np.mean(data.shared_label.get_value(borrow=True))
        print "Next..."

if __name__ == "__main__":
    test()