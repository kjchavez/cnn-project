# -*- coding: utf-8 -*-
"""
Data layer
Created on Fri Feb 27 19:14:35 2015

@author: kevin
"""
import time
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
    def __init__(self,db_name,video_shape,mem_batch_size,verbose=False):
        self.fetcher = DataFetcher(db_name,video_shape,mem_batch_size,dtype=theano.config.floatX)
        self.batch_size = mem_batch_size
        self.video_shape = video_shape
        self.current_batch = 0
        self.verbose = verbose
        
        X = np.empty((mem_batch_size,3)+video_shape,dtype=theano.config.floatX)
        y = np.empty((mem_batch_size,),dtype=theano.config.floatX)
        self.shared_data = theano.shared(X,borrow=True)
        self.shared_label = theano.shared(y,borrow=True)
        self.X = self.shared_data
        self.y = T.cast(self.shared_label,'int32')
        
    def load_batch(self):
        """ Updates the Theano shared variables (will incur CPU-GPU communication)
        """
        tic = time.time()
        X, y, epoch = self.fetcher.load_data()
        self.shared_data.set_value(X,borrow=True)
        self.shared_label.set_value(y,borrow=True)
        toc = time.time()
        if self.verbose:
            print "DataLayer: loading time = %0.6f" % (toc - tic)
        if epoch:
            return True
        else:
            return False
            
            
def test():
    import numpy as np
    video_shape = (16,112,112)
    data = DataLayer("data/tinytraindb.lmdb",video_shape,8,verbose=True)
    data2 = DataLayer("data/tinyvaldb.lmdb",video_shape,8,verbose=True)
    
    for i in range(1000):
        data.load_batch()
        data2.load_batch()
        a = np.random.randn(400,600).dot(np.random.randn(600,400))
        print np.linalg.norm(a)
        print np.linalg.norm(data.X.get_value(borrow=True))
        print np.linalg.norm(data2.X.get_value(borrow=True))
        print np.mean(data.shared_label.get_value(borrow=True))
        #print "Next..."

if __name__ == "__main__":
    test()