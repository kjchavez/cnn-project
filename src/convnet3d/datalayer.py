# -*- coding: utf-8 -*-
"""
Data layer
Created on Fri Feb 27 19:14:35 2015

@author: kevin
"""
import os
import time
import multiprocessing
import ctypes
import theano
import theano.tensor as T
import numpy as np
from src.dataio.fetcher import DataFetcher

def fetcher_loop(fetcher,X_shmem,y_shmem,queue):
    """ Worker process that fills the queue as needed and as fast as it can.
    """
    index = 0
    print 'DataFetcher running with process id:', os.getpid()
    while True:
        X, y, crossed_epoch = fetcher.load_data()

        # Put into shared memory array
        X_shmem[index] = X
        y_shmem[index] = y
        queue.put((index,crossed_epoch))
        index = (index + 1) % X_shmem.shape[0]

class DataLayer:
    """ Wrapper around lmdb database for creating/updating GPU-sized chunks of
    training data. Multiprocessed.
    """
    def __init__(self,db_name,video_shape,mem_batch_size,
                 verbose=False,buffer_size=6):
        self.fetcher = DataFetcher(db_name,video_shape,mem_batch_size,
                                   dtype=theano.config.floatX)
        self.batch_size = mem_batch_size
        self.video_shape = video_shape
        self.current_batch = 0
        self.verbose = verbose
        
        # Could manage with a buffer size of exactly 2, but need to change
        # the interprocess communication somewhat
        assert buffer_size > 2
        
        X = np.empty((mem_batch_size,3)+video_shape,dtype=theano.config.floatX)
        y = np.empty((mem_batch_size,),dtype=theano.config.floatX)
        self.shared_data = theano.shared(X,borrow=True)
        self.shared_label = theano.shared(y,borrow=True)
        self.X = self.shared_data
        self.y = T.cast(self.shared_label,'int32')
        
        # Create shared memory object for async loading
        X_shared_array_base = multiprocessing.Array(ctypes.c_float, 
                                  buffer_size*self.batch_size*3*np.prod(video_shape))
        X_shared_array = np.ctypeslib.as_array(X_shared_array_base.get_obj())
        self.X_shared_array = X_shared_array.reshape(buffer_size,self.batch_size,
                                                     3,*video_shape)
 
        y_shared_array_base = multiprocessing.Array(ctypes.c_float,
                                                    buffer_size*self.batch_size)
        y_shared_array = np.ctypeslib.as_array(y_shared_array_base.get_obj())
        self.y_shared_array = y_shared_array.reshape(buffer_size,self.batch_size)

        # Start up worker process
        self.queue = multiprocessing.Queue(maxsize=buffer_size - 2)
        self.worker = multiprocessing.Process(target=fetcher_loop,
                                         args=(self.fetcher,
                                               self.X_shared_array,
                                               self.y_shared_array,
                                               self.queue))
        self.worker.start()
        
    def load_batch(self):
        """ Updates the Theano shared variables (will incur CPU-GPU communication)
        """
        tic = time.time()
        idx, epoch = self.queue.get()
        X = self.X_shared_array[idx]
        y = self.y_shared_array[idx]
        self.shared_data.set_value(X,borrow=True)
        self.shared_label.set_value(y,borrow=True)
        toc = time.time()
        if self.verbose:
            print "DataLayer: loading time = %0.6f" % (toc - tic)
        if epoch:
            return True
        else:
            return False

    def __del__(self):
        self.worker.terminate()
            
def test():
    import numpy as np
    video_shape = (16,112,112)
    data = DataLayer("data/tinytraindb.lmdb",video_shape,16,verbose=True)
    data2 = DataLayer("data/tinyvaldb.lmdb",video_shape,16,verbose=True)
    
    synch_data = []
    # Retrieve data synchronously as a reference for correctness
    fetcher = DataFetcher("data/tinytraindb.lmdb",video_shape,16,dtype='float32')
    for i in range(10):
        X, y, epoch = fetcher.load_data()
        synch_data.append(X)
        #print X
    
    for i in range(10):
        data.load_batch()
        data2.load_batch()
        tic = time.time()
        # Do some work
        a = np.random.randn(400,600).dot(np.random.randn(600,400))
        #print np.linalg.norm(a)
        toc = time.time()
        print "Work took %0.6f seconds" % (toc - tic)
        assert np.linalg.norm(data.X.get_value(borrow=True) - synch_data[i]) < 1e-8
   
if __name__ == "__main__":
    test()