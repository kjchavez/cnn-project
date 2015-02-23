# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 14:11:38 2015

@author: kevin
"""
import lmdb
import numpy as np
from src.dataio.datum import Datum4D
from src.constants import *

class DataFetcher(object):
    def __init__(self,database_name):
        self.db_name = database_name
        self.env = lmdb.open(database_name)
        self.epoch = 0
        
    def load_data(self,batch_size,video_shape):
        """Fetches a set of videos from an LMDB database.
        
        Args:
            batch_size - number of videos to load into memory
            video_shape - 3 tuple (frames, height, width) for videos
        """
        TT, HH, WW = video_shape
        X = np.empty((batch_size,3) + video_shape,dtype=np.uint8)
        y = np.empty((batch_size,),dtype=np.uint32)
        with self.env.begin() as txn:
            cursor = txn.cursor()
            it = iter(cursor)
            for n in xrange(batch_size):
                try:
                    key,value = next(it)
                except StopIteration:
                    cursor.first() # reset to beginning
                    it = iter(cursor)
                    print "DataFetcher: Completed epoch", self.epoch
                    self.epoch += 1
                    key,value = next(it)
                    
                datum = Datum4D.fromstring(value)
                X[n] = datum.array
                y[n] = datum.label
                
        return X, y
                
    def __del__(self):
        self.env.close()


def test():
    import time
    import cv2
    
    db = "data/tinyvideodb.lmdb"
    shape = (16,240,320)
    fetcher = DataFetcher(db)
    
    # Fetch from database, possibly overflowing and repeating
    tic = time.time()
    X, y = fetcher.load_data(100,shape)
    toc = time.time()
    im = X[10,:,0].transpose(1,2,0) + APPROXIMATE_MEAN
    cv2.imshow("window",im)
    print "Retrieved %d videos in %0.4f milliseconds" % (X.shape[0],1000*(toc-tic))
    print "Average retrieval time: %0.3f ms" % (1000*(toc-tic)/X.shape[0])

if __name__ == "__main__":
    test()