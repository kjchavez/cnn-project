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
    def __init__(self,database_name,video_shape,batch_size,dtype='float32'):
        self.db_name = database_name
        self.video_shape = video_shape
        self.batch_size = batch_size
        self.env = lmdb.open(database_name)
        self.txn = self.env.begin()
        self.cursor = self.txn.cursor()
        self.cursor.first()
        self.iterator = iter(self.cursor)
        self.dtype = dtype
        self.epoch = 0
        
    def load_data(self):
        """Fetches a set of videos from an LMDB database.
        
        Args:
            batch_size - number of videos to load into memory
            video_shape - 3 tuple (frames, height, width) for videos
        """
        TT, HH, WW = self.video_shape
        X = np.empty((self.batch_size,3) + self.video_shape,dtype=self.dtype)
        y = np.empty((self.batch_size,),dtype=self.dtype)
        crossed_epoch = False
        for n in xrange(self.batch_size):
            try:
                key,value = next(self.iterator)
            except StopIteration:
                self.cursor.first() # reset to beginning
                self.iterator = iter(self.cursor)
                crossed_epoch = True
                self.epoch += 1
                key,value = next(self.iterator)
                
            datum = Datum4D.fromstring(value)
            X[n] = datum.array
            y[n] = datum.label
                
        return X, y, crossed_epoch
                
    def __del__(self):
        self.txn.commit()
        self.env.close()


def test():
    import time
    import cv2
    
    db = "data/tinytraindb.lmdb"
    shape = (16,112,112)
    fetcher = DataFetcher(db,shape,8)
    
    # Fetch from database, possibly overflowing and repeating
    tic = time.time()
    X, y, _ = fetcher.load_data()
    toc = time.time()
    im = X[0,:,0].transpose(1,2,0) + APPROXIMATE_MEAN
    cv2.imshow("window",im.astype('uint8'))
    print "Retrieved %d videos in %0.4f milliseconds" % (X.shape[0],1000*(toc-tic))
    print "Average retrieval time: %0.3f ms" % (1000*(toc-tic)/X.shape[0])

if __name__ == "__main__":
    test()