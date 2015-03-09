# -*- coding: utf-8 -*-
"""
Hyperparameter Random Search
Multi-processed hyperparameter random search
Created on Fri Mar  6 00:12:39 2015

@author: Kevin Chavez
"""
import sys
import numpy as np

if len(sys.argv) > 1:
    device_num = int(sys.argv[1])
    import theano.sandbox.cuda
    theano.sandbox.cuda.use("gpu"+str(device_num))
else:
    device_num = 0
    
import theano
theano.config.warn_float64 = 'warn'
from train import train

# Default values
net_file = 'models/videonet.txt'
logfilename = 'results/videonet-hyperparameters.txt'
kwargs = {
    'mom_init' : 0.5,
    'mom_final' : 0.9,
    'mom_step' : 0.1,
    'num_iter' : 2000,
    'snapshot_rate' : 500,
    'validate_rate' : 500
}

#pool = multiprocessing.Pool(processes=MAX_PROCESSES)
# Searching on parameters:
# learning_rate, reg, dropout
N = 10
for n in xrange(N):
    kwargs['lr'] = np.float32(10**np.random.uniform(-8,-2))
    kwargs['reg'] = np.float32(10**np.random.uniform(-8,1))
    kwargs['dropout'] = [np.float32(np.random.choice([0.2,0.4,0.6,0.8]))]
    
    print "Starting trial with lr %0.4e, reg %0.4e, dropout %0.2f..." % \
          (kwargs['lr'], kwargs['reg'], kwargs['dropout'][0])
          
    def log_result(best_val_acc,best_val_iter):
        with open(logfilename,'a') as fp:
            print >> fp, "%03d\t%0.4e\t%0.4e\t%s\t%0.4f\t%06d" % \
                         (n, kwargs['lr'], kwargs['reg'], str(kwargs['dropout']), 
                          best_val_acc, best_val_iter)
        print "Completed trial %d." % n
                    
    val_acc, val_iter = train(net_file, n + N*device_num, **kwargs.copy())
    if val_acc and val_iter:
        log_result(val_acc,val_iter)
    else:
        print "Trial failed."
