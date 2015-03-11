# -*- coding: utf-8 -*-
"""
Hyperparameter Random Search
Multi-processed hyperparameter random search
Created on Fri Mar  6 00:12:39 2015

@author: Kevin Chavez
"""
import os
import shutil
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("net_file")
parser.add_argument("logfilename")
parser.add_argument("--optflow",action="store_true")
parser.add_argument("--device",default=None,
                    help="Specify a particular GPU or CPU. Note to specify a "
                         "particular GPU, should invoke script with THEANO_FLAGS "
                         "=device=cpu.")
parser.add_argument("--trial-id",dest="trial_id",type=int,default=0,
                    help="ID for first trial, rest will be sequential")
parser.add_argument("--num-trials","-n",dest="num_trials",type=int,default=10,
                    help="Number of trials of random search to run")
parser.add_argument("--num-iterations","-i",dest="num_iter",type=int,default=5000,
                    help="Number of iterations to run each trial")
                    
args = parser.parse_args()

if args.device:
    import theano.sandbox.cuda
    theano.sandbox.cuda.use(args.device)

import theano
theano.config.warn_float64 = 'warn'
from train import train

# Default values
net_file = args.net_file
logfilename = args.logfilename
kwargs = {
    'mom_init' : 0.5,
    'mom_final' : 0.9,
    'mom_step' : 0.1,
    'num_iter' : args.num_iter,
    'snapshot_rate' : 200,
    'validate_rate' : 200
}

# Searching on parameters:
# learning_rate, reg, dropout
for n in xrange(args.trial_id,args.trial_id+args.num_trials):
    kwargs['lr'] = np.float32(10**np.random.uniform(-8,-4))
    kwargs['reg'] = np.float32(10**np.random.uniform(-9,-1))
    kwargs['dropout'] = [np.float32(np.random.choice([0.1,0.2,0.3,0.4]))]

    if args.optflow:
        # Run optflow experiments
        optflow_weight = np.float32(10**np.random.uniform(-1,1))
        print "Starting trial with lr %0.4e, reg %0.4e, dropout %0.2f, optflow-reg %0.3e" % \
              (kwargs['lr'], kwargs['reg'], kwargs['dropout'][0], optflow_weight)
              
        def log_result(best_val_acc,best_val_iter,optflow_acc,optflow_iter):
            if best_val_iter is None:
                best_val_iter = -1
            if optflow_iter is None:
                optflow_iter = -1
            with open(logfilename,'a') as fp:
                print >> fp, "%03d\t%0.4e\t%0.4e\t%s\t%0.3e\t%0.4f\t%06d\t%0.4f\t%06d" % \
                         (n, kwargs['lr'], kwargs['reg'], str(kwargs['dropout']), 
                          optflow_weight,best_val_acc, best_val_iter,
                          optflow_acc, optflow_iter)
                              
            print "Completed trial %d." % n

        kwargs['optflow_weight'] = optflow_weight
        optflow_acc, optflow_iter = train(net_file, n + 1000,**kwargs.copy())
        del kwargs['optflow_weight']
	val_acc, val_iter = train(net_file, n, **kwargs.copy())
        
        log_result(val_acc,val_iter,optflow_acc,optflow_iter)

              
    else:
        # Just regular hyperparameter searching
        print "Starting trial with lr %0.4e, reg %0.4e, dropout %0.2f..." % \
              (kwargs['lr'], kwargs['reg'], kwargs['dropout'][0])
          
        def log_result(best_val_acc,best_val_iter):
            with open(logfilename,'a') as fp:
                print >> fp, "%03d\t%0.4e\t%0.4e\t%s\t%0.4f\t%06d" % \
                         (n, kwargs['lr'], kwargs['reg'], str(kwargs['dropout']), 
                          best_val_acc, best_val_iter)
            print "Completed trial %d." % n
    
        while True:
            try:                    
                val_acc, val_iter = train(net_file, n, **kwargs.copy())
            except:
                print "Train function failed. Restarting..."
                shutil.rmtree("results/%s-%04d" % (net_file.split('/')[1].split('.')[0],n))
                continue
            
            # Once successful, log results
            log_result(val_acc,val_iter)
            break
