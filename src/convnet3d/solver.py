# -*- coding: utf-8 -*-
"""
ConvNet Trainer
Created on Fri Feb 27 19:08:50 2015

@author: Kevin Chavez
"""
import sys, os
import time
import theano
import theano.tensor as T
import numpy as np
import cPickle
from src.convnet3d.regularization import *
from collections import OrderedDict

class Solver:
    def __init__(self,conv_net,reg_params,opt_params):
        self.conv_net = conv_net
        self.method = opt_params["method"]
        base_learning_rate = opt_params["lr_base"]
        if self.method == "rmsprop":
            rmsprop_decay = opt_params['rmsprop_decay']
        elif self.method == "momentum":
            momentum = opt_params['initial']
            self.final_momentum = opt_params['final']
            momentum_step = opt_params['step']
        else:
            raise NotImplemented("Optimization method %s is not implemented" %
                                 opt_params["method"])

        learning_rate_decay = opt_params['lr_decay']    
        
        # Might be unnecessary... I think our minibatch is limited by the amount
        # of GPU memory so we'll only ever use index = 0....
        index = T.lscalar()
        
        self.learning_rate = theano.shared(np.asarray(base_learning_rate,
            dtype=theano.config.floatX))

        self.decay_learning_rate = \
            theano.function(
                inputs=[], 
                outputs=self.learning_rate,
                updates={self.learning_rate: self.learning_rate * learning_rate_decay})
                

        if self.method == "momentum":
            self.momentum = theano.shared(np.asarray(momentum,
                                          dtype=theano.config.floatX))
            self.increase_momentum = \
                theano.function(
                    inputs=[], 
                    outputs=self.momentum+momentum_step,
                    updates={self.momentum: self.momentum + momentum_step})
            

        print "Compiling validation function..."
        # Compile theano function for validation.
        val_X = conv_net.val_data.X
        val_y = conv_net.val_data.y
        batch_size = conv_net.batch_size
        self.validate_model = \
            theano.function(inputs=[index],
                outputs = conv_net.accuracy,
                givens={
                    conv_net.X: val_X[index * batch_size:(index + 1) * batch_size],
                    conv_net.y: val_y[index * batch_size:(index + 1) * batch_size]})
        #theano.printing.pydotprint(validate_model, outfile="test_file.png",
        #        var_with_name_simple=True)
        
        # Compute gradients of the model wrt parameters. We will also explicitly
        # construct and incorporate the regularization terms in this loop.
        # For simple regularization, it's sufficient to add the term to the loss
        # and let Theano work its magic. However, this project proposes a
        # more sophisticated regularization scheme which Theano doesn't know
        # how to automatically differentiate.
        loss = conv_net.data_loss
        gparams = []
        for param in conv_net.parameters:
            gparam = T.grad(conv_net.data_loss, param)
            
            if param.name in reg_params:
                print "Adding regularization to", param.name
                reg, reg_grad, updates, info = l2_regularizer(param) 
                loss = loss + reg_params[param.name] *reg
                gparams.append(gparam + reg_params[param.name]*reg_grad)
            else:
                gparams.append(gparam)

        # Container to hold parameter updates regardless of optimization scheme
        updates = OrderedDict()    
    
        if self.method == 'rmsprop':
            # #####################################################################
            # RMS Prop. 
            # See Geoff Hinton's slides. Or the material for CS231N at Stanford. 
            # Great resource!
            #######################################################################
            rmsprop_cache = []
            for param in conv_net.parameters:
                cache = theano.shared(np.zeros(param.get_value(borrow=True).shape,
                    dtype=theano.config.floatX))
                rmsprop_cache.append(cache)
       
            # Update the rms prop cache
            for cached, gparam in zip(rmsprop_cache, gparams):
                # change the update rule to match Hinton's dropout paper
                updates[cached] = rmsprop_decay * cached  + (1-rmsprop_decay) * T.square(gparam)
                
            # ... and take a gradient step
            for param, gparam, cached in zip(conv_net.parameters,gparams,rmsprop_cache):
                stepped_param = param - self.learning_rate * gparam / (T.sqrt(cached)+1e-8)
                updates[param] = stepped_param
            
        if self.method == "momentum":
                # ... and allocate mmeory for momentum'd versions of the gradient
            momenta = []
            for param in conv_net.parameters:
                m = theano.shared(np.zeros(param.get_value(borrow=True).shape,
                    dtype=theano.config.floatX))
                momenta.append(m)
        
            # Update the step direction using momentum
            updates = OrderedDict()
            for gparam_mom, gparam in zip(momenta, gparams):
                # change the update rule to match Hinton's dropout paper
                updates[gparam_mom] = momentum * gparam_mom - (1. - momentum) \
                                      * self.learning_rate * gparam
                
            for param, gparam_mom in zip(conv_net.parameters, momenta):
                # since we have included learning_rate in gparam_mom, 
                # we don't need it here
                stepped_param = param + updates[gparam_mom]
                updates[param] = stepped_param
            
        print "Compiling train function..."
        # Compile theano function for training.  This returns the training cost and
        # updates the model parameters.
        train_X = conv_net.train_data.X
        train_y = conv_net.train_data.y
        self.train_model = \
            theano.function(
                inputs = [index], 
                outputs = loss,
                updates = updates,
                givens = {
                    conv_net.X: train_X[index * batch_size:(index + 1) * batch_size],
                    conv_net.y: train_y[index * batch_size:(index + 1) * batch_size]})
        #theano.printing.pydotprint(train_model, outfile="train_file.png",
        #        var_with_name_simple=True)
    

                
    def validate(self):
        epoch_ended = False
        accuracies = []
        while not epoch_ended:
            epoch_ended = self.conv_net.val_data.load_batch()
            accuracies.append(self.validate_model(0))
            
        return np.mean(accuracies)
            
    def train(self,n_iter,snapshot_params,validate_rate=100,loss_rate=10):
        """ Train the solvers network for a number of iterations.
        
        Args:
            n_iter:          number of minibatches to train with
            snapshot_params: a dictionary of parameters dictating when and
                             where to save snapshots of the model
            validate_rate:   compute and print validation accuracy every this
                             many iterations of training
            loss_rate:       print minibatch training loss every loss_rate iters
        """
        snapshot_rate = snapshot_params['rate']
        snapshot_dir = snapshot_params['dir']
        if not os.path.isdir(snapshot_dir):
            os.makedirs(snapshot_dir)
        # Write the list of parameters:
        with open(os.path.join(snapshot_dir,"parameter-names.txt"),'w') as fp:
            for param in self.conv_net.parameters:
                print >> fp, param.name
                
        filepattern = os.path.join(
                        snapshot_dir,
                        self.conv_net.name+".snapshot.iter-%06d")
            
        best_validation_acc = 0
        val_history_filename = os.path.join(snapshot_dir,"validation-history.txt")
        loss_history_filename = os.path.join(snapshot_dir,"loss-history.txt")
        epoch_counter = 0
        start_time = time.clock()

        print "Starting training..."    
        loss_history = []
        for iteration in xrange(n_iter):
            epoch_ended = self.conv_net.train_data.load_batch()

            # Train this batch
            minibatch_avg_cost = self.train_model(0)
            loss_history.append(minibatch_avg_cost)
                                                                
            if iteration % loss_rate == 0:
                print "minibatch loss:", minibatch_avg_cost

            if iteration == 0:
                continue

            if epoch_ended:
                epoch_counter += 1
                print "Completed epoch %d" % epoch_counter
                self.decay_learning_rate()
                if self.method == "momentum":
                    if self.momentum.get_value() < self.final_momentum:
                        momentum = self.increase_momentum()
                        print "  new momentum = %0.4f" % momentum
                
            if iteration % validate_rate == 0:
                # Compute accuracy on validation set
                val_accuracy = self.validate()
                
                print "iter %d, val accuracy = %0.4f, learning rate = %0.4e" % \
                        (iteration,val_accuracy,self.learning_rate.get_value())
                
                with open(val_history_filename,'a') as fp:
                    print >> fp, iteration, val_accuracy
                
                # Flush loss history
                with open(loss_history_filename,'a') as fp:
                    for loss in loss_history:
                        print >> fp, loss
                    loss_history = []
                        
                if (val_accuracy > best_validation_acc):
                    print "** Best score so far **"
                    filename = os.path.join(snapshot_dir,"best-model")
                    with open(filename,'wb') as fp:
                        for param in self.conv_net.parameters:
                            cPickle.dump(param.get_value(borrow=True),fp,-1)
                
                best_validation_acc = max(best_validation_acc,val_accuracy)
                                
            if iteration % snapshot_rate == 0:
                # Save a snapshot of the parameters
                filename = filepattern % (iteration,)
                with open(filename,'wb') as fp:
                    for param in self.conv_net.parameters:
                        cPickle.dump(param.get_value(borrow=True),fp,-1)
            
        end_time = time.clock()
        print(('Optimization complete. Best validation score of %f %% ') %
              (best_validation_acc * 100))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

def test():
    net = get_test_net()
    opt_params = {
        "method": "momentum",
        "initial": 0.5,
        "final": 0.9,
        "step": 0.1, # per epoch
        "lr_decay": 0.95,
        "lr_base": 1e-1}

    reg_params = {} #{'conv1_W': 0.1,'conv2_W': 0.2,'fc1_W': 0.3,'softmax_W':0.1}
    snapshot_params = {"dir": "models/"+net.name,"rate":2}
    
    solver = Solver(net,reg_params,opt_params)
    solver.train(60,snapshot_params,validate_rate=2,loss_rate=1)
    
if __name__ == "__main__":
    from src.convnet3d.cnn3d import get_test_net
    if len(sys.argv) > 1 and sys.argv[1] == "-v":
        theano.config.exception_verbosity = "high"
        
    test()