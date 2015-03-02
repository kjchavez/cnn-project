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
    def __init__(self,conv_net,reg_params,lr_params,rmsprop_decay):
        self.conv_net = conv_net
        initial_learning_rate = lr_params['rate']
        learning_rate_decay = lr_params['decay']
        self.step_rate = lr_params['step'] # in number of minibatches        
        
        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a minibatch
        self.learning_rate = theano.shared(np.asarray(initial_learning_rate,
            dtype=theano.config.floatX))

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
        #theano.printing.pydotprint(test_model, outfile="test_file.png",
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
        updates = OrderedDict()
        for cached, gparam in zip(rmsprop_cache, gparams):
            # change the update rule to match Hinton's dropout paper
            updates[cached] = rmsprop_decay * cached  + (1-rmsprop_decay) * T.square(gparam)
            
        # ... and take a gradient step
        for param, gparam, cached in zip(conv_net.parameters,gparams,rmsprop_cache):
            stepped_param = param - self.learning_rate * gparam / (T.sqrt(cached)+1e-8)
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
    
        # Theano function to decay the learning rate, this is separate from the
        # training function because we only want to do this once each epoch instead
        # of after each minibatch.
        self.decay_learning_rate = \
            theano.function(
                inputs=[], 
                outputs=self.learning_rate,
                updates={self.learning_rate: self.learning_rate * learning_rate_decay})
                
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
                        self.conv_net.name+".snapshot.iter-%06d.val-%0.4f")
            
        best_validation_acc = 0
        epoch_counter = 0
        start_time = time.clock()

        print "Starting training..."    
        for iteration in xrange(n_iter):
            epoch_ended = self.conv_net.train_data.load_batch()

            if epoch_ended:
                epoch_counter += 1
                print "Completed epoch %d" % epoch_counter
                
            if iteration % validate_rate == 0:
                # Compute accuracy on validation set
                val_accuracy = self.validate()
                
                print "iter %d, val accuracy = %0.4f, learning rate = %0.4e" % \
                        (iteration,val_accuracy,self.learning_rate.get_value())
                        
                if (val_accuracy > best_validation_acc):
                    print "** Best score so far **"
                
                best_validation_acc = max(best_validation_acc,val_accuracy)
                
            if iteration % self.step_rate == 0:
                self.decay_learning_rate()
                
            if iteration % snapshot_rate == 0:
                # Save a snapshot of the parameters
                filename = filepattern % (iteration,val_accuracy)
                with open(filename,'wb') as fp:
                    for param in self.conv_net.parameters:
                        cPickle.dump(param.get_value(borrow=True),fp,-1)
                            
            # Train this batch
            minibatch_avg_cost = self.train_model(0)
            if iteration % loss_rate == 0:
                print "minibatch loss:", minibatch_avg_cost
            
                    
        end_time = time.clock()
        print(('Optimization complete. Best validation score of %f %% ') %
              (best_validation_acc * 100))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

def test():
    net = get_test_net()
    lr_params = {'rate': 1e-8, 'decay': 0.95, 'step': 1}
    reg_params = {'conv1_W': 0.1,'conv2_W': 0.2,'fc1_W': 0.3,'softmax_W':0.1}
    snapshot_params = {"dir": "models/"+net.name,"rate":2}
    
    rmsprop_decay = 0.99
    solver = Solver(net,reg_params,lr_params,rmsprop_decay)
    solver.train(8,snapshot_params,validate_rate=2,loss_rate=1)
    
if __name__ == "__main__":
    from src.convnet3d.cnn3d import get_test_net
    if len(sys.argv) > 1 and sys.argv[1] == "-v":
        theano.config.exception_verbosity = "high"
        
    test()