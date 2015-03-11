# -*- coding: utf-8 -*-
"""
CS 231N Project: 3D CNN with Optical Flow Regularization
Experiments
Created on Sun Mar  1 14:47:40 2015

@author: Kevin Chavez
"""
import sys, os
import shutil
import argparse

from src.convnet3d.cnn3d import ConvNet3D
from src.convnet3d.solver import Solver

def train(net_file,trial_id,resume=None,seed=1234,dropout=[0.5],snapshot_rate=500,
          validate_rate=500,num_iter=20000,loss_rate=1,reg=1e-3,mom_init=0.5,
          mom_final=0.9,mom_step=0.1,lr_decay=0.95,lr=1e-5,optflow_weight=0):
    """Trains a network described in the file |net| with particular settings.
    
    Args:
        net - text file describing network architecture
        trial_id - unique integer identifying trial number which corresponds 
                   to the parameter settings
        resume - integer indicating iteration from which to resume training
        ...
        
    Returns:
        A tuple of best validation accuracy and the iteration when it occurred.
    """
    properties = {}
    layers = []    
    with open(net_file) as fp:
        for line in fp:
            if line == '\n':
                continue
            
            prop, value = line.split(":")
            if prop in ('video-shape','train','val','batch-size','name'):
                properties[prop] = value.strip().rstrip()
            elif prop in ('pool','conv','fc','softmax'):
                layers.append((prop,value.rstrip()))

    # Assert all necessary fields are present and valid
    assert 'name' in properties
    assert 'train' in properties
    assert 'val' in properties
    assert 'batch-size' in properties
    try:
        properties['batch-size'] = int(properties['batch-size'])
    except:
        print >> sys.stderr, "batch-size must be an integer"
        return None, None
    assert 'video-shape' in properties
    try:
        properties['video-shape'] = \
            tuple(int(x) for x in properties['video-shape'] \
                                  .strip('(').rstrip(')').split(','))
    except:
        print >> sys.stderr, "video-shape not in valid format"
        return None, None

    # Create directory to store results
    savepath = os.path.join("results",properties['name']+"-%04d"%trial_id)
    if os.path.isdir(savepath) and resume is None:
        print "Attempted to overwrite %s with brand new training." % savepath
        print "Training aborted. If you wish to proceed, please delete " \
              "%s explicitly, then rerun command" % savepath
        return None, None
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    
    # Create convnet
    net = ConvNet3D(properties['name'],
                   properties['video-shape'],
                   properties['batch-size'],
                   seed=seed)
                   
    # Add train / val databases
    net.add_train_data(properties['train'])
    net.add_val_data(properties['val'])

    reg_multipliers = {}

    # We will follow convention of naming layers based on how many convolutions
    # deep in the architecture they are. For example, a pool layer coming after
    # the 6th conv layer will be pool6, even if it isn't the 6th pooling layer.
    conv_count = 0
    fc_count = 0
    for layer_type, value in layers:
        if layer_type == "conv":
            conv_count += 1
            shape, num_filters, reg_mult = value.split()
            shape = shape.strip("( )")
            shape = tuple(int(x) for x in shape.split(','))
            num_filters = int(num_filters)
            name = "conv%d"%conv_count
            net.add_conv_layer(name, shape, num_filters)
            reg_multipliers[name+"_W"] = float(reg_mult.split('=')[1])
        
        if layer_type == "pool":
            value = value.strip("( )")
            shape = tuple(int(x) for x in value.split(','))
            net.add_pool_layer("pool%d"%conv_count,shape)
            
        if layer_type == "fc":
            fc_count += 1
            num_units_str, reg_mult = value.split()
            num_units = int(num_units_str)
            p = dropout[min(fc_count,len(dropout))-1]
            name = "fc%d"%fc_count
            net.add_fc_layer(name,num_units, p)
            reg_multipliers[name+"_W"] = float(reg_mult.split('=')[1])
            
        if layer_type == "softmax":
            num_classes_str, reg_mult = value.split()
            num_classes = int(num_classes_str)
            net.add_softmax_layer("softmax",num_classes)
            reg_multipliers["softmax_W"] = float(reg_mult.split('=')[1])
        
    snapshot_params = {
        "dir": "snapshots",
        "rate": snapshot_rate,
        "resume": resume}
    
    opt_params = {
        "method": "momentum",
        "initial": mom_init,
        "final": mom_final,
        "step": mom_step, # per epoch
        "lr_decay": lr_decay,
        "lr_base": lr}
        
    reg_params = dict((param,mult*reg) for param,mult in reg_multipliers.items())

    # Copy the network architecture description file to the results folder
    shutil.copy(net_file,os.path.join(savepath,'architecture.txt'))    

    solver = Solver(net,reg_params,opt_params)
    best_val_accuracy, best_val_iter = solver.train(
                                         num_iter,
                                         snapshot_params,
                                         savepath,
                                         validate_rate=validate_rate,
                                         loss_rate=loss_rate,
                                         optflow_weight=optflow_weight)
    return best_val_accuracy, best_val_iter
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("net",
                        help="Text file describing the cnn architecture")
    parser.add_argument("trial_id",type=int,
                        help="identification number for hyperparameter "
                             "settings. Must be unique.")
    parser.add_argument("--reg",type=float,default=1e-3,
                        help="Base regularization parameter")
    parser.add_argument("--dropout",type=float,nargs='+',default=[0.5],
                        help="probability of dropping out a neuron's activation")
    parser.add_argument("--lr",type=float,default=1e-5,
                        help="Base learning rate")
    parser.add_argument("--lr-decay",dest="lr_decay",type=float,default=0.95,
                        help="Multiplicative decay to learning rate after "
                             "each epoch")
    parser.add_argument("--mom-init",dest="mom_init",type=float,default=0.5,
                        help="Initial value of momentum parameter")
    parser.add_argument("--mom-final",dest="mom_final",type=float,default=0.9,
                        help="Final value of momentum parameter")
    parser.add_argument("--mom-step", dest="mom_step",type=float,default=0.1,
                        help="Increase in momentum per epoch")
    parser.add_argument("--snapshot-rate",dest="snapshot_rate",type=int,
                        default=500,help="How often to save snapshots of model")
    parser.add_argument("--validate-rate",dest="validate_rate",type=int,
                        default=500,help="How often to check validation score")
    parser.add_argument("--loss-rate", dest="loss_rate",type=int,
                        default=1,help="How often to print loss")
    parser.add_argument("--seed",type=int,default=1234,
                        help="Seed value for random number generator")
    parser.add_argument("--num-iter",'-n',dest="num_iter",type=int,default=20000,
                        help="Number of iterations to train network")
    parser.add_argument("--optflow-weight",dest='optflow_weight',type=float,
			default=0,help="Optflow regularization parameter")
    parser.add_argument("--resume",type=int,default=None,
                        help="Snapshot file to resume training. Up to user to "
                             "provide valid snapshots, since only parameter "
                             "sizes will be checked. This means you can change"
                             " regularization parameters during training "
                             "(including dropout parameter) if so desired")

    args = parser.parse_args()
    
    kwargs = vars(args).copy()
    del kwargs['net']
    del kwargs['trial_id']
    
    best_val_acc, best_val_iter = train(args.net,args.trial_id,**kwargs)
    print best_val_acc, best_val_iter


if __name__ == "__main__":
    net = main()
