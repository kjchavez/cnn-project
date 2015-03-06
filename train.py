# -*- coding: utf-8 -*-
"""
CS 231N Project: 3D CNN with Optical Flow Regularization
Experiments
Created on Sun Mar  1 14:47:40 2015

@author: Kevin Chavez
"""
import sys, os
import argparse

from src.convnet3d.cnn3d import ConvNet3D
from src.convnet3d.solver import Solver

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
    parser.add_argument("--resume",type=int,default=None,
                        help="Snapshot file to resume training. Up to user to "
                             "provide valid snapshots, since only parameter "
                             "sizes will be checked. This means you can change"
                             " regularization parameters during training "
                             "(including dropout parameter) if so desired")

    args = parser.parse_args()

    properties = {}
    layers = []    
    with open(args.net) as fp:
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
        exit(1)
    assert 'video-shape' in properties
    try:
        properties['video-shape'] = \
            tuple(int(x) for x in properties['video-shape'] \
                                  .strip('(').rstrip(')').split(','))
    except e:
        print e
        print >> sys.stderr, "video-shape not in valid format"
        exit(1)
    
    # Create convnet
    net = ConvNet3D(properties['name'],
                   properties['video-shape'],
                   properties['batch-size'],
                   seed=args.seed)
                   
    # Add train / val databases
    print properties['train'], properties['val']
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
            shape = shape.strip(" (").rstrip(')')
            shape = tuple(int(x) for x in shape.split(','))
            num_filters = int(num_filters)
            name = "conv%d"%conv_count
            net.add_conv_layer(name, shape, num_filters)
            reg_multipliers[name+"_W"] = float(reg_mult.split('=')[1])
        
        if layer_type == "pool":
            value = value.strip(" (").rstrip(')')
            shape = tuple(int(x) for x in value.split(','))
            net.add_pool_layer("pool%d"%conv_count,shape)
            
        if layer_type == "fc":
            fc_count += 1
            num_units_str, reg_mult = value.split()
            num_units = int(num_units_str)
            p = args.dropout[min(fc_count,len(args.dropout))-1]
            name = "fc%d"%fc_count
            net.add_fc_layer(name,num_units, p)
            reg_multipliers[name+"_W"] = float(reg_mult.split('=')[1])
            
        if layer_type == "softmax":
            num_classes_str, reg_mult = value.split()
            num_classes = int(num_classes_str)
            net.add_softmax_layer("softmax",num_classes)
            reg_multipliers["softmax_W"] = float(reg_mult.split('=')[1])

    snapshot_params = {
        "dir": "models/" + properties['name']+"-%04d"%args.trial_id,
        "rate": args.snapshot_rate,
        "resume": args.resume}
    
    opt_params = {
        "method": "momentum",
        "initial": args.mom_init,
        "final": args.mom_final,
        "step": args.mom_step, # per epoch
        "lr_decay": args.lr_decay,
        "lr_base": args.lr}
        
    reg_params = dict((param,mult*args.reg) for param,mult in reg_multipliers.items())
           
    # Save the setting to the log file of hyper-parameter settings
    if not os.path.isdir('logs'):
        os.makedirs('logs')
    
    solver = Solver(net,reg_params,opt_params)
    best_val_accuracy, best_val_iter = solver.train(
                                         args.num_iter,
                                         snapshot_params,
                                         validate_rate=args.validate_rate,
                                         loss_rate=args.loss_rate)

    log_filename = 'logs/'+properties['name']+'-trials.txt'
    if not os.path.exists(log_filename):
        with open(log_filename,'w') as fp:
            print >> fp, '\t'.join(['trial-id','best-val-acc',
                                    'best-val-iter','settings'])

    with open(log_filename,'a') as fp:
        print >> fp, "\t".join([str(args.trial_id),str(best_val_accuracy),
                                str(best_val_iter),str(vars(args))])


if __name__ == "__main__":
    net = main()