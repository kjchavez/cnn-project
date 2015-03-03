# -*- coding: utf-8 -*-
"""
CS 231N Project: 3D CNN with Optical Flow Regularization
Experiments
Created on Sun Mar  1 14:47:40 2015

@author: Kevin Chavez
"""
from src.convnet3d.cnn3d import ConvNet3D
from src.convnet3d.solver import Solver

video_shape = (16,240,320)
batch_size = 1
seed = 1234

# Baseline A: Small CNN
smallnet = ConvNet3D("small-net",video_shape,batch_size,seed=seed)
smallnet.add_train_data("data/traindb.lmdb")
smallnet.add_val_data("data/valdb.lmdb")
smallnet.add_conv_layer("conv1",(5,7,7),8)
smallnet.add_pool_layer("pool1",(2,2,2))
smallnet.add_conv_layer("conv2",(3,3,3),16)
smallnet.add_pool_layer("pool2",(2,2,2))
smallnet.add_conv_layer("conv3",(3,3,3),16)
smallnet.add_pool_layer("pool3",(2,2,2))
smallnet.add_conv_layer("conv4",(3,3,3),16)
smallnet.add_pool_layer("pool4",(2,2,2))
smallnet.add_fc_layer("fc1",128,0.5)
smallnet.add_softmax_layer("softmax",101)

reg = 5e-3
reg_params = {
    "conv1_W": reg,
    "conv2_W": reg,
    "conv3_W": reg,
    "conv4_W": reg,
    "fc1_W": reg,
    "softmax_W": reg}

snapshot_params = {
    "dir": "models/smallnet",
    "rate": 4000}

opt_params = {
    "method": "momentum",
    "initial": 0.5,
    "final": 0.9,
    "step": 0.1, # per epoch
    "lr_decay": 0.95,
    "lr_base": 1e-5}

solver = Solver(smallnet,reg_params,opt_params)
solver.train(40000,snapshot_params,validate_rate=4000,loss_rate=1)
