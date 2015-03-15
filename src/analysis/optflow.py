# -*- coding: utf-8 -*-
"""
Analysis of Optflow Regularizer
Created on Mon Mar  9 22:40:34 2015

@author: Kevin Chavez
"""
import time
import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from src.convnet3d.regularization import *

APPROXIMATE_MEAN = 127.0

def display(filt, Vx, Vy,std=30,mean=127.0):
    plt.figure()
    TT = filt.shape[1]
    HH, WW = filt.shape[2:]
    frames = [std*filt[:,i].transpose(1,2,0) + mean for i in xrange(filt.shape[1])]
    for t, frame in zip(range(TT),frames):
        frame = np.clip(frame,0,255)
        plt.figure(1)
        plt.subplot(2,TT,TT+t+1)
        I,J = np.meshgrid(range(HH),range(WW),indexing='ij')
        Q = plt.quiver(Vx[t],Vy[t])
        qk = plt.quiverkey(Q, 0.5, 0.92, 0, r'', labelpos='W',
                       fontproperties={'weight': 'bold'})
        #plt.streamplot(I[:,0], -J[0], Vx[t], -Vy[t])            
        plt.axis('image')
        plt.tick_params(\
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')
        plt.tick_params(\
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left='off',      # ticks along the bottom edge are off
            right='off',         # ticks along the top edge are off
            labelleft='off')
            
        plt.subplot(2,TT,t+1)
        #plt.axis('image')
        plt.tick_params(\
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')
        plt.tick_params(\
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left='off',      # ticks along the bottom edge are off
            right='off',         # ticks along the top edge are off
            labelleft='off')
        plt.imshow(frame.astype('uint8'),interpolation='none')
    
    #plt.show()
    
def computation_time(num_trials):
    sizes = [(7,7,7),(5,7,7),(3,7,7),(3,5,5),(5,5,5),(3,3,3),(5,9,9),(4,11,11)]
    x = [np.prod(s) for s in sizes for n in range(num_trials)]
    data = []
    for s in sizes:
        for n in range(num_trials):
            W = np.random.randn(1,3,*s)
            tic = time.time()
            loss, grad, mat_time = optflow_regularizer_fast(W,gamma=1.0)
            toc = time.time()
            data.append(mat_time)
            
    comp = {'size': np.array(x,dtype=np.int), 'time': np.array(data)}
    sns.lmplot("size", "time", pandas.DataFrame(comp),x_estimator=np.mean)
    plt.ylabel("time (seconds)")    
    plt.show()
    return x, data
            
def compare_costs():
    # Part 1: Analyze relative regularization cost of moving bar vs noise
    N, C, TT, HH, WW = (1,3,7,7,7)
    gamma = 1.0
    
    rng = np.random.RandomState(1234)
    random_filter = 1.83*rng.randn(N,C,TT,HH,WW)
    print "Norm of random:", np.linalg.norm(random_filter)
    
    # Hand constructed filter
    moving_blob = np.zeros((N,C,TT,HH,WW))
    for t in xrange(TT):
        moving_blob[:,:,t,2:5,t:t+2] = 5.0
        
    # Add noise
    moving_blob += 0.5*rng.randn(N,C,TT,HH,WW)
    print "Norm of moving blob:", np.linalg.norm(moving_blob)
    
    plt.close('all')
    cost, grad, (vx, vy) = optflow_regularizer_fast(random_filter,gamma=gamma)
    print cost, smoothness(vx,vy) / N
    display(random_filter[0],vx[0],vy[0],std=25)
    
    cost, grad, (vx, vy) = optflow_regularizer_fast(moving_blob,gamma=gamma)
    print cost, smoothness(vx,vy) / N
    display(moving_blob[0],vx[0],vy[0],std=25.)

def norm_vs_cost():
    # Random filters norm vs cost
    normal_norms = []
    normal = []
    uniform_norms = []
    uniform = []
    laplace_norms = []
    laplace = []
    for n in xrange(200):
        filt = rng.uniform(0,20)*rng.randn(N,C,TT,HH,WW)
        normal_norms.append(np.linalg.norm(filt))
        cost, grad, _ = optflow_regularizer_fast(filt,gamma=gamma)
        normal.append(cost)
    
        filt = rng.uniform(0,30)*rng.rand(N,C,TT,HH,WW)
        uniform_norms.append(np.linalg.norm(filt))
        cost, grad, _ = optflow_regularizer_fast(filt,gamma=gamma)
        uniform.append(cost)
        
        filt = rng.laplace(loc=0.0,scale=rng.uniform(0,15),size=(N,C,TT,HH,WW))
        laplace_norms.append(np.linalg.norm(filt))
        cost, grad, _ = optflow_regularizer_fast(filt,gamma=gamma)
        laplace.append(cost)
    
    plt.figure()    
    plt.scatter(normal_norms,normal,c='b',alpha=0.6,edgecolors='none')
    plt.scatter(uniform_norms,uniform,c='r',alpha=0.6,edgecolors='none')
    plt.scatter(laplace_norms,laplace,c='g',alpha=0.6,edgecolors='none')
    plt.legend(['Normal','Uniform','Laplace'])
    plt.xlabel(r'filter $L_2$ norm')
    plt.ylabel('regularization loss')
    plt.show()

def norm_after_step():
    # Random filters, norm decay after taking varying sized steps along gradient
    rel_diff = []
    rel_cost = []
    steps = []
    for n in xrange(400):
        filt = rng.randn(N,C,TT,HH,WW)
        step_size = 10**rng.uniform(-2,0)
        cost, grad, _ = optflow_regularizer_fast(filt,gamma=gamma)
        orig_norm = np.linalg.norm(filt)
        new_cost, _, _ = optflow_regularizer_fast(filt - step_size*grad,gamma=gamma)
        new_norm = np.linalg.norm(filt - step_size*grad)
        #print "Step size:", step_size
        #print "Cost %0.2f -> %0.2f" % (cost, new_cost)
        #print "Norm %0.2f -> %0.2f" % (orig_norm, new_norm)
        steps.append(step_size)
        rel_diff.append((new_norm - orig_norm)/orig_norm * 100)
        rel_cost.append((new_cost - cost)/cost * 100)
        
    #plt.scatter(steps,rel_diff)
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(steps , rel_diff, c='blue', alpha=0.6, edgecolors='none')
    ax.scatter(steps , rel_cost, c='red', alpha=0.6, edgecolors='none')
    #ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlabel('step size')
    plt.ylabel('% relative change')
    plt.legend(['L2 norm', 'Regularization loss'],loc=3)
    plt.show()