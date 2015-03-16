# -*- coding: utf-8 -*-
"""
Regularization models for video classification.
Created on Sat Feb 28 14:02:45 2015

@author: Kevin Chavez
"""
import time
import theano
import theano.sparse
import theano.tensor as T
import theano.tensor.slinalg
import numpy as np
from numpy.random import RandomState
import scipy.sparse
#from src.tests.optflow_reg import display

BGR_WEIGHTS = [0.2989,0.5870,0.1140]

def flat_join(*args):
    # Reduce all inputs to vector 
    #(https://groups.google.com/forum/#!msg/theano-users/A-RcItll8eA/z8eZyrTwX9wJ)
    join_args = []
    for i,arg in enumerate(args):
        if arg.type.ndim: # it is not a scalar
            join_args.append(arg.flatten())
        else:
            join_args.append( T.shape_padleft(arg))
            # join them into a vector
    return T.join(0, *join_args)

def brightness_constancy(kernel,Vx,Vy,bgr=True):
    # Collapse channels    
    if kernel.ndim == 5:
        if bgr:
            kernel = 0.2989*kernel[:,2] + 0.5870*kernel[:,1] + 0.1140*kernel[:,0]
        else:
            kernel = np.mean(kernel,axis=1)

    N, TT, HH, WW = kernel.shape
    
    Dx = np.diag([1.]*(WW-1),1) + np.diag([-1.]*(WW-1),-1)
    Dy = np.diag([1.]*(HH-1),1) + np.diag([-1.]*(HH-1),-1)
    
    B = np.tensordot(kernel, Dx.T,axes=[(3,),(0,)])*Vx + \
        np.tensordot(Dy,kernel, axes=[(1,),(2,)]).transpose(1,2,0,3) * Vy + \
        np.concatenate((kernel[:,1:] - kernel[:,:-1],-kernel[:,[TT-1]]),axis=1)

    return B
    
def smoothness(Vx,Vy):
    N, TT, HH, WW = Vx.shape
    Dx = np.diag([1.]*(WW-1),1) + np.diag([-1.]*(WW-1),-1)
    Dy = np.diag([1.]*(HH-1),1) + np.diag([-1.]*(HH-1),-1)
    Dt = np.diag([1.]*(TT-1),1) + np.diag([-1.]*(TT),0)
    
    Vxdx = np.tensordot(Vx,Dx.T,axes=[(3,),(0,)]) #Vx[:,:,:,2:] - Vx[:,:,:,0:-2]
    Vxdy = np.tensordot(Dy,Vx, axes=[(1,),(2,)]) #Vx[:,:,2:,:] - Vx[:,:,0:-2,:]
    Vxdt = np.tensordot(Dt,Vx, axes=[(1,),(1,)]) #Vx[:,2:,:,:] - Vx[:,0:-2,:,:]

    Vydx = np.tensordot(Vy,Dx.T,axes=[(3,),(0,)]) #Vx[:,:,:,2:] - Vx[:,:,:,0:-2]
    Vydy = np.tensordot(Dy,Vy, axes=[(1,),(2,)]) #Vx[:,:,2:,:] - Vx[:,:,0:-2,:]
    Vydt = np.tensordot(Dt,Vy, axes=[(1,),(1,)])
    
    penalty = np.sum(Vxdx*Vxdx) + np.sum(Vxdy*Vxdy) + \
              np.sum(Vxdt*Vxdt) + np.sum(Vydx*Vydx) + \
              np.sum(Vydy*Vydy) + np.sum(Vydt*Vydt)
              
    return 0.5*penalty

def cost(filt,Vx,Vy,gamma):
    B = brightness_constancy(filt,Vx,Vy) 
    return (0.5*np.sum(B*B) + gamma * smoothness(Vx,Vy)) / filt.shape[0]
    
def grad(filt,Vx,Vy,gamma,bgr=True):
    N, TT, HH, WW = Vx.shape
    Dx = np.diag([1.]*(WW-1),1) + np.diag([-1.]*(WW-1),-1)
    Dy = np.diag([1.]*(HH-1),1) + np.diag([-1.]*(HH-1),-1)
    Dt = np.diag([1.]*(TT-1),1) + np.diag([-1.]*(TT),0)    
    DxDx = Dx.T.dot(Dx)
    DyDy = Dy.T.dot(Dy)
    DtDt = Dt.T.dot(Dt)

    if bgr:
        W = 0.2989*filt[:,2] + 0.5870*filt[:,1] + 0.1140*filt[:,0]
    else:
        W = np.mean(filt,axis=1)    
     
    B = brightness_constancy(filt,Vx,Vy) 
    dVx = B * np.tensordot(W, Dx.T,axes=[(3,),(0,)]) + \
            gamma*(np.tensordot(Vx, DxDx,axes=[(3,),(0,)]) + \
                   np.tensordot(DyDy,Vx, axes=[(1,),(2,)]).transpose(1,2,0,3) + \
                   np.tensordot(Vx,DtDt,axes=[(1,),(0,)]).transpose(0,3,1,2) )
                   # 2Vxt - Vxt+1 - Vxt-1
    dVy = B* np.tensordot(Dy,W, axes=[(1,),(2,)]).transpose(1,2,0,3) + \
            gamma*(np.tensordot(Vy, DxDx,axes=[(3,),(0,)]) + \
                   np.tensordot(DyDy,Vy, axes=[(1,),(2,)]).transpose(1,2,0,3) +\
                   np.tensordot(Vy,DtDt,axes=[(1,),(0,)]).transpose(0,3,1,2))
                   
    return dVx / Vx.size, dVy / Vy.size
    
def optflow_regularizer_fast(kernel,bgr=True,gamma=1.0):
    """ Scipy-based implementation of the optical flow regularizer.
    
    Args:
        kernel: 5D tensor of filter weights
        bgr: boolean indicating if this filter is applied to raw pixel data
        gamma: relative importance of the smoothness criterion in the optical
               flow computation
               
    Returns:
        A tuple containing:
        
            (regularization cost, gradient wrt kernel, (Vx_star, Vy_star))
        
        The last element can usually be disregarded--it's the velocity vector
        field that achieves the minimum of the optical flow loss function, but
        this is not needed in the larger ConvNet framework.
    """
    N, C, TT, HH, WW = kernel.shape
    
    # Flatten kernel along channels
    if bgr:
        kernel = 0.2989*kernel[:,2] + 0.5870*kernel[:,1] + 0.1140*kernel[:,0]
    else:
        kernel = np.mean(kernel,axis=1)
    
    # Derivative approximation matrices
    Dx = np.diag([1.]*(WW-1),1) + np.diag([-1.]*(WW-1),-1)
    DxDx = Dx.T.dot(Dx)
    Dy = np.diag([1.]*(HH-1),1) + np.diag([-1.]*(HH-1),-1)
    DyDy = Dy.T.dot(Dy)
        
    # Derivatives of kernel with respect to x, y, and t
    Wx = np.tensordot(kernel, Dx.T,axes=[(3,),(0,)])
    Wy = np.tensordot(Dy,kernel, axes=[(1,),(2,)]).transpose(1,2,0,3)
    Wt = np.concatenate((kernel[:,1:] - kernel[:,:-1],-kernel[:,[TT-1]]),axis=1)
   
    # Create gradients assuming initial point is Vx = Vy = 0
    gVx = Wt*Wx
    gVy = Wt*Wy
    
    Hxx_diag_blocks = []
    Hyy_diag_blocks = []
    Hxy_diag_blocks = []
    
    # Precompute some values
    upper_band = np.diag([-gamma]*HH*WW*(TT-1),k=HH*WW)
    lower_band = np.diag([-gamma]*HH*WW*(TT-1),k=-HH*WW)

    gamma_band = np.zeros((TT*HH*WW,TT*HH*WW))
    for j,ell in zip(*np.nonzero(DxDx)):
        for i in xrange(HH):
            idx1 = [i*HH + ell + t*HH*WW for t in range(TT)]                    
            idx2 = [i*HH + j + t*HH*WW for t in range(TT)]
            gamma_band[idx1, idx2] += gamma*DxDx[j,ell]
            
    for k,i in zip(*np.nonzero(DyDy)):
        for j in xrange(WW):
            idx1 = [k*HH + j + t*HH*WW for t in range(TT)]                     
            idx2 = [i*HH + j +t*HH*WW for t in range(TT)]
            gamma_band[idx1, idx2] += gamma*DyDy[k,i]

    for n in range(N):
        Hnxx_diag_blocks = []
        Hnyy_diag_blocks = []
        for t in range(TT):
            # Compute sparse Hessian
            Hxx = np.zeros((HH*WW,HH*WW))
            Hyy = np.zeros((HH*WW,HH*WW))
            Hxy = np.zeros((HH*WW,HH*WW))
            
            np.fill_diagonal(Hxx,np.ravel(Wx[n,t]**2) + gamma*(2.0 if t > 0 else 1.0))
            np.fill_diagonal(Hyy,np.ravel(Wy[n,t]**2) + gamma*(2.0 if t > 0 else 1.0))
                
            np.fill_diagonal(Hxy,np.ravel(Wy[n,t]*Wx[n,t]))
            Hnxx_diag_blocks.append(Hxx)
            Hnyy_diag_blocks.append(Hyy)
            #Hxy_diag_blocks.append(Hxy)
            
        Hnxx = scipy.linalg.block_diag(*Hnxx_diag_blocks) + gamma_band
        Hnyy = scipy.linalg.block_diag(*Hnyy_diag_blocks) + gamma_band
        Hnxy = np.diag(np.ravel(Wy[n]*Wx[n]))

        # Note: There are also two diagonal bands of -1's for the time coupling 
        # between frames.
        Hnxx += upper_band
        Hnxx += lower_band
        Hnyy += upper_band
        Hnyy += lower_band
        
        Hxx_diag_blocks.append(Hnxx)
        Hyy_diag_blocks.append(Hnyy)
        Hxy_diag_blocks.append(Hnxy)
        
                
    # Construct full hessian
    hess_xx = scipy.sparse.block_diag(Hxx_diag_blocks)
    hess_yy = scipy.sparse.block_diag(Hyy_diag_blocks)
    hess_xy = scipy.sparse.block_diag(Hxy_diag_blocks)

    hess = scipy.sparse.bmat([[hess_xx, hess_xy],[hess_xy,hess_yy]])

    hess = hess.tocsr()
    g = np.hstack([gVx.ravel(),gVy.ravel()])
    #tic = time.time()
    x_star = -scipy.sparse.linalg.spsolve(hess,g)
    #toc = time.time()
    #print "Solving sparse matrix equation took %0.6f seconds" % (toc - tic)
    Vx_star = x_star[0:kernel.size].reshape(kernel.shape)
    Vy_star = x_star[kernel.size:].reshape(kernel.shape)
    
    # Now compute the actual cost
    reg_cost = cost(kernel,Vx_star,Vy_star,gamma)
    B = brightness_constancy(kernel,Vx_star,Vy_star)
    Vxdx = np.tensordot(Vx_star,Dx,axes=[(3,),(0,)])
    Vydy = np.tensordot(Dy.T,Vy_star, axes=[(1,),(2,)]).transpose(1,2,0,3)
    B_shifted = np.concatenate((np.zeros_like(B[:,[0]]), B[:,0:-1]),axis=1)
    grad = (B * (Vxdx + Vydy - 1) + B_shifted).reshape((N,1,TT,HH,WW))
    
    if bgr:
        grad = np.reshape(BGR_WEIGHTS,(1,3,1,1,1))*grad / N
    else:
        grad = np.repeat(grad,C,axis=1) / N
        
    return reg_cost, grad, (Vx_star, Vy_star) 
    
def l2_regularizer(kernel):
    """ Creates theano expressions to evaluate regularization and its gradient.
    """
    updates = []
    info = None
    return 0.5*T.sum(kernel*kernel), kernel, updates, info
    
def test():
    import matplotlib.pyplot as plt
    
    N, C, TT, H, W = 32, 3, 5, 7, 7
    gamma = 20.
    rng = np.random.RandomState(seed=1)
    noise = 20.*rng.randn(N,C,TT,H,W)
    filt = noise #theano.shared(noise.astype(theano.config.floatX),borrow=True)
    tic = time.time()
    vx,vy = optflow_regularizer_fast(filt,filt.shape,gamma=gamma)
    toc = time.time()
    print "Full operation took %0.6f seconds" % (toc - tic)
    opt_cost = cost(filt,vx,vy,gamma)
    print "Optimal cost:", opt_cost
    
    # Check random jitter:
    plt.close('all')
    plt.figure()
    #display(filt[0],vx[0],vy[0])
    num_test = 5
    for _ in xrange(num_test):
        sx = 0.01*np.random.randn(*vx.shape)
        sy = 0.01*np.random.randn(*vy.shape)
        sx /= np.linalg.norm(sx)
        sy /= np.linalg.norm(sy)
        # Plot line along direction
        alpha = np.linspace(-100,100,100)
        costs = []
        for a in alpha:
            newcost = cost(filt,vx+a*sx,vy+a*sy,gamma)
            costs.append(newcost)
        
        plt.plot(alpha,costs)
        
    plt.show()
    #loss, grad, (Vx, Vy) = optical_flow_regularizer(filt,(N,C,TT,H,W),gamma=10.)

    # Compile function     
    return filt, vx, vy

def test_gradient():
    import matplotlib.pyplot as plt
    import cProfile, pstats, StringIO
    N, C, TT, H, W = 16, 3, 7, 9, 9
    gamma = 2.
    rng = np.random.RandomState(seed=5)
    filt = 20.*rng.randn(N,C,TT,H,W)
  
    plt.close('all')  
    # Randomly initialize Vx and Vy
    Vx = np.random.randn(N,TT,H,W)
    Vy = np.random.randn(N,TT,H,W)

    # Do gradient descent. We know its slow, but it will be a sanity check for 
    # expression for the gradient
    costs = []
    for i in range(0):
        gVx, gVy = grad(filt,Vx,Vy,gamma)
        Vx -= 1e-1*gVx
        Vy -= 1e-1*gVy
        new_cost = cost(filt,Vx,Vy,gamma)
        norms = np.linalg.norm(gVx)/ gVx.size, np.linalg.norm(gVy) / gVy.size
        if norms[0] < 1e-5 and norms[1] < 1e-5:
            break
        print norms
        costs.append(new_cost)
        
    plt.plot(costs)
    print "Final cost obtained by gradient descent:", costs[-1]

    # Try with the Hessian, profile computation
    tic = time.time()
    pr = cProfile.Profile()
    pr.enable()
    _, _, (Vx_star2, Vy_star2) = optflow_regularizer_fast(filt,filt.shape,gamma=gamma)
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()    
    toc = time.time()
    print "Fast ran in %0.6f seconds" % (toc - tic)
    
def test_regularization_step():
    import matplotlib.pyplot as plt
    N, C, TT, H, W = 16, 3, 7, 7, 7
    gamma = 2.
    rng = np.random.RandomState(seed=5)
    
    trials = 4
    costs = []
    norms = []
    lrs = np.logspace(-4,0,trials)
    orig_filt = 1.*rng.randn(N,C,TT,H,W)
    
    plt.close('all')
    for t in xrange(trials):
        filt = np.copy(orig_filt)    
        trial_cost = []
        trial_norms = []
        for step in range(100):
            c, g, (Vx, Vy) = optflow_regularizer_fast(filt,filt.shape,gamma=gamma)
            trial_cost.append(c)
            trial_norms.append(np.linalg.norm(filt))
            print "|grad| =", np.linalg.norm(g)
            print "|filt| =", np.linalg.norm(filt)
            print "cost   =", c
            filt -= lrs[t]*g
            
        costs.append(trial_cost)
        norms.append(trial_norms)
        plt.figure(1)
        plt.plot(trial_cost)
        plt.figure(2)
        plt.plot(trial_norms)
    
    plt.figure(1)
    plt.xlabel('iteration')
    plt.ylabel('regularization penalty')
    plt.legend(['step size = %0.3e' % x for x in lrs])
    plt.title('Comparing Step Sizes')
    plt.figure(2)
    plt.xlabel('iteration')
    plt.ylabel('Frobenius norm')
    plt.title('Weight Decay')
    
def characterize_hyperparameters():
    """ Explore the effects of the hyperparameter gamma and the weight given
    to this regularization term as part of a larger optimization problem."""
    pass
    
    
if __name__ == "__main__":
    #filt, vx, vy = test()
    #test_gradient()
    test_regularization_step()
