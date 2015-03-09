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
    return (0.5*np.sum(B*B) + gamma * smoothness(Vx,Vy)) / filt.size * filt.shape[1]
    
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
    
def optflow_regularizer_fast(kernel,kernel_shape,bgr=True,gamma=0.16):
    N, C, TT, HH, WW = kernel_shape
    # Random initialization. Only matters to make sure problem is well-
    # conditioned.
    #Vxt = np.zeros((HH,WW)) # for all t
    #Vyt = np.zeros((HH,WW)) # for all t
    
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
            idx1 = [i*HH + ell + t*HH*WW for t in range(TT)] #np.ravel_multi_index((i,ell),(HH,WW))                        
            idx2 = [i*HH + j + t*HH*WW for t in range(TT)] #np.ravel_multi_index((i,j),(HH,WW))
            gamma_band[idx1, idx2] += gamma*DxDx[j,ell]
            
    for k,i in zip(*np.nonzero(DyDy)):
        for j in xrange(WW):
            idx1 = [k*HH + j + t*HH*WW for t in range(TT)] #np.ravel_multi_index((k,j),(HH,WW))                        
            idx2 = [i*HH + j +t*HH*WW for t in range(TT)] #np.ravel_multi_index((i,j),(HH,WW))
            gamma_band[idx1, idx2] += gamma*DyDy[k,i]

    for n in range(N):
        Hnxx_diag_blocks = []
        Hnyy_diag_blocks = []
        for t in range(TT):
            # Compute sparse Hessian
            Hxx = np.zeros((HH*WW,HH*WW))
            Hyy = np.zeros((HH*WW,HH*WW))
            Hxy = np.zeros((HH*WW,HH*WW))
            
            # Hint: Use np.ravel_multi_index
            #start_idx = np.ravel_multi_index((n,c,t,0,0),kernel.shape)
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
    tic = time.time()
    x_star = -scipy.sparse.linalg.spsolve(hess,g)
    toc = time.time()
    print "Solving sparse matrix equation took %0.6f seconds" % (toc - tic)
    Vx_star = x_star[0:kernel.size].reshape(kernel.shape)
    Vy_star = x_star[kernel.size:].reshape(kernel.shape)
    
    # Now compute the actual cost
    reg_cost = cost(kernel,Vx_star,Vy_star,gamma)
    B = brightness_constancy(kernel,Vx_star,Vy_star)
    Vxdx = np.tensordot(Vx_star,Dx,axes=[(3,),(0,)])
    Vydy = np.tensordot(Dy.T,Vy_star, axes=[(1,),(2,)]).transpose(1,2,0,3)
    B_shifted = np.concatenate((np.zeros_like(B[:,[0]]), B[:,0:-1]),axis=1)
    grad = (B * (Vxdx + Vydy - 1) + B_shifted).reshape((N,1,TT,HH,WW))
    
    #grad = (Wx * Vx_star**2 + Wy * Vy_star**2).reshape((N,1,TT,HH,WW))
    if bgr:
        grad = np.reshape(BGR_WEIGHTS,(1,3,1,1,1))*grad
    else:
        grad = np.repeat(grad,C,axis=1)
        
    return reg_cost, grad, (Vx_star, Vy_star) 
                   
# Manual Hessian version
def optflow_regularizer_slow(kernel,kernel_shape,bgr=True,gamma=0.16):
    N, C, TT, HH, WW = kernel_shape
    # Random initialization. Only matters to make sure problem is well-
    # conditioned.
    #Vxt = np.zeros((HH,WW)) # for all t
    #Vyt = np.zeros((HH,WW)) # for all t
    
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
    
    gVx = np.empty_like(kernel)
    gVy = np.empty_like(kernel)
    
    Hxx_diag_blocks = []
    Hyy_diag_blocks = []
    Hxy_diag_blocks = []
    for n in range(N):
        Hnxx_diag_blocks = []
        Hnyy_diag_blocks = []
        for t in range(TT):
            Wt = kernel[n,t]
            Wt1 = kernel[n,t+1] if t < TT-1 else np.zeros_like(Wt)
            # Create gradients
#                B = (Wt.dot(Dx.T) * Vxt + Dy.dot(Wt) * Vyt + Wt1 - Wt)
#                dVxt = B * Wt.dot(Dx.T) + gamma*(Vxt.dot(DxDx) + DyDy.dot(Vxt)
#                                                 + 0) #
#                dVyt = B * Dy.dot(Wt) + gamma*(Vyt.dot(DxDx) + DyDy.dot(Vyt)
#                                                + 0)
            # Create gradients assuming initial point is Vx = Vy = 0
            B = Wt1 - Wt
            dVxt = B * Wt.dot(Dx.T)
            dVyt = B * Dy.dot(Wt)
            
            gVx[n,t] = dVxt
            gVy[n,t] = dVyt
            
            # Compute sparse Hessian
            Hxx = scipy.sparse.lil_matrix((HH*WW,HH*WW))
            Hyy = scipy.sparse.lil_matrix((HH*WW,HH*WW))
            Hxy = scipy.sparse.lil_matrix((HH*WW,HH*WW))
            
            # Hint: Use np.ravel_multi_index
            #start_idx = np.ravel_multi_index((n,c,t,0,0),kernel.shape)
            Hxx.setdiag(np.ravel(Wt.dot(Dx.T)**2) + gamma*(2.0 if t > 0 else 1.0))
            Hyy.setdiag(np.ravel(Dy.dot(Wt)**2) + gamma*(2.0 if t > 0 else 1.0))
            
            for j,ell in zip(*np.nonzero(DxDx)):
                for i in xrange(HH):
                    idx1 = np.ravel_multi_index((i,ell),(HH,WW))                        
                    idx2 = np.ravel_multi_index((i,j),(HH,WW))
                    Hxx[idx1, idx2] += gamma*DxDx[j,ell]
                    Hyy[idx1, idx2] += gamma*DxDx[j,ell]
                    
            for k,i in zip(*np.nonzero(DyDy)):
                for j in xrange(WW):
                    idx1 = np.ravel_multi_index((k,j),(HH,WW))                        
                    idx2 = np.ravel_multi_index((i,j),(HH,WW))
                    Hxx[idx1, idx2] += gamma*DyDy[k,i]
                    Hyy[idx1, idx2] += gamma*DyDy[k,i]
                    
            Hxy.setdiag(np.ravel(Dy.dot(Wt)*Wt.dot(Dx.T)))
            Hnxx_diag_blocks.append(Hxx)
            Hnyy_diag_blocks.append(Hyy)
            Hxy_diag_blocks.append(Hxy)
            
        Hnxx = scipy.sparse.block_diag(Hnxx_diag_blocks,format='lil')
        Hnyy = scipy.sparse.block_diag(Hnyy_diag_blocks,format='lil')
 
        # Note: There are also two diagonal bands of -1's for the time coupling 
        # between frames.
        Hnxx.setdiag([-gamma]*HH*WW*(TT-1),k=HH*WW)
        Hnxx.setdiag([-gamma]*HH*WW*(TT-1),k=-HH*WW)
        Hnyy.setdiag([-gamma]*HH*WW*(TT-1),k=HH*WW)
        Hnyy.setdiag([-gamma]*HH*WW*(TT-1),k=-HH*WW)
        
        Hxx_diag_blocks.append(Hnxx)
        Hyy_diag_blocks.append(Hnyy)
        
                
    # Construct full hessian
    hess_xx = scipy.sparse.block_diag(Hxx_diag_blocks)
    hess_yy = scipy.sparse.block_diag(Hyy_diag_blocks)
    hess_xy = scipy.sparse.block_diag(Hxy_diag_blocks)
    hess = scipy.sparse.bmat([[hess_xx, hess_xy],[hess_xy,hess_yy]])

    hess = hess.tocsr()
    g = np.hstack([gVx.ravel(),gVy.ravel()])
    print g.dtype
    print hess.dtype
    tic = time.time()
    x_star = -scipy.sparse.linalg.spsolve(hess,g)
    toc = time.time()
    print "Solving sparse matrix equation took %0.6f seconds" % (toc - tic)
    Vx_star = x_star[0:kernel.size].reshape(kernel.shape)
    Vy_star = x_star[kernel.size:].reshape(kernel.shape)
    return Vx_star, Vy_star
                        
def optical_flow_regularizer(kernel,kernel_shape,
                             rng=RandomState(1234),bgr=True,gamma=0.16,
                             smoothness='L2'):
    """ Creates theano expressions to evaluate regularization and its gradient
    
    Args:
        kernel - theano.tensor (5D)
        
    Returns:
        A tuple of theano expressions which compute the regularization penalty
        and its gradient.
        
        See technical report by Kevin Chavez for details.
    """
    # Create holders for optimal optical flow vectors
    N, C, TT, HH, WW = kernel_shape
    Vx = theano.shared(rng.randn(N,TT,HH,WW).astype(theano.config.floatX),
                       borrow=True)
    Vy = theano.shared(rng.randn(N,TT,HH,WW).astype(theano.config.floatX),
                       borrow=True)
    
    # Formulate optical flow cost
    # If bgr, convert to grayscale (first layer of network)
    # Note: kernel is a 5D tensor, W is a 4D tensor (n_filt,frame,height,width)
    if bgr:
        W = 0.2989*kernel[:,2] + 0.5870*kernel[:,1] + 0.1140*kernel[:,0]
    else:
        # Otherwise take average across channels (untested)
        W = T.mean(kernel,axis=1)
        
#    gx_kernel = [[1, 0, -1],
#                 [2, 0, -2],
#                 [1, 0, -1]]
#                 
#    gy_kernel = [[1, 2, 1],
#                 [0, 0, 0],
#                 [-1, -2, -1]]
    #dX = T.extra_ops.diff(W,n=1,axis=-1)
    dX = T.concatenate((W[:,:,:,1:],T.zeros((N,TT,HH,1))),axis=3) - \
         T.concatenate((T.zeros((N,TT,HH,1)),W[:,:,:,0:WW-1]),axis=3)
    
    #dY = T.extra_ops.diff(W,n=1,axis=-2)
    dY = T.concatenate((W[:,:,1:,:],T.zeros((N,TT,1,WW))),axis=2) - \
         T.concatenate((T.zeros((N,TT,1,WW)),W[:,:,0:HH-1,:]),axis=2)
    #dT = T.extra_ops.diff(W,n=1,axis=-3)
    dT = T.concatenate((W[:,1:,:,:],T.zeros((N,1,HH,WW))),axis=1) - \
         T.concatenate((T.zeros((N,1,HH,WW)),W[:,0:TT-1,:,:]),axis=1) 
    
    # Only use 'valid' region where we can compute dX, dY and dT to compute
    # brightness constancy violation
    bc_diff = dX*Vx + dY*Vy + dT
    bc_penalty = T.sum(bc_diff * bc_diff)
    
    # Smoothness constraint
    Vxdx = Vx[:,:,:,1:] - Vx[:,:,:,0:-1]
    Vxdy = Vx[:,:,1:,:] - Vx[:,:,0:-1,:]
    Vxdt = Vx[:,1:,:,:] - Vx[:,0:-1,:,:]

    Vydx = Vy[:,:,:,1:] - Vy[:,:,:,0:-1]
    Vydy = Vy[:,:,1:,:] - Vy[:,:,0:-1,:]
    Vydt = Vy[:,1:,:,:] - Vy[:,0:-1,:,:]
    
    smoothness_penalty =  T.sum(Vxdx*Vxdx) + \
                         T.sum(Vxdy*Vxdy) + \
                         T.sum(Vxdt*Vxdt) + \
                         T.sum(Vydx*Vydx) + \
                         T.sum(Vydy*Vydy) + \
                         T.sum(Vydt*Vydt)

    cost = 0.5*(bc_penalty + gamma*smoothness_penalty)
    
    # Gradients for Vx and Vy, W constant
    gVx = T.grad(cost,Vx,consider_constant=[W])
    gVy = T.grad(cost,Vy,consider_constant=[W])
    
    # Hessian for Vx, Vy. Again W is constant.
    # Note: We can save a *substantial* amount of time by expressing the 
    # Hessian in its sparse format. It has O(n) non-zero terms, or more
    # precisely less than 9n non-zero terms. It's also symmetric, so only 5n
    # of those are unique. (n is the number of elements in [Vx; Vy])
    g_all = flat_join(gVx,gVy)
    H,updates = theano.scan( 
                    lambda i,g_all,Vx,Vy: flat_join(T.grad(g_all[i],Vx),
                                                    T.grad(g_all[i], Vy)), 
                    sequences = T.arange(g_all.shape[0]), 
                    non_sequences = [g_all, Vx, Vy])
    
    # Sparse solve would be fantastic. In the mean-time Cholesky factorization
    # reduces running time by a factor of about 6
#    L = T.slinalg.cholesky(H)
#    lower_solve = T.slinalg.Solve(A_structure="lower_triangular")
#    upper_solve = T.slinalg.Solve(A_structure="upper_triangular")
#    b = lower_solve(L,g_all)
#    # Can solve for optimum in a single newton step
#    x_star = flat_join(Vx,Vy) - upper_solve(T.transpose(L),b)
    Hsparse = theano.sparse.csr_from_dense(H)
        
    x_star = flat_join(Vx,Vy) - T.slinalg.spsolve(Hsparse,g_all)
                                    
    Vx_star = x_star[0:Vx.size].reshape(Vx.shape)
    Vy_star = x_star[Vx.size:].reshape(Vy.shape)
    
    # Now we can compute the minimum optical flow cost, which is the
    # regularization loss
        
    # Brightness constancy violation
    bc_diff_star = dX*Vx_star + dY*Vy_star + dT
    opt_bc_penalty = T.sum(bc_diff_star * bc_diff_star)
    
    # Smoothness constraint
    Vxdx_star = Vx_star[:,:,:,1:] - Vx_star[:,:,:,0:-1]
    Vxdy_star = Vx_star[:,:,1:,:] - Vx_star[:,:,0:-1,:]
    Vxdt_star = Vx_star[:,1:,:,:] - Vx_star[:,0:-1,:,:]

    Vydx_star = Vy_star[:,:,:,1:] - Vy_star[:,:,:,0:-1]
    Vydy_star = Vy_star[:,:,1:,:] - Vy_star[:,:,0:-1,:]
    Vydt_star = Vy_star[:,1:,:,:] - Vy_star[:,0:-1,:,:]
    
    opt_smoothness_penalty = T.sum(Vxdx_star*Vxdx_star) + \
                             T.sum(Vxdy_star*Vxdy_star) + \
                             T.sum(Vxdt_star*Vxdt_star) + \
                             T.sum(Vydx_star*Vydx_star) + \
                             T.sum(Vydy_star*Vydy_star) + \
                             T.sum(Vydt_star*Vydt_star)
    
    reg_loss = 0.5*(opt_bc_penalty + gamma*opt_smoothness_penalty)
        
    # Gradient with respect to kernel, consider Vx, Vy constant.
    # IMPORTANT: The regularization term is the MINIMUM achievable loss given
    # a particular kernel. However, if Vx and Vy currently store the optimal
    # optical flow vector field, the gradient of the minimum loss with respect
    # to the kernel is identical to the gradient of the loss with respect to 
    # the kernel, EVALUATED at (Vx, Vy). In other words, this gradient is only
    # valid as long as Vx and Vy are kept updated.
    g_kernel = T.grad(reg_loss,kernel,consider_constant=[Vx_star,Vy_star])
    
    # Alternatively, it's possible Theano's expression graph can efficiently
    # compute the gradient of reg_loss with respect to the kernel. But I 
    # highly doubt it
    #g_kernel2 = T.grad(reg_loss,kernel,consider_constant=[Vx_star,Vy_star])
    updates = [(Vx, Vx_star), (Vy, Vy_star)]
    return reg_loss, g_kernel, (Vx_star, Vy_star)
    
    
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
    vx,vy = optflow_regularizer_slow(filt,filt.shape,gamma=gamma)
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
        
    #plt.plot(costs)
    #print costs[-1]

    # Try with the Hessian
    tic = time.time()
    Vx_star, Vy_star = optflow_regularizer_slow(filt,filt.shape,gamma=gamma)
    toc = time.time()
    print "Slow ran in %0.6f seconds" % (toc - tic)
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
    print "Difference in cost:", cost(filt,Vx_star,Vy_star,gamma) - cost(filt,Vx_star2,Vy_star2,gamma)
    
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