"""
3D ConvNet layers using Theano, Pylearn and Numpy

ConvLayer: convolutions, filter bank
NormLayer: normalization (LCN, GCN, local mean subtraction)
PoolLayer: pooling, subsampling
RectLayer: rectification (absolute value)

"""

import theano
from theano.tensor.nnet.conv3d2d import conv3d
from maxpool3d import max_pool_3d
from activations import relu, softplus

from numpy import sqrt, prod, ones, floor, repeat, pi, exp, zeros, sum
from numpy.random import RandomState

from theano import shared, config, _asarray
import theano.tensor.slinalg
import theano.tensor as T
floatX = config.floatX

dtensor5 = theano.tensor.TensorType(floatX, (False,)*5)

class ConvLayer(object):
    """ Convolutional layer, Filter Bank Layer """

    def __init__(self, input, n_in_maps, n_out_maps, kernel_shape, video_shape, 
        batch_size, activation,  rng, layer_name="Conv", 
        borrow=True, W=None, b=None):

        """
        video_shape: (frames, height, width)
        kernel_shape: (frames, height, width)

        W_shape: (out, in, kern_frames, kern_height, kern_width)
        """

        self.__dict__.update(locals())
        del self.self
        
        # init W
        if W != None: W_val = W
        else: 
            # fan in: filter time x filter height x filter width x input maps
            fan_in = prod(kernel_shape)*n_in_maps
            norm_scale = sqrt( 2. / fan_in )
            W_shape = (n_out_maps, n_in_maps)+kernel_shape
            W_val = _asarray(rng.normal(loc=0, scale=norm_scale, size=W_shape),\
                        dtype=floatX)
        self.W = shared(value=W_val, borrow=borrow, name=layer_name+'_W')
        self.params = [self.W]

        # init bias
        if b != None: 
            b_val = b
        else: 
            b_val = zeros((n_out_maps,), dtype=floatX)
        
	self.b = shared(b_val, name=layer_name+"_b", borrow=borrow)
        self.params.append(self.b)
        
        # Zero pad to simulate a 'same' convolution
        pad_T, pad_H, pad_W = ((kernel_shape[i]-1)/2 
                               for i in range(len(kernel_shape)))
        N = input.shape[0]
        C = n_in_maps
        TT, HH, WW = video_shape        
        T_zeros = T.zeros((N,C,pad_T,HH,WW))
        H_zeros = T.zeros((N,C,TT+2*pad_T,pad_H,WW))
        W_zeros = T.zeros((N,C,TT+2*pad_T,HH+2*pad_H,pad_W))
                                   
        paddedT = T.concatenate([T_zeros,input,T_zeros],axis=2)
        paddedTH = T.concatenate([H_zeros,paddedT,H_zeros],axis=3)
        paddedTHW = T.concatenate([W_zeros,paddedTH,W_zeros],axis=4)

        # 3D convolution; dimshuffle: last 3 dimensions must be (in, h, w)
        n_fr = video_shape[0] + 2*pad_T
        h = video_shape[1] + 2*pad_H
        w = video_shape[2] + 2*pad_W
        n_fr_k, h_k, w_k = kernel_shape
        out = conv3d(
                signals=paddedTHW.dimshuffle([0,2,1,3,4]), 
                filters=self.W, 
                signals_shape=(batch_size, n_fr, n_in_maps, h, w), 
                filters_shape=(n_out_maps, n_fr_k, n_in_maps, h_k, w_k),         
                border_mode='valid').dimshuffle([0,2,1,3,4])

        out += self.b.dimshuffle('x',0,'x','x','x')

        self.output = activation(out)

class PoolLayer(object):
    """ Subsampling and pooling layer """

    def __init__(self, input, pool_shape, method="max"):
        """
        method: "max", "avg", "L2", "L4", ...
        """

        self.__dict__.update(locals())
        del self.self

        if method=="max":
            out = max_pool_3d(input,pool_shape)
        else:
            raise NotImplementedError()

        self.output = out
        
        
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
    Vxdx = Vx[:,:,:,1:] - Vx[:,:,:,0:WW-1]
    Vxdy = Vx[:,:,1:,:] - Vx[:,:,0:HH-1,:]
    Vxdt = Vx[:,1:,:,:] - Vx[:,0:TT-1,:,:]

    Vydx = Vy[:,:,:,1:] - Vy[:,:,:,0:WW-1]
    Vydy = Vy[:,:,1:,:] - Vy[:,:,0:HH-1,:]
    Vydt = Vy[:,1:,:,:] - Vy[:,0:TT-1,:,:]
    
    if smoothness == 'L1':
        smoothness_penalty =  T.sum(abs(Vxdx)) + \
                             T.sum(abs(Vxdy)) + \
                             T.sum(abs(Vxdt)) + \
                             T.sum(abs(Vydx)) + \
                             T.sum(abs(Vydy)) + \
                             T.sum(abs(Vydt))
        
    else:
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
    L = T.slinalg.cholesky(H)
    lower_solve = T.slinalg.Solve(A_structure="lower_triangular")
    upper_solve = T.slinalg.Solve(A_structure="upper_triangular")
    b = lower_solve(L,g_all)
    # Can solve for optimum in a single newton step
    x_star = flat_join(Vx,Vy) - upper_solve(T.transpose(L),b)
                                    
    Vx_star = x_star[0:Vx.size].reshape(Vx.shape)
    Vy_star = x_star[Vx.size:].reshape(Vy.shape)
    
    # Now we can compute the minimum optical flow cost, which is the
    # regularization loss
        
    # Brightness constancy violation
    bc_diff_star = dX*Vx_star + dY*Vy_star + dT
    opt_bc_penalty = T.sum(bc_diff_star * bc_diff_star)
    
    # Smoothness constraint
    Vxdx_star = Vx_star[:,:,:,1:] - Vx_star[:,:,:,0:WW-1]
    Vxdy_star = Vx_star[:,:,1:,:] - Vx_star[:,:,0:HH-1,:]
    Vxdt_star = Vx_star[:,1:,:,:] - Vx_star[:,0:TT-1,:,:]

    Vydx_star = Vy_star[:,:,:,1:] - Vy_star[:,:,:,0:WW-1]
    Vydy_star = Vy_star[:,:,1:,:] - Vy_star[:,:,0:HH-1,:]
    Vydt_star = Vy_star[:,1:,:,:] - Vy_star[:,0:TT-1,:,:]
    
    if smoothness == 'L1':
        opt_smoothness_penalty = T.sum(abs(Vxdx_star)) + \
                                 T.sum(abs(Vxdy_star)) + \
                                 T.sum(abs(Vxdt_star)) + \
                                 T.sum(abs(Vydx_star)) + \
                                 T.sum(abs(Vydy_star)) + \
                                 T.sum(abs(Vydt_star))
    else:
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
    return reg_loss, updates, g_kernel, Vx_star, Vy_star
    
def l2_regularizer(kernel):
    """ Creates theano expressions to evaluate regularization and its gradient.
    """
    return 0.5*T.sum(kernel*kernel), kernel
    

def gaussian_filter(kernel_shape):

    x = zeros((kernel_shape, kernel_shape), dtype='float32')

    def gauss(x, y, sigma=2.0):
        Z = 2 * pi * sigma**2
        return  1./Z * exp(-(x**2 + y**2) / (2. * sigma**2))

    mid = floor(kernel_shape/ 2.)
    for i in xrange(0,kernel_shape):
        for j in xrange(0,kernel_shape):
            x[i,j] = gauss(i-mid, j-mid)

    return x / sum(x)


def mean_filter(kernel_size):
    s = kernel_size**2
    x = repeat(1./s, s).reshape((kernel_size, kernel_size))
    return x
