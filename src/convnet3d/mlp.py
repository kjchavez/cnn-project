"""
MLP Layers using Theano

    LogRegr
    HiddenLayer
    DropoutLayer
"""

from numpy import zeros, sqrt, ones
from numpy.random import RandomState
from theano import shared, config, _asarray
from activations import  sigmoid, relu, softplus
import theano
from theano.tensor.nnet import  softmax
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as T

floatX = config.floatX


class LogRegr(object):
    """ Logistic Regression Layer, Top layer, Softmax layer, Output layer """

    def __init__(self, input, n_in, n_out, activation, rng, layer_name="LogReg", 
        W=None, b=None, borrow=True):

        # Weigth matrix W
        if W != None: 
            self.W = W
       # elif activation in (relu,softplus): 
        else:
            print sqrt(2.0/n_in)
            W_val = _asarray(rng.normal(loc=0, scale=0.001,#sqrt(2.0/n_in), 
                size=(n_in, n_out)), dtype=floatX)
            self.W = shared(W_val, name=layer_name+"_W", borrow=borrow)

#        else:
#            self.W = shared(zeros((n_in, n_out), dtype=floatX), 
#                name=layer_name+"_W",
#                borrow=borrow)

        # Bias vector
        if b != None: 
            self.b = b
        elif activation in (relu,softplus): 
            b_val = ones((n_out,), dtype=floatX)
            self.b = shared(value=b_val, borrow=True)
        else:
            self.b = shared(zeros((n_out,), dtype=floatX),
                name=layer_name+"_b",
                borrow=borrow)
            

        # Vector of prediction probabilities
        self.p_y_given_x = softmax(T.dot(input, self.W) + self.b)
        # Prediction
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # Parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """ Cost function: negative log likelihood """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y],
                       dtype=theano.config.floatX,
                       acc_dtype=theano.config.floatX)

    def errors(self, y):
        """ Errors over the total number of examples (in the minibatch) """
        return T.mean(T.neq(self.y_pred, y),
                      dtype=theano.config.floatX,
                      acc_dtype=theano.config.floatX)


class HiddenLayer(object):
    """ Hidden Layer """

    def __init__(self, input, n_in, n_out, activation, rng, 
        layer_name="HiddenLayer", W=None, b=None, borrow=True):

        if W != None: 
            self.W = W
        else:
            W_val = _asarray(rng.normal(loc=0, scale=sqrt(2.0/n_in), 
                size=(n_in, n_out)), dtype=floatX)
            self.W = shared(W_val, name=layer_name+"_W", borrow=borrow)        

        if b != None: 
            self.b = b
        else: 
            # Initialize b with zeros
            self.b = shared(value=zeros((n_out,), dtype=config.floatX),
                            borrow=True, name=layer_name+"_b")

        # Parameters of the model
        self.params = [self.W, self.b]
        # Output of the hidden layer
        self.output = activation(T.dot(input, self.W) + self.b)


def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX) * T.cast(1./(1. - p),theano.config.floatX)
    return output

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, input, n_in, n_out,
                 activation, dropout_rate, rng, W=None, b=None,
                 layer_name="FC"):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation,layer_name=layer_name)
        
        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)
