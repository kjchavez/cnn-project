# 3D Convolutional Neural Networks with Optical Flow Regularization for Video Recognition

## Proposed idea
Infuse the 3D CNN model with the assumptions used in optical flow computaitons in a *soft* way through a special regularization on the filters.

## In this project
We test the merit of this idea by training ConvNets from scratch on the UCF101 Human Action Recognition data set using Theano. See the [report](http://cs231n.stanford.edu/reports/kjchavez_final.pdf).

## Dependencies
* Numpy, Scipy
* Theano (0.7rc1 or later)
* OpenCV (for reading videos)
* LMDB
* Matplotlib (to use analysis tools)
