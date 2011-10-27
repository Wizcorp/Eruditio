#!/usr/bin/python

## Copyright 2011, Wizcorp, www.wizcorp.jp

## This code was based heavily upon some matlab code from 
## http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html
## which included the following notice: 
#
# Code provided by Ruslan Salakhutdinov and Geoff Hinton
#
# Permission is granted for anyone to copy, use, modify, or distribute this
# program and accompanying programs and documents for any purpose, provided
# this copyright notice is retained and prominently displayed, along with
# a note saying that the original programs are available from our
# web page.
# The programs and documents are distributed without any warranty, express or
# implied.  As the programs were written for research purposes only, they have
# not been tested to the degree that would be advisable in any important
# application.  All use of these programs is entirely at the user's own risk.


import numpy as np
from flattenUtils import *

def CG_CLASSIFY_INIT(VV, Dim, inputs, targets):

    #### Un-Flatten all of our parameters from the 1-D array
    matrices = multiUnFlatten(VV, Dim)
    W = matrices[0]
    hB = matrices[1]

    ## This is the last layer of the neural network, in bottom-up, 
    ## "recognition" mode:
    nnTargetOut = np.exp(np.dot(inputs, W) + hB)

    # Normalize our outputs into probability distributions:
    nnTargetOut = nnTargetOut / np.tile(nnTargetOut.sum(1)[:, np.newaxis], (1,10) )

    # We use cross-entropy rather than squared error for our error function:
    f = -(targets * np.log(nnTargetOut)).sum(0).sum(0)

    classError = nnTargetOut - targets

    ## Flatten the gradients into the same shape as VV:
    (df, Dim2) = multiFlatten((   np.dot(inputs.T, classError), 
                                  classError.sum(0)[np.newaxis, :]  ))
                                  #classError.sum(0)[:, np.newaxis].T  ))
    assert Dim2 == Dim

    return (f, df)

