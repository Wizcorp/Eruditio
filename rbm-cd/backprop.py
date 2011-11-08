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


def backprop_only3(VV, Dim, inputs, targets):

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
    assert Dim2 == Dim

    return (f, df)



def backprop(VV, Dim, inputs, targets):
    W = [0]*4  #synaptic weight matrix
    hB = [0]*4  #hidden biases

    #### Un-Flatten all of our parameters from the 1-D array
    matrices = multiUnFlatten(VV, Dim)
    W[0]  = matrices[0]
    hB[0] = matrices[1]
    W[1]  = matrices[2]
    hB[1] = matrices[3]
    W[2]  = matrices[4]
    hB[2] = matrices[5]
    W[3]  = matrices[6]
    hB[3] = matrices[7]

    # Logistic activation function:
    actF = lambda x: 1./(1. + np.exp(-x))

    ## The four-layer neural network:
    layer0out = actF(   np.dot(inputs,    W[0]) + hB[0]) #numpy auto-tiles hB
    layer1out = actF(   np.dot(layer0out, W[1]) + hB[1]) #numpy auto-tiles hB
    layer2out = actF(   np.dot(layer1out, W[2]) + hB[2]) #numpy auto-tiles hB
    layer3out = np.exp( np.dot(layer2out, W[3]) + hB[3]) #numpy auto-tiles hB

    targetout = layer3out

    # Normalize our outputs into probability distributions:
    targetout = targetout / np.tile( targetout.sum(1)[:, np.newaxis], (1,10) )

    # We use cross-entropy rather than squared error for our error function:
    f = -(targets * np.log(targetout)).sum(0).sum(0)

    # Classification error:
    Ix_class = targetout - targets

    # propagate the error back down the neural network ("backprop"):
    deltaW3 = np.dot(layer2out.conj().T, Ix_class)
    deltaHB3 = Ix_class.sum(0)[np.newaxis, :]

    ## For backprop, we take the derivative actF acting on the
    ## input data.  
    ## For the Logistic function, the derivative of actF(x) 
    ## is actF(x) * (1 - actF(x))
    Ix3 = np.dot(Ix_class, W[3].T) * layer2out * (1-layer2out)
    deltaW2 = np.dot(layer1out.T, Ix3)
    deltaHB2 = Ix3.sum(0)[np.newaxis, :]

    Ix2 = np.dot(Ix3, W[2].T) * layer1out * (1-layer1out)
    deltaW1 = np.dot(layer0out.T, Ix2)
    deltaHB1 = Ix2.sum(0)[np.newaxis, :]

    Ix1 = np.dot(Ix2, W[1].T) * layer0out * (1-layer0out)
    deltaW0 = np.dot(inputs.T, Ix1)
    deltaHB0 = Ix1.sum(0)[np.newaxis, :]

    ## Flatten the gradients into the same shape as VV:
    (df, Dim2) = multiFlatten(( deltaW0, deltaHB0, 
                                deltaW1, deltaHB1, 
                                deltaW2, deltaHB2, 
                                deltaW3, deltaHB3 ))
    assert Dim2 == Dim

    return (f, df)

