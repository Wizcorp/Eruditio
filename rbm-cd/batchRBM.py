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



import sys
import time
import numpy as np
import scipy.io as sio

def batchRBM(nn, whichLayer, allInputData, maxepoch=10, baseFileName=""):
    numCases = allInputData.shape[0]
    assert numCases == 60000, "Expecting 60,000 inputs."
    # We will divide the data into 600 batches:
    numbatches = 600
    batchsize = 100
    assert batchsize * numbatches == numCases

    (numvis, numhid) = nn.W[whichLayer].shape

    deltaW = np.zeros(nn.W[whichLayer].shape) # Synaptic weight matrix
    deltaHB = np.zeros(nn.hB[whichLayer].shape) # Hidden biases
    deltaVB = np.zeros(nn.vB[whichLayer].shape) # Visible biases
    batchHidProbs = np.zeros((numCases, numhid))

    initialmomentum = 0.5
    finalmomentum = 0.9

    print "Doing %d epochs * %d batches." % (maxepoch, numbatches)
    for epoch in xrange(maxepoch):
        tStart = time.time()
        print "epoch ", epoch
        try:
            del randCompare # can be huge memory consumption
        except:
            pass
        randCompare = sio.loadmat('%s-randCompare%d.mat' % (baseFileName, epoch+1), 
                                  struct_as_record=True)['randCompare']
        errsum = 0.
        for batch in xrange(numbatches):
            sys.stdout.write("epoch %d batch %d\r" % (epoch, batch))
            sys.stdout.flush()
            start = batch * batchsize
            stop = (batch+1) * batchsize

            inputData = allInputData[start:stop, :]
            batchRandCompare = randCompare[batch, :, :]

            (lDeltaW, lDeltaVB, lDeltaHB, poshidprobs, err) = nn.rbm(whichLayer, inputData, batchRandCompare)
            del batchRandCompare

            batchHidProbs[start:stop, :] = poshidprobs

            errsum += err
            if (epoch > 4):
                momentum = finalmomentum
            else:
                momentum = initialmomentum

            deltaW  = momentum * deltaW  + lDeltaW
            deltaVB = momentum * deltaVB + lDeltaVB
            deltaHB = momentum * deltaHB + lDeltaHB

            nn.W[whichLayer]  = nn.W[whichLayer]  + deltaW
            nn.vB[whichLayer] = nn.vB[whichLayer] + deltaVB
            nn.hB[whichLayer] = nn.hB[whichLayer] + deltaHB


        del randCompare
        print "\nepoch %d error %6.1f" % (epoch, errsum)

        tEnd = time.time()
        print "total time: %.3f seconds\n" % (tEnd-tStart)

    return batchHidProbs


