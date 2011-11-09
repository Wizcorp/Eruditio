#!/usr/bin/python

## Copyright 2011, Wizcorp, www.wizcorp.jp

## This code was inspired by some matlab code from 
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



import time, sys, os
import numpy as np
import scipy.io as sio

from flattenUtils import *
import minimize as cg
from backprop import backprop, backprop_only3


class LogisticHinton2006:
    def __init__(self):
        ## The "up" methods are for bottom-up recognition.
        self.up = [self.up0, self.up1, self.up2, self.up3]
        ## The "down" methods are for top-down generation.  (ie, asking the 
        ## network what it "believes" in)
        self.down = [self.down0, self.down1, self.down2, self.down3]

        ## Four layers of neural networks means Five layers of neurons
        ##   (eg: Neurons -- Layer -- Neurons -- Layer -- Neurons -- etc...)
        ##
        ## The model from the paper is:
        #
        #              10 classification neurons ("what digit do I see?")
        # Layer3:    2000x10 synapses
        #              2000 neurons
        # Layer2:    500x2000 synapses
        #              500 neurons
        # Layer1:    500x500 synapses
        #              500 neurons
        # Layer0:    784x500 synapses
        #              784 neurons (28x28 pixels on a retina)

        ##  Note that Hinton and crew use Matlab, so they are in the habit of
        ##  counting from 1.  This code is all Python+Numpy, so we count from 0.

        ## "Hidden" vs "Visible" neurons:
        ##   An example:  
        ##      When we talk about Layer2, the neurons BELOW are VISIBLE,
        ##                                 the neurons ABOVE are HIDDEN.  
        ##      When we talk about Layer3, the neurons BELOW are VISIBLE,
        ##                                 the neurons ABOVE are HIDDEN.  
        ##  So Layer2's hidden neurons are the same as Layer3's visible neurons.  

    def initRBM(self):
        self.W = [ 0.1*np.random.randn(784, 500),
                   0.1*np.random.randn(500, 500),
                   0.1*np.random.randn(500, 2000),
                   0.1*np.random.randn(2000, 10)    ]

        self.hB = [ np.zeros((1,500)),
                    np.zeros((1,500)),
                    np.zeros((1,2000)),
                    np.zeros((1,10))    ]

        self.vB = [ np.zeros((1,784)),
                    np.zeros((1,500)),
                    np.zeros((1,500)),
                    np.zeros((1,2000))  ]

    def save(self, filename):
        return sio.savemat(filename, {'W0': self.W[0], 
                                      'W1': self.W[1],
                                      'W2': self.W[2],
                                      'W3': self.W[3],
                                      'hB0': self.hB[0],
                                      'hB1': self.hB[1],
                                      'hB2': self.hB[2],
                                      'hB3': self.hB[3],
                                      'vB0': self.vB[0],
                                      'vB1': self.vB[1],
                                      'vB2': self.vB[2],
                                      'vB3': self.vB[3] }, oned_as='row')

    def load(self, filename):
        myDict = sio.loadmat(filename, struct_as_record=True)
        self.W = []
        self.hB = []
        self.vB = []
        self.W.append(myDict['W0'])
        self.W.append(myDict['W1'])
        self.W.append(myDict['W2'])
        self.W.append(myDict['W3'])
        self.hB.append(myDict['hB0'])
        self.hB.append(myDict['hB1'])
        self.hB.append(myDict['hB2'])
        self.hB.append(myDict['hB3'])
        self.vB.append(myDict['vB0'])
        self.vB.append(myDict['vB1'])
        self.vB.append(myDict['vB2'])
        self.vB.append(myDict['vB3'])

    def recognize(self, inputData):
        return self.up3(self.up2(self.up1(self.up0(inputData))))

    def recognize012(self, inputData):
        return self.up2(self.up1(self.up0(inputData)))

    def recognize3(self, inputData):
        return self.up3(inputData)

    def up0(self, inputData):
        return 1./(1. + np.exp(-np.dot(inputData, self.W[0]) - self.hB[0]))

    def up1(self, inputData):
        return 1./(1. + np.exp(-np.dot(inputData, self.W[1]) - self.hB[1]))

    def up2(self, inputData):
        return 1./(1. + np.exp(-np.dot(inputData, self.W[2]) - self.hB[2]))

    def up3(self, inputData):
        return np.exp( np.dot(inputData, self.W[3]) + self.hB[3])

    def down0(self, inputData):
        return 1./(1. + np.exp(-np.dot(inputData, self.W[0].T) - self.vB[0]))

    def down1(self, inputData):
        return 1./(1. + np.exp(-np.dot(inputData, self.W[1].T) - self.vB[1]))

    def down2(self, inputData):
        return 1./(1. + np.exp(-np.dot(inputData, self.W[2].T) - self.vB[2]))

    def down3(self, inputData):
        return 1./(1. + np.exp(-np.dot(inputData, self.W[3].T) - self.vB[3]))


    def minimizeLayer3(self, inputData, targets, max_iter):
        layer2out = self.recognize012(inputData)

        #### Flatten all of our parameters into a 1-D array
        (VV, Dim) = multiFlatten(( self.W[3], self.hB[3] ))

        (X, fX, iters) = cg.minimize(VV, backprop_only3, (Dim, layer2out, targets), max_iter)

        #### Un-Flatten all of our parameters from the 1-D array
        matrices = multiUnFlatten(X, Dim)
        self.W[3]  = matrices[0]
        self.hB[3] = matrices[1]


    def minimizeAllLayers(self, inputData, targets, max_iter):
        #### Flatten all of our parameters into a 1-D array
        (VV, Dim) = multiFlatten((  self.W[0], self.hB[0],
                                    self.W[1], self.hB[1],
                                    self.W[2], self.hB[2],
                                    self.W[3], self.hB[3]  ))

        (X, fX, iters) = cg.minimize(VV, backprop, (Dim, inputData, targets), max_iter)

        #### Un-Flatten all of our parameters from the 1-D array
        matrices = multiUnFlatten(X, Dim)
        self.W[0]  = matrices[0]
        self.hB[0] = matrices[1]
        self.W[1]  = matrices[2]
        self.hB[1] = matrices[3]
        self.W[2]  = matrices[4]
        self.hB[2] = matrices[5]
        self.W[3]  = matrices[6]
        self.hB[3] = matrices[7]


    # 1-step Constrastive Divergence:
    def cd1(self, whichLayer, inputData, randomNumbers = None):

        #  randomNumbers
        # We can use pre-defined "random" numbers so that repeat runs
        # will produce identical results.  The default is to create our own
        # random numbers on the fly. 

        epsilonW = 0.1
        epsilonVB = 0.1
        epsilonHB = 0.1
        weightcost = 0.0002

        # The neural network used for bottom-up recognition:
        up = self.up[whichLayer]

        # The neural network used for top-down generation (ie, a "belief/fantasy"):
        down = self.down[whichLayer]

        #### Positive Phase:
        poshidprobs = up(inputData)
        if (randomNumbers == None):
            randomNumbers = np.random.rand(poshidprobs.shape[0], poshidprobs.shape[1])
        poshidstates = 1 * (poshidprobs > randomNumbers)
        del randomNumbers
        posprods = np.dot(inputData.T, poshidprobs)  #an unbiased sample, <v_i, h_j>_data

        #### Negative Phase:
        negdata = down(poshidstates)  # What does the network believe?
        neghidprobs = up(negdata)
        negprods = np.dot(negdata.T, neghidprobs)

        beliefError = ( (inputData-negdata)**2 ).sum(0).sum(0)

        numcases = inputData.shape[0]
        posvisact = inputData.sum(0)
        poshidact = poshidprobs.sum(0)
        negvisact = negdata.sum(0)
        neghidact = neghidprobs.sum(0)

        # Reduce the probability incorrect beliefs:
        lDeltaW = epsilonW * ( (posprods-negprods)/numcases - weightcost*self.W[whichLayer])
        # Renormalize the biases: 
        lDeltaVB = (epsilonVB/numcases) * (posvisact-negvisact)
        lDeltaHB = (epsilonHB/numcases) * (poshidact-neghidact)

        return lDeltaW, lDeltaVB, lDeltaHB, poshidprobs, beliefError


