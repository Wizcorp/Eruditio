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
import minimize
import CG_CLASSIFY_INIT
import CG_CLASSIFY



class LogisticHinton2006:
    def loadmat(self, filename):
        return sio.loadmat( os.path.join(self.baseDir, filename), struct_as_record=True)

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


    def loadPreRBM(self):
        self.baseDir = 'nnData'
        baseName = 'Classify-randomInit-Layer%d.mat'
        self.loadNN(baseName)

    def loadPostRBM(self):
        self.baseDir = 'nnData'
        baseName = 'ClassifyAfterRBM-Layer%d.mat'
        self.loadNN(baseName)

    def loadNN(self, baseName):
        Layer0 = self.loadmat(baseName % (0))
        Layer1 = self.loadmat(baseName % (1))
        Layer2 = self.loadmat(baseName % (2))
        Layer3 = self.loadmat(baseName % (3))

        ## W is the synaptic weight matrix
        self.W = [  Layer0['W'], 
                    Layer1['W'], 
                    Layer2['W'], 
                    Layer3['W']   ]

        ## hB is the hidden biases, ie, the biases of the hidden neurons
        ##  ("hidden neurons" is a relative term -- it means the neurons that
        ##   sit atop the current W matrix in question...)
        self.hB = [  Layer0['hiddenBias'], 
                     Layer1['hiddenBias'], 
                     Layer2['hiddenBias'], 
                     Layer3['hiddenBias']   ]

        ## hB is the visible biases, ie, the biases of the visible neurons
        ##  ("visible neurons" is a relative term -- it means the neurons that
        ##   sit below the current W matrix in question...)
        self.vB = [  Layer0['visibleBias'], 
                     Layer1['visibleBias'], 
                     Layer2['visibleBias'], 
                     Layer3['visibleBias']   ]


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


    ## This is the up3() method wrapped for use with the CG-Minimizer:
    ##  FIXME:  This no worky.
    def up3cgWrap(self, VV, Dim, inputs, targets):
        #### Un-Flatten all of our parameters from the 1-D array
        matrices = multiUnFlatten(VV, Dim)
        W = matrices[0]
        hB = matrices[1]
    
        nnTargetOut = self.up3(inputs)
        nnTargetOut = nnTargetOut / np.tile(nnTargetOut.sum(1)[:, np.newaxis], (1,10) )
    
        f = -(targets * np.log(nnTargetOut)).sum(0).sum(0)
    
        classError = nnTargetOut - targets
    
        ## re-stack the gradient into the same shape as VV:
        (df, Dim2) = multiFlatten((   np.dot(inputs.T, classError),
                                      classError.sum(0)[:, np.newaxis].T  ))
        assert Dim == Dim2

        return (f, df[:, 0])



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

        (X, fX, iters) = minimize.minimize(VV, CG_CLASSIFY_INIT.CG_CLASSIFY_INIT, (Dim, layer2out, targets), max_iter  )
        #(X, fX, iters) = minimize.minimize(VV, self.up3cgWrap, (Dim, layer2out, targets), max_iter  )

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

        (X, fX, iters) = minimize.minimize(VV, CG_CLASSIFY.CG_CLASSIFY, (Dim, inputData, targets), max_iter  )

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


    # The Restricted Boltzmann Machine:
    def rbm(self, whichLayer, inputData, randomNumbers = None):

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
            randomNumbers = np.random.rand(poshidprobs.shape)
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


