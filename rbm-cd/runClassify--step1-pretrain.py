#!/usr/bin/python

import time, sys, os
import numpy as np
import scipy.io as sio

import NeuralNetwork

from batchCD1 import batchCD1

## We want a Restricted Boltzmann Machine (RBM) which is a type of
## neural network.  Specifically, we want the 4-layer model described
## in (Hinton, Osindero, Teh, 2006).
nn = NeuralNetwork.LogisticHinton2006()
nn.initRBM()

# Load the MNIST training data:
trainData = sio.loadmat('../datasets/MNIST/trainImagesAndTargets.mat', struct_as_record=True)

trainImages = trainData['images']
trainTargets = trainData['targets']
assert trainImages.shape[0] == trainTargets.shape[0]


## We use a "batched" version of 1-step Constrastive Divergence (CD) to 
## pre-train the first 3 layers (0,1,2) of our neural network.  

layer0out = batchCD1(nn, 0, trainImages, maxepoch=5)

### To save here:
#nn.save('...filename...')

### To restart here:
#nn.load('...filename...')
#layer0out = nn.up0(trainImages)

layer1out = batchCD1(nn, 1, layer0out, maxepoch=5)

### To save here:
#nn.save('...filename...')

### To restart here:
#nn.load('...filename...')
#layer0out = nn.up0(trainImages)
#layer1out = nn.up1(layer0out)

layer2out = batchCD1(nn, 2, layer1out, maxepoch=5)


### Give layer3 some random biases:
l3numVis = nn.W[2].shape[1]
l3numHid = trainTargets.shape[1]

nn.vB[3] = 0.1*np.random.randn(1, l3numVis)
nn.hB[3] = 0.1*np.random.randn(1, l3numHid)

# Save the pre-trained neural network:
nn.save('nnData/NN_afterPreTrain.mat')


