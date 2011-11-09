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




import time, sys
import numpy as np
import scipy.io as sio

import NeuralNetwork

nn = NeuralNetwork.LogisticHinton2006()
# Load the neural network that was created from step1:
nn.load('nnData/NN_afterPreTrain.mat');


#### START:  Load the MNIST training and testing data
print
sys.stdout.write("Loading training data...")
sys.stdout.flush()
trainData = sio.loadmat('../datasets/MNIST/trainImagesAndTargets.mat', struct_as_record=True)
print " done."

sys.stdout.write("Loading testing data...")
sys.stdout.flush()
testData = sio.loadmat('../datasets/MNIST/testImagesAndTargets.mat', struct_as_record=True)
print " done."

trainImages = trainData['images']
trainTargets = trainData['targets']
testImages = testData['images']
testTargets = testData['targets']

(numCases, numPixels) = trainImages.shape
(numCases2, numTargets) = trainTargets.shape
assert numCases == numCases2

(numTestCases, numPixels2) = testImages.shape
(numTestCases2, numTargets2) = testTargets.shape
assert numTestCases == numTestCases2
assert numPixels == numPixels2
assert numTargets == numTargets2
#### END:  Load the MNIST training and testing data




def countErrors(inputData, targets):
    newTargetOut = nn.recognize(inputData)
    newTargetOut = newTargetOut/np.tile( newTargetOut.sum(1)[:, np.newaxis], (1,10) )
    J = newTargetOut.argmax(1)
    J1 = targets.argmax(1)
    numErrors = np.sum(1*(J==J1))
    errCr = - (targets * np.log(newTargetOut)).sum(0).sum(0)
    return numErrors, errCr

def batchCountErrors(allImages, allTargets, numbatches):
    ## Divide the job into batches so that we don't run out of memory
    ## (Here, batchsize doesn't affect results or CPU time, just memory usage.)
    numCases = allImages.shape[0]
    batchsize = numCases / numbatches
    err_cr = 0.
    counter = 0
    for batch in xrange(numbatches):
        sys.stdout.write("    batch %d (of %d)\r" % (batch, numbatches))
        sys.stdout.flush()
        start = batch * batchsize
        stop = (batch+1) * batchsize

        images = allImages[start:stop, :]
        targets = allTargets[start:stop, :]

        (numErrs, errCr) = countErrors(images, targets)

        counter = counter + numErrs
        err_cr = err_cr + errCr
    print
    sys.stdout.flush()
    t_err = allImages.shape[0] - counter
    t_crerr = err_cr / numbatches
    percent_error = 100 * float(t_err) / float(numCases)

    print "    Misclassified %d out of %d images.  (%.2f%% error)" % (t_err, numCases, percent_error)
    return t_err, t_crerr


print 
print "Before doing any backprop:"

print "  Counting the number of mis-classifications in the training set..."
(pre_train_err, pre_train_crerr) = batchCountErrors(trainImages, trainTargets, 600)

print "  Counting the number of mis-classifications in the test set..."
(pre_test_err, pre_test_crerr) = batchCountErrors(testImages, testTargets, 100)

print
print ' === Training model by minimizing cross entropy error === '
maxepoch=10

test_err = [0]*maxepoch
train_err = [0]*maxepoch
test_crerr = [0.]*maxepoch
train_crerr = [0.]*maxepoch
for epoch in xrange(maxepoch):
    print "Starting epoch", epoch
    ## Divide the data into 60 batches to save memory
    assert numCases == 60000, "Expecting 60,000 training images."
    numbatches = 60
    batchsize = 1000
    assert batchsize*numbatches == numCases
    print '  %d batches of %d cases each.' % (numbatches, batchsize)

    for batch in xrange(numbatches):
        print "    batch %d of %d:" % (batch, numbatches)
        start = batch * batchsize
        stop = (batch+1) * batchsize

        data = trainImages[start:stop, :]
        targets = trainTargets[start:stop, :]

        ### START:  Conjugate gradient with 3 linesearches
        max_iter = 3

        if (epoch < 5):  # At first, we only update the final layer
            nn.minimizeLayer3(data, targets, max_iter)
        else:
            nn.minimizeAllLayers(data, targets, max_iter)

    print "After Epoch %d:" % (epoch)
    print "  Counting the number of mis-classifications in the training set..."
    (train_err[epoch], train_crerr[epoch]) = batchCountErrors(trainImages, trainTargets, 600)
 
    print "  Counting the number of mis-classifications in the test set..."
    (test_err[epoch], test_crerr[epoch]) = batchCountErrors(testImages, testTargets, 100)

    print
    print

