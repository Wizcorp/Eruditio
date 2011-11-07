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


#### START:  Load the MNIST training and testing data
sys.stdout.write("Loading training data...")
sys.stdout.flush()
trainData = sio.loadmat('../datasets/MNIST/trainImagesAndTargets.mat', struct_as_record=True)
sys.stdout.write(" done.\n")
sys.stdout.flush()

sys.stdout.write("Loading testing data...")
sys.stdout.flush()
testData = sio.loadmat('../datasets/MNIST/testImagesAndTargets.mat', struct_as_record=True)
sys.stdout.write(" done.\n")
sys.stdout.flush()

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
    #numbatches = 600
    batchsize = allImages.shape[0] / numbatches
    err_cr = 0.
    counter = 0
    for batch in xrange(numbatches):
        sys.stdout.write("epoch %d,  batch %d (of %d)\r" % (epoch, batch, numbatches))
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
    return t_err, t_crerr


nn = NeuralNetwork.LogisticHinton2006()
nn.loadPostRBM();

maxepoch=10

test_err = [0]*maxepoch
train_err = [0]*maxepoch
test_crerr = [0.]*maxepoch
train_crerr = [0.]*maxepoch
for epoch in xrange(maxepoch):
#    print " ** Epoch", epoch
#    print "Counting the number of mis-classifications in training set..."
#    (train_err[epoch], train_crerr[epoch]) = batchCountErrors(trainImages, trainTargets, 600)
 
#    print "Counting the number of mis-classifications in test set..."
#    (test_err[epoch], test_crerr[epoch]) = batchCountErrors(testImages, testTargets, 100)

#    print "Before epoch %d: " % (epoch)
#    print "  Train # misclassified: %d (from %d)." % (train_err[epoch], numCases)
#    print "  Test # misclassified: %d (from %d)" % (test_err[epoch], numTestCases)
#    sys.stdout.flush()

    ## Divide the data into 60 batches to save memory
    assert numCases == 60000, "Expecting 60,000 training images."
    numbatches = 60
    batchsize = 1000
    assert batchsize*numbatches == numCases
    print 'Training discriminative model on MNIST by minimizing cross entropy error.'
    print '60 batches of 1000 cases each.'

    for batch in xrange(numbatches):
        sys.stdout.write("epoch %d,  batch %d (of %d)\r" % (epoch, batch, numbatches))
        sys.stdout.flush()
        start = batch * batchsize
        stop = (batch+1) * batchsize

        data = trainImages[start:stop, :]
        targets = trainTargets[start:stop, :]

        ### START:  Conjugate gradient with 3 linesearches
        max_iter = 3
        # For some epochs, we only update the final layer:
        # (depending on what kind of mood I'm in...)
        if (epoch < 2):
            #nn.minimizeLayer3(data, targets, max_iter)
            nn.minimizeAllLayers(data, targets, max_iter)
        elif (epoch == 2):
            nn.minimizeAllLayers(data, targets, max_iter)
        elif (epoch < 6):
            nn.minimizeLayer3(data, targets, max_iter)
        else:
            nn.minimizeAllLayers(data, targets, max_iter)


    print " ** Epoch", epoch
    print "Counting the number of mis-classifications in training set..."
    (train_err[epoch], train_crerr[epoch]) = batchCountErrors(trainImages, trainTargets, 600)
 
    print "Counting the number of mis-classifications in test set..."
    (test_err[epoch], test_crerr[epoch]) = batchCountErrors(testImages, testTargets, 100)

    print "AFTER epoch %d: " % (epoch)
    print "  Train # misclassified: %d (from %d)." % (train_err[epoch], numCases)
    print "  Test # misclassified: %d (from %d)" % (test_err[epoch], numTestCases)
    sys.stdout.flush()


