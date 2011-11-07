#!/usr/bin/python

## Copyright 2011, Wizcorp, www.wizcorp.jp


# This script will load the MNIST data (in its own format) and convert
# it into numpy arrays.  It will also shuffle the data.  
# 
#  For more info on the MNIST data format, see:
#    http://yann.lecun.com/exdb/mnist/


import os
import sys
import struct

import numpy as np
import scipy.io as sio


def readImages( imagesFile, labelsFile):
    # images format is:
    #  32-bit magic number ( = 0x00000803 (2051) )
    #  32-bit count
    #  32-bit y-resolution (number of rows, like 28)
    #  32-bit x-resolution (number of columns, like 28)
    #    many (count * y-res * x-rex) 8-bit pixels
    f = open(imagesFile,'rb');
    f_mn = struct.unpack('>i', f.read(4))[0]
    if (f_mn != 2051):
        print "Error: Images file has the wrong magic number."
    f_count = struct.unpack('>i', f.read(4))[0]
    f_yRes = struct.unpack('>i', f.read(4))[0]
    f_xRes = struct.unpack('>i', f.read(4))[0]
    imageSize = f_xRes * f_yRes
    if ( os.path.getsize(imagesFile) != f_count * imageSize + f.tell() ):
        print "Error: Images file is the wrong size or resolution"

    # labels format is:
    #  32-bit magic number ( = 0x00000801 (2049) )
    #  32-bit number of items
    #    many (count) 8-bit labels
    g = open(labelsFile,'rb');
    g_mn = struct.unpack('>i', g.read(4))[0]
    if (g_mn != 2049):
        print "Error: Labels file has the wrong magic number."
    g_count = struct.unpack('>i', g.read(4))[0]
    if (g_count != f_count):
        print "Error: Labels file has a different number of items from the images file"
    if ( os.path.getsize(labelsFile) != g_count + g.tell() ):
        print "Error:  Labels file is the wrong size"

    images = range(10)
    for i in range(len(images)):
        images[i] = []
    unpackFmt = str(f_xRes * f_yRes) + 'B'
    for label in g.read():
        rawImage = f.read(imageSize)
        image =struct.unpack( unpackFmt, rawImage)
        images[ ord(label) ].append(image)
    #numpy-ize it:
    numpyImages = []
    for i in range(len(images)):
        numpyImages.append( np.array(images[i], dtype='uint8') )

    return numpyImages




def stackAndShuffle(imagesByLabel):
    # Re-stack the training data into a single array:
    imagesStacked = np.vstack(( imagesByLabel))

    # Go from uint8 to float64; scale to 1.0:
    imagesStacked = imagesStacked/255.

    # Create the output target set based on the labels:
    targets = np.vstack(( np.tile([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], (imagesByLabel[0].shape[0], 1)),
                          np.tile([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], (imagesByLabel[1].shape[0], 1)),
                          np.tile([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], (imagesByLabel[2].shape[0], 1)),
                          np.tile([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], (imagesByLabel[3].shape[0], 1)),
                          np.tile([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], (imagesByLabel[4].shape[0], 1)),
                          np.tile([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], (imagesByLabel[5].shape[0], 1)),
                          np.tile([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], (imagesByLabel[6].shape[0], 1)),
                          np.tile([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], (imagesByLabel[7].shape[0], 1)),
                          np.tile([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], (imagesByLabel[8].shape[0], 1)),
                          np.tile([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], (imagesByLabel[9].shape[0], 1))  ))

    numImages = imagesStacked.shape[0]
    print 'Size of this dataset: ', numImages

    ## Shuffle the training data:
    np.random.seed(0) #so we know the permutation of the training data
    randomOrder = np.random.permutation(numImages)
    newImages = imagesStacked[randomOrder, :]
    newTargets = targets[randomOrder, :]

    return {'images': newImages, 'targets': newTargets }



def usageAndExit():
    sys.stderr.write('Usage:  ' + sys.argv[0] + ' train|test\n')
    sys.exit(1)


if len(sys.argv) != 2:
    usageAndExit()

if sys.argv[1] == 'train':
    imageFile = 'train-images-idx3-ubyte'
    labelFile = 'train-labels-idx1-ubyte'
    outFileName = 'trainImagesAndTargets.mat'

elif sys.argv[1] == 'test':
    imageFile = 't10k-images-idx3-ubyte'
    labelFile = 't10k-labels-idx1-ubyte'
    outFileName = 'testImagesAndTargets.mat'

else:
    usageAndExit()


print "Loading the " + sys.argv[1] + "ing images...",
sys.stdout.flush()
imagesByLabel = readImages(imageFile, labelFile)
print " done."

## imagesByLabel[0] should all look like zeros
## imagesByLabel[1] should all look like ones
## imagesByLabel[2] should all look like twos
##  ...etc

print "Stacking and shuffling the data..."
shuffledDataset = stackAndShuffle(imagesByLabel)

print "Saving the data..."
sio.savemat(outFileName, shuffledDataset, oned_as='row')

print "Done."
