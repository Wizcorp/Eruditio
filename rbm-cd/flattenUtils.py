import numpy as np

## Copyright 2011, Wizcorp, www.wizcorp.jp


### Apparently, the CG solver likes to have all its parameters as
### a 1-D vector.  (Not 100% sure about this, though.)  So whenever
### a bunch of matrices go through the minimize() function, they
### need to be reshaped into 1-D going in, and then back to the
### original matrices coming back out.  

def multiFlatten(matrices):
    VV = np.hstack(map(lambda x: x.flatten(1), matrices))[:, np.newaxis]
    Dim = []
    for matrix in matrices:
        Dim.append(matrix.shape)
    return VV, Dim


def multiUnFlatten(VV, Dim):
    matrices = []
    start = 0
    for shape in Dim:
        stop = start + np.array(shape).prod()
        oneSlice = VV[start:stop]
        oneMatrix = oneSlice.reshape( shape, order='F')
        matrices.append(oneMatrix)
        start = stop
    return matrices



