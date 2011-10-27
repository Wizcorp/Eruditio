#!/usr/bin/python

## Copyright 2011, Wizcorp, www.wizcorp.jp


import numpy as np
import Image

def showOneDigit(oneDigit):
    oneDigit = oneDigit / oneDigit.max() * 255
    aDigit = 255 - oneDigit.astype('uint8')
    aDigit.shape = (28, 28)
    im = Image.fromarray(aDigit, 'L')
    im.show()

def showTwoRowsOfDigits(digits):
    assert len(digits) > 1     , "Need at least two digits..."
    assert (len(digits)%2) == 0, "Need an even number of digits..."
    width = len(digits)/2
    digits[0].shape = (28,28)
    digits[width].shape = (28,28)
    topRow = digits[0]
    botRow = digits[width]
    for i in range(1, width):
        digits[i].shape = (28,28)
        digits[width+i].shape = (28,28)
        topRow = np.hstack( (topRow, digits[i]) )
        botRow = np.hstack( (botRow, digits[width+i]) )
    bothRows = np.vstack( (topRow, botRow) )
    im = Image.fromarray(bothRows, 'L')
    im.show()


def showTwoDigits(digits):
    assert len(digits) == 2, "Need two digits."
    leftDigit = digits[0][0:784]
    rightDigit = digits[1][0:784]
    leftDigit = leftDigit / leftDigit.max()
    rightDigit = rightDigit / rightDigit.max()
    leftDigit = (leftDigit * 255.).astype('uint8')
    rightDigit = (rightDigit * 255.).astype('uint8')
    leftDigit.shape = (28, 28)
    rightDigit.shape = (28, 28)
    bothDigits = np.hstack( (leftDigit, rightDigit) )
    im = Image.fromarray(bothDigits, 'L')
    im.show()

