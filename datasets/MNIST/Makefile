
## Copyright 2011, Wizcorp, www.wizcorp.jp

##
## This will download (and then re-package) the MNIST dataset from
## Dr. LeCun's website.
##


WEBDIR = http://yann.lecun.com/exdb/mnist

.PHONY: all

all: testImagesAndTargets.mat trainImagesAndTargets.mat

testImagesAndTargets.mat: t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
	python MNISTconverter.py test

trainImagesAndTargets.mat: train-images-idx3-ubyte train-labels-idx1-ubyte 
	python MNISTconverter.py train

train-images-idx3-ubyte.gz:
	wget $(WEBDIR)/$@

train-labels-idx1-ubyte.gz:
	wget $(WEBDIR)/$@

t10k-images-idx3-ubyte.gz:
	wget $(WEBDIR)/$@

t10k-labels-idx1-ubyte.gz:
	wget $(WEBDIR)/$@

%-idx3-ubyte: %-idx3-ubyte.gz
	gzip -dc $< > $@

%-idx1-ubyte: %-idx1-ubyte.gz
	gzip -dc $< > $@




