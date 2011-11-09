
A Neural Network (specifically, a Restricted Boltzmann Machine) to 
Recognize Hand-Written Digits. 
------------------------------------------------------------------------

For the impatient:
- You will need:
    a Linux-like (/Unix-like) system with a shell
    make
    wget, gzip
    Python + numpy + scipy

- To run the code:
    cd ../datasets/MNIST/
    make  #(this will take a few minutes to download some data)
    cd ../../rbm-cd
    python runClassify--step1-pretrain.py   #(this will take many minutes)
    python runClassify--step2-backprop.py   #(this will take hours)

(Step 1 takes about 30 minutes on my laptop.  Layer0 is about 85 seconds per
epoch * 5 epochs.  Layer1 is about 55 seconds per epoch * 5 epochs.  Layer2
is about 225 seconds per epoch * 5 epochs.  Total, about 30 minutes.)

Step 2, backprop, should give you plenty of feedback as it goes.  At first,
you should have about a 90% error rate (ie, the network is essentially 
"guessing" at what the digit is).  After the first pass (60 batches of 3
Linesearches) you should have about a 5% error rate.  After many hours,
the error rate should drop to less than 2%.  




# What's going on here?

This entire directory is my attempt to understand "Restricted Boltzmann 
Machines" (RBMs) created mostly by Prof. Hinton at U Toronto.  The main
paper (that I care about for the purposes of this directory) is:  

  http://www.cs.toronto.edu/~hinton/absps/ncfast.pdf

The software here is based heavily on software from Hinton and 
Salakhutdinov which can be found here:

  http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html


The files:
  /datasets/MNIST/Makefile   - download and re-package the MNIST source 
                               data (70,000 hand-written digits.)

  /rbm-cd/NeuralNetwork.py   - four-layer neural-network, RBM

  /rbm-cd/runClassify--step1-pretrain.py  - pretrain the RBM using CD

  /rbm-cd/runClassify--step2-backprop.py  - train the RBM using regular 
                                            backprop (including a test
                                            at each iteration)



# Why do I care?

I'm very interested in Machine Learning (neural networks, Bayesian networks,
things of that nature).  The RBM played a significant role in the success of
the Netflix prize which was worth 10 million dollars.  


