
The purpose of this Makefile and script is to download the source 
data (70,000 labelled images of hand-written digits) and convert
it into a form that Python+Numpy can use to train and test a
neural network.  

Requirements:
 - "make" to execute the Makefile
 - "wget" to download the files
 - "gzip" to unzip the files
 - Python+Numpy, to convert the files


Quickstart:

    Type "make".


Expected results:

  The Makefile will download 4 files.  The final output that we care 
  about will be two files:

      trainImagesAndTargets.mat (~364M)
      testImagesAndTargets.mat (~61M)



For more information about the MNIST dataset, please see:
   http://yann.lecun.com/exdb/mnist/

