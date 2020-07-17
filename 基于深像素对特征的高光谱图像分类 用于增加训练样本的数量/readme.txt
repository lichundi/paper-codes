In order to run this demo, you need to install tensorflow for python 2.7. My platform is Ubuntu 14.04(other platform will be OK if tensorflow supports), and all tested Graphics Cards have at least 4G memory, check your card's memory and make sure it is able to run this demo.

the code was written with tensorflow 0.7(tested on version 0.11, 0.10), and may not compatiable with the newest tensorflow(0.12 by now, not tested), make sure you install the right version of tensorflow if problem occurs.
Or you can hack the code yourself(I think serval small changes will be enough).

This is a demo for Pavia University dataset.
1. Run demo.m in matlab to generate training set and test set(Make sure you have 8G memory or your computer will get stuck). You will get several .mat file after running demo.m in matlab(or octave, not tested). And my matlab version is 2014a(check your matlab version if problem occurs).
2. Run run_cnn_demo.sh to start training and testing, prediction will be saved in "prediction.txt"

After these two steps, you will see some output in your terminal. I have provided my screen shot in this folder.


