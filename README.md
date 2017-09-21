# DLV


NB: the software is currently under active development. Please feel free to contact the developer by email: xiaowei.huang@cs.ox.ac.uk. 

Together with the software, there are two documents in Documents/ directory, one is the theory paper and the other is an user manual. The user manual will be updated from time to time. Please refer to the documents for more details about the software. 

(1) Installation: 

To run the program, one needs to install the following packages: 

           Python 2.7 (please use the commands "conda install opencv numpy scikit-image cvxopt" and "pip install stopit" to install other software packages)
           Theano 0.9
           Keras 1.2.2  (Note: the software currently does not work well with Keras 2.x because of image dimension ordering problems, please use a previous 1.x version)
           z3 (pip install pySMT z3) 
           
(2) Check the backend of Keras: 

The backend of Keras needs to be changed by editing the ~/.keras/keras.json file into the following format: 

    "backend": "theano",
    "image_dim_ordering": "th"

(3) Usage: 

Use the following command to call the program: 

           python DLV.py

Please use the file ''configuration.py'' to set the parameters for the system to run. 



Xiaowei Huang
