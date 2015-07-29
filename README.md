# Machine-Learning-Library

This is a machine learning library I built in C++ with Eigen. 

For more information about Eigen:
http://eigen.tuxfamily.org/index.php?title=Main_Page

Methods supported:

**classification : logistic regression, linear discriminant analysi, naive bayes **

**regression : ridge regression, kernel ridge regression, robust regression**

**cluster: kmeans, spectral cluster**

**other unsupervised: kernel density estimation, principal component analysis**


To use this library, you need Eigen, which I already include. Each method could be used independently, you can check the example case comes with each method. ReadMatrix is used by all method to preprocess raw data from file with different format. 
