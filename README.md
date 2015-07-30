# Machine-Learning-Library

This is a machine learning library I built in C++ with Eigen. 

Methods supported:

**Classification** : logistic regression, linear discriminant analysi, naive bayes

**Regression** : ridge regression, kernel ridge regression, robust regression

**Cluster**: kmeans, spectral cluster**

**Other unsupervised**: kernel density estimation, principal component analysis


To use this library, you need Eigen, which I already include. Each method could be used independently, you can check the example case comes with each method. ReadMatrix is used by all method to preprocess raw data from file with different format. 

*Set compiler optimization as -O3 if gcc, for visual studio the easy way to change compile option is just use release mode.* 


For more information about Eigen:
http://eigen.tuxfamily.org/index.php?title=Main_Page
