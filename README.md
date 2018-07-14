# spec-img-finesse

In their work Makantasis et al. (2015) show that using CNN, hyperspectral images can be successfully classified. CNN can encode the spectral and spatial features of pixel. The
low-to-high hierarchy of features improve the performance of classification greatly. In our CNN implementation we extend and optimize their method with layer pruning and layer
compression methods. 

Every plant has its unique spectral ’signature’ over the electromagnetic spectrum, and this can be captured using hyperspectral sensors. Treating the hyperspectral bands in the images as features and each pixel as a sample, plants are classified with convolutional neural networks(CNN)
and support vector machines(SVM). CNN optimization helps prevent overfitting, accelerate inference, and reduce the resources it uses with respect to memory, battery and
computational power.

Keras 2.1.5 was used in conjunction with Tensorflow 1.7.0. Indian Pines dataset was used. 83.9% test accuracy was achieved using SVM with Polynomial kernel, and 99.2% test accuracy was achieved with CNN. 

More information can be found on the project report, Vegetation Classification using Hyperspectral Images.

This was a project for ECE 228 - Machine Learning and Physical Applications class. 
My contribution was the CNN model and CNN optimization. 

