# Convolutional Neural Networks

## Hey There!

This repo consists of some of the code I've written for studying and implementing Convolutional Neural Networks in Python. In Deep Learning, CNNs are a class of networks that are commonly used for analyzing visual imagery. In CNNs the weights, in the form of filters or kernels are shared along the inputs to learn feature represenations and produce feature maps. Sparse connectivity is another important attribute of such networks, i.e. the kernels used are smaller than the image being processed and therefore the connections between the input and output layers are reduced. In the notebook attached, we will work with some convolution operations, build our network with PyTorch and also import and train some popular CNNs available in published research.


**Convolutional_Neural_Networks** Notebook Outline:
- Importing and visualizing data from the CIFAR-10 datset.
- Writing basic classes with CNN layers to visualize the convolution operation.
- Writing the LeNet network for training on our dataset, and writing functions for prediction, given a batch of images.
- Visualizing the output of one convolution layer of LeNet and studying changes in training on using ReLU instead of Tanh activation.
- Downloading and visualizing data from the MNIST dataset.
- Training on the MNIST dataset with LeNet.
- Importing the VGG-16 batch-normalized network, modifying it to predict on the CIFAR-10 dataset by training only on the last layer.
- Repeating the above process by importing the pre-trained model for transfer learning.
- Importing and training on the ResNet and InceptionNet networks as well.

Note: The CIFAR-10 dataset is a challenging one to predict on, and appreciable, state-of-the-art accuracies can only be acheived after hours of training with deep networks along with dedicated hardware support. Since the network has only been trained for a very short period of time and that too on the last layer, we can only hit accuracies in the range of ~50%. The training process might not also be very smooth.

**CNNs_continued** Notebook Outline:
- Using torchvision and its packaged to load, transform and visualize data stored locally on our machine.
- Using a common method of visualization known occlusion where we study how the performance of a network on an image is affected on removing certain sections of it.
- Visualizing kernels of the Alexnet network to see what kind of features are picked up in the image.
- Writing a custom deep neural networks to train on the CIFAR-10 dataset.

Note: The data.zip file used in the notebook has been attached in this repository. It contains some images from the ImageNet dataset that is commonly used for training Convolutional Neural Networks. Since the pre-trained models we import have already been trained by experts on this data, we do not need to implement training by ourself and can focus on visualization and analysis.

**SqueezeNet** Notebook Outline:
- Importing the CIFAR-10 dataset, applying transformations and training on the SqueezeNet architecture, imported (pre-trained).
