# Potato Disease Classification with VGG16

This project uses a modified VGG16 model to classify potato diseases into three categories: 'Potato___Early_Blight', 'Potato___Late_Blight', and 'Potato___Healthy'. The model is trained on a dataset containing images of diseased and healthy potato plants.

## Overview

This project aims to develop a deep learning model for the classification of potato diseases based on images. The model uses a modified VGG16 architecture to achieve high accuracy in distinguishing between 'Potato___Early_Blight', 'Potato___Late_Blight', and 'Potato___Healthy' classes.

## Dataset

The training and testing data consist of a labeled dataset containing images of diseased and healthy potato plants. The dataset used in this project can be found at  https://www.kaggle.com/datasets/nafishamoin/new-bangladeshi-crop-disease?rvi=1. Please download and organize the dataset according to the provided instructions before running the code.

## Model Architecture

The model architecture is based on the VGG16 convolutional neural network. Additional layers have been added to adapt the model to the specific task of potato disease classification. Below is a summary of the model architecture:

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 vgg16 (Functional)          (None, 7, 7, 512)         14714688  
                                                                 
 conv2d (Conv2D)             (None, 7, 7, 256)         524544    
                                                                 
 max_pooling2d (MaxPooling2  (None, 3, 3, 256)         0         
 D)                                                              
                                                                 
 dropout (Dropout)           (None, 3, 3, 256)         0         
                                                                 
 conv2d_1 (Conv2D)           (None, 3, 3, 128)         131200    
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 1, 1, 128)         0         
 g2D)                                                            
                                                                 
 dropout_1 (Dropout)         (None, 1, 1, 128)         0         
                                                                 
 flatten (Flatten)           (None, 128)               0         
                                                                 
 dense (Dense)               (None, 3)                 387       
                                                                 
=================================================================
Total params: 15370819 (58.64 MB)
Trainable params: 656131 (2.50 MB)
Non-trainable params: 14714688 (56.13 MB)

## Training

The model was trained for 20 epochs using the Adam optimizer and categorical crossentropy loss. The training process and hyperparameters are defined in the training code. 

## Evaluation

The model achieved an accuracy of 93.83% on the test set, with a corresponding loss of 0.1517.



