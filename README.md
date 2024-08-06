Facial Expression Recognition
This repository contains a complete pipeline for facial expression recognition using TensorFlow and Keras. The project involves data preprocessing, model training, and evaluation. It uses a Convolutional Neural Network (CNN) to classify images of facial expressions into different emotion categories.

Table of Contents
Project Overview
Data Preparation
Model Architecture
Dependencies
Training the Model
Usage
License
Project Overview
The goal of this project is to develop a machine learning model that can classify images based on facial expressions. The model is trained on a dataset of facial images, each labeled with an emotion category.

The project consists of the following steps:

Data Preparation: Organizing and splitting the dataset into training and testing subsets.
Model Building: Designing and compiling a Convolutional Neural Network (CNN).
Model Training: Training the model on the prepared dataset.
Data Preparation
Cloning the Repository
To get started, clone the repository:

Data Organization
The dataset is organized into two main directories: training and testing. Each directory is further divided into subdirectories for each emotion category. The data is split with 80% allocated for training and 20% for testing.

Here’s how the data is prepared:

Loading Data: The legend.csv file is read to extract image paths and their corresponding emotion labels.
Creating Directories: master_data/training and master_data/testing directories are created, with subdirectories for each emotion category.
Copying Images: Images are split into training and testing sets and copied to the respective directories.
Model Architecture
The model is a Convolutional Neural Network (CNN) designed for image classification. It includes the following layers:

Convolutional Layers: Three convolutional layers with increasing filter sizes to extract features from images.
MaxPooling Layers: Followed by max pooling layers to reduce dimensionality.
Flatten Layer: Flattens the 3D output of the last convolutional layer into a 1D vector.
Dense Layers: A dense layer with 1024 neurons followed by a softmax output layer to classify the images into 8 emotion categories.
The model is compiled with the Adam optimizer and categorical crossentropy loss function.

Dependencies
The following libraries are required for this project:

TensorFlow
Keras
NumPy
OS
Shutil
CSV


The model is trained using the fit method from Keras. It uses early stopping to prevent overfitting by monitoring the validation accuracy.

Usage
To use this code:

Prepare Data: Ensure that the dataset is correctly organized and split into training and testing sets.
Build and Compile Model: Define the CNN architecture and compile it.
Train Model: Use the training script to fit the model on the training data.
Evaluate Model: Evaluate the model performance using the test data.
Feel free to modify and extend the model or preprocessing steps based on your specific requirements.
