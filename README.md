# Melanoma Detection using CNN
An assignment to create, compile and train a Convolutional Neural Network (CNN) model to classify 9 types of skin cancer accurately.

## Table of Contents
* [General Info](#general-information)
* [Project Contents](#project-contents)
* [Conclusion](#conclusion)
* [Software and Library Versions](#software-and-library-versions)
* [Acknowledgements](#acknowledgements)

### General Information
To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths.

#### Data Set Brief Information
A zipped file containing 2 folders - Train and Test are given.
Train and Testfolders are present for training-validation and predicting the model respectively.
Each folder contains sub-folders with names of 9 types of skin cancer - actinic keratosis, basal cell carcinoma, dermatofibroma, melanoma, nevus, pigmented benign keratosis, seborrheic keratosis, squamous cell carcinoma, vascular lesion.
These sub-folders contains respective skin cancer images.

#### Business Objective
Create, compile and train a CNN based model which can classify 9 types of skin cancer accurately.

#### Business Solution
The trained model is provided as a solution, which can evaluate images and alert the dermatologists about the presence of melanoma, that can reduce a lot of manual effort needed in diagnosis.


### Project Contents
* **Melanoma Detection.ipynb** - Jupyter Notebook for Melanoma Detection using CNN (Language : Python)
* **Melanoma Detection.pdf** - A PDF File for briefly explanation the assignement steps and CNN model outcome.
* **README.md** - Readme file


### Conclusion
Any Image classification can be solved using CNN.
* **Data Downloading-Unzipping** → Dataset was downloaded from shared drive and unzipped.
* **Data Reading/Data Understanding** → Defined the path for train and test images.
* **Dataset Creation** → Created train & validation dataset from the train directory with a batch size of 32. Also, resized images to 180x180.
* **Dataset Visualization** → Created a code to visualize one instance of all the nine classes present in the dataset.
* **Model Building & training** → Created a CNN model to predict 9 types of skin cancer classes. Normalized pixel value to range [0,1]. Chose optimizer as Adam and Loss as Sparse Categorical Cross Entropy. Trained the model for 20 epochs. Observed the model performance - Overfitting of train dataset.
    * Training Dataset => loss: 0.35 - accuracy: 86.05 %
	* Validation Dataset => loss: 2.19 - accuracy: 52.13 %
* **Data Augmentation** → Created a augmentation layer to enhance/preprocess images for classification.
* **Model Building & training on the Augmented data** → Created a CNN model to predict 9 types of skin cancer classes. Preprocessed images using image augmentation layer. Normalized pixel value to range [0,1]. Added drop out layer to avoid overfitting. Chose optimizer as Adam and Loss as Sparse Categorical Cross Entropy. Trained the model for 20 epochs. Observed the model performance - Avoided overfitting but training dataset accuracy reduced and validation dataset accuracy is not improving.
    * Training Dataset => loss: 1.44 - accuracy: 48.38 %
	* Validation Dataset => loss: 1.43 - accuracy: 48.10 %
* **Class Distribution** → Examined class distribution.
    * seborrheic keratosis has the least number of samples only 77.
    * pigmented benign keratosis (462 Samples), melanoma (438 Samples), basal cell carcinoma (376 Samples), and nevus (357 Samples) classes dominates the data in terms proportionate number of samples .
* **Handling class imbalances** → Rectified class imbalances present in the training dataset with Augmentor library.
* **Model Building & training on the rectified class imbalance data** → Created a CNN model to predict 9 types of skin cancer classes. Normalized pixel value to range [0,1]. Added drop out layer to avoid overfitting. Chose optimizer as Adam and Loss as Sparse Categorical Cross Entropy. Trained the model for 20 and 30 epochs. Observed the model performance - Model trained for 30 epochs performs better in terms of accuracy and loss of training and validation dataset respectively. Although overfitting was avoided, the test data accuracy and loss is very less when compared to trained model.
    * Training Dataset => loss: **0.37** - accuracy: **85.35** %
	* Validation Dataset => loss: **0.64** - accuracy: **80.70** %
	* Test Dataset => loss: **4.6** - accuracy: **39.83** %

#### Recommendation
Model trained on the rectified class imbalance data with 30 epochs is chosen to classify 9 types of skin cancer accurately.


### Software and Library Versions
* ![Jupyter Notebook](https://img.shields.io/static/v1?label=Jupyter%20Notebook&message=4.9.2&color=blue&labelColor=grey)

* ![NumPy](https://img.shields.io/static/v1?label=numpy&message=1.21.6&color=blue&labelColor=grey)

* ![Pandas](https://img.shields.io/static/v1?label=pandas&message=1.3.5&color=blue&labelColor=grey)

* ![Tensorflow](https://img.shields.io/static/v1?label=Tensorflow&message=2.9.2&color=blue&labelColor=grey)

* ![Keras](https://img.shields.io/static/v1?label=Keras&message=2.9.0&color=blue&labelColor=grey)

* ![matplotlib](https://img.shields.io/static/v1?label=matplotlib&message=3.2.2&color=blue&labelColor=grey)

* ![seaborn](https://img.shields.io/static/v1?label=seaborn&message=0.11.2&color=blue&labelColor=grey)

* ![sklearn](https://img.shields.io/static/v1?label=sklearn&message=1.0.2&color=blue&labelColor=grey)


### Acknowledgements
This CNN based model classification assignment was done as part of [Upgrad](https://www.upgrad.com/ ) - **Master of Science in Machine Learning & Artificial Intelligence** programme.


### Contact
Created by [Sanjeev Surendran](https://github.com/Sanjeev-Surendran)


<!-- ## License -->
<!-- This project is not a open source and sharing the project files is prohibited. -->
