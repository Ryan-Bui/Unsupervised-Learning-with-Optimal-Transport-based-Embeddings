![](UTA-DataScience-Logo.png)

# Unsupervised Learning Using Optimal Transport Based Embeddings

* This project aims to provide a comprehensive evaluation of optimal transport-based techniques in real-world scenarios.

## Overview
  * The goal of this project is to gain a greater understanding of optimal transport based manifold learning by comparing it to other more widely used manifold learning techniques on a variety of datasets. 
  * The approach in this repository is to use a variety of different types of manifold learning embeddings (including our optimal transport based embedding) on different types of datasets. We will then use these embeddings in different clustering algorithms to gauge the performance of these embeddings.
 

## Summary of Workdone

### Data

* Data: Yale Face Dataset 
  * Type: The database contains 165 single light source images of 15 subjects. One for each of the following facial expressions or configurations: center-light, w/glasses, happy, left-light, w/no glasses, normal, right-light, sad, sleepy, surprised, and wink.     
  * Size: The total size of the compressed dataset is about 12mb. Each Image is 320x243.
![](yaleSamples2.gif)

#### Preprocessing / Clean up

* For this dataset I needed to transform the image into a numpy array of pixel values and make sure to grayscale the image.
* I also decided to use a package called MTCNN (Multi-Task Cascaded Convolutional Neural Networks) which detects faces and facial landmarks on images.

#### Data Visualization

![](yaleimages.png)

### Problem Formulation

* Define:
  * After pre-processing the images each image will be of size 224x224 and these will be our inputs. Our output will be an embedding
  * Embeddings:
    * As of right now there will be 7 methods that I will test out which are:
      * MDS(multidimensional scaling)
      * Isomap
      * Locally Linear Embedding (LLE)
      * t-distributed Stochastic Neighbor Embedding (t-SNE)
      * LTSA (Local tangent space alignment)
      * Diffusion Maps
      * Variations of Wassmap

  * Each embedding has its own parameters that we will need to tune and we will experimetn with different parameter setting to try and get the best clustering performance.



### Performance Comparison

* We will have multiple performance metrics which include:
   * Normalized Mutual Information (NMI)
   * Accuracy
   * F1 Score
   * Adjusted Rand Index (ARI)
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * Apply this package to other data. For example, how to use the model you trained.
   * Use this package to perform their own study.
* Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: various functions that are used in cleaning and visualizing data.
  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * models.py: Contains functions that build the various models.
  * training-model-1.ipynb: Trains the first model and saves model during training.
  * training-model-2.ipynb: Trains the second model and saves model during training.
  * training-model-3.ipynb: Trains the third model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.

* Note that all of these notebooks should contain enough text for someone to understand what is happening.

### Software Setup
* List all of the required packages.
* If not standard, provide or point to instruction for installing the packages.
* Describe how to install your package.

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* Provide any references.
