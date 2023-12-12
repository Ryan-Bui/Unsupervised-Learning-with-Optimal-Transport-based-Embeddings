![](UTA-DataScience-Logo.png)

# Unsupervised Learning Using Optimal Transport Based Embeddings

* This project aims to provide a comprehensive evaluation of optimal transport-based techniques in real-world scenarios.

## Overview
  * The goal of this project is to gain a greater understanding of optimal transport based manifold learning by comparing it to other more widely used manifold learning techniques on a variety of datasets. 
  * The approach in this repository is to use a variety of different types of manifold learning embeddings (including our optimal transport based embedding) on different types of datasets. We will then use these embeddings in different clustering algorithms to gauge the performance of these embeddings.
 

## Summary of Workdone

### Data

* Yale Face Dataset 
  * Type: The database contains 165 single light source images of 15 subjects. One for each of the following facial expressions or configurations: center-light, w/glasses, happy, left-light, w/no glasses, normal, right-light, sad, sleepy, surprised, and wink.
  * Origin: This is a public dataset provided by Yale.  
  * Size: The total size of the compressed dataset is about 12mb. Each Image is 320x243.
![](yaleSamples2.gif)

* Newsgroup Dataset
  * Type: The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups.
  * This data does come with a .data file which can be read in as a pandas dataframe where the first column represents the document ID the second column represents the word ID and the 3 column represents how many times that word was used in that particular document.
  * This data also comes with a .label file which represents the genre of each document.
  * Origin: It was originally collected by Ken Lang.  
  * Size: The total size of the compressed database is about 14mb.  
#### Preprocessing / Clean up

* For the Yale Face Dataset I needed to transform the image into a numpy array of pixel values and make sure to grayscale the image.
* I also decided to use a package called MTCNN (Multi-Task Cascaded Convolutional Neural Networks) which detects faces and facial landmarks on images.
* For the Newsgroup dataset I turned the original data file into a pandas data frame where each row represented a document and each column represents a word in the document and the value in the column represents how many time the word appears in that particular document.


#### Data Visualization

![](yaleimages.png)
![](newsgroupdist.png)
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
* Yale Face Dataset Embeddings:
![](isomap.png)
![](mds.png)
![](tsne.png)

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

### Data

* http://qwone.com/~jason/20Newsgroups/
* https://www.kaggle.com/datasets/olgabelitskaya/yale-face-database



