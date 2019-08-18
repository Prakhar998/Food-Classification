## **Food Classification**
![foodbanner](https://www.vision.ee.ethz.ch/datasets_extra/food-101/static/img/food-101.jpg)

Food applications based on image food classification opened a new realm of challenges for computer vision. Despite of various attempts to solve the same. We believe that better results can be obtained by performing successive augmentation techniques on the data followed by a better pretrained model. To evaluate our proposed architecture, we have conducted experimental results on a benchmark dataset (Food-101). Results demonstrate that our solution shows better performance with respect to existing approaches. (e.g Top-1 accuracy as 93.27% and Top-5 accuracy around 99.02%)

| Method 	| Top - 1  	| Top - 5  	| Publication  	|
|---	|---	|---	|---	|
| HoG    	|8.85   	| - | ECCV2014  	|
|   SURF BoW-1024 	|  33.47  	|   -	| ECCV2014  	|
|   SURF IFV-64 	|  44.79   	|   -	|   ECCV2014 	|
|    SURF IFV-64 + Color Bow-64	|  49.40 	|   -	|   ECCV2014   	|
|   IFV	| 38.88   	| -  	|  ECCV2014  	|
|  RF	|   37.72 	| -  	|   ECCV2014  	|
|   RCF	|   28.46 	| -  	|    ECCV2014	|
|   MLDS 	|    42.63  	| -  	|  ECCV2014	|
|  RFDC	|   50.76   	|  - 	|   ECCV2014 	|
|  SELC 	|     55.89 	|   -	|  CVIU2016 	|
|   AlexNet-CNN 	|  56.40  	|   -	|    ECCV2014	|
|  DCNN-FOOD  	|  70.41  	|   - 	|   ICME2015	|
|   DeepFood 	|   77.4   	|   93.7	|  COST2016 	|
| Inception V3  	|  88.28  	|   96.88 	|   ECCVW2016 	|
|   ResNet-200	|   88.38 	|   	97.85 |    CVPR2016	|
|   WRN 	|   88.72 	|   	 97.92|   BMVC2016	|
|   WISeR 	|   90.27 	|   98.71	|   UNIUD2016 	|
|   **DenseNet - 161**	|  **93.26** 	|   **99.01**	|  **Proposed** 	|


### The Challenge
<hr>

> *The first goal is to be **able to automatically classify an unknown image** using the dataset, but beyond this there are a number of possibilities for looking at what regions / image components are important for making classifications, identify new types of food as combinations of existing tags, build object detectors which can find similar objects in a full scene.*


### Approach
<hr>

**Dataset**

Deep learning-based algorithms require large dataset. Foodspoting's FOOD-101  dataset contains a number of different subsets of the full food-101 data. The idea is to make a more exciting simple training set for image analysis than CIFAR10 or MNIST. For this reason the data includes massively downscaled versions of the images to enable quick tests. The data has been reformatted as HDF5 and specifically Keras HDF5Matrix which allows them to be easily read in. The file names indicate the contents of the file. For example

-   food_c101_n1000_r384x384x3.h5 means there are 101 categories represented, with n=1000 images, that have a resolution of 384x384x3 (RGB, uint8)
    
-   food_test_c101_n1000_r32x32x1.h5 means the data is part of the validation set, has 101 categories represented, with n=1000 images, that have a resolution of 32x32x1 (float32 from -1 to 1)

***
**Model**

Convolutional neural networks increasingly became powerful in large scale image recognition. Alexnet introduced in ILSVRC 2012 had 60 million parameters with 650,000 neurons, consisting of five convolutional layers. Those layers are followed by max pooling,  globally connected layers and around 1000 softmax layers.
With this inspiration, several other architectures were produced to provide better solution. Few of honourable mentions include ZFNet by Zeiler and Fergus, VGGNet by Simonyan et al., GoogLeNet (Inception-v1)  by Szegedy et al and ResNet  by He et al.
 
**Dense Convolutional Network (DenseNet) [arixv 1608.0699]** is another state-of-the-art CNN architecture inspired by the cascade-correlation learning architecture proposed in NIPS, 1989. The architecture connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections—one between each layer and its subsequent layer—our network has L(L+1) 2 direct connections. For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers.

![warfare](https://i.imgur.com/ZdySvOP.jpg)

_Why Densenet?_

They are very popular now because of the subsequent advantages including their ability to alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters.

![net](https://miro.medium.com/max/875/1*UgVPefF8XKR5aITCzD_5sQ.png)

