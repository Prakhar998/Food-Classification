## **Food Classification with DenseNet-161**
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
|ResNext-101| 85.4|96.5| **Proposed**
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

![lol](https://cloud.githubusercontent.com/assets/8370623/17981494/f838717a-6ad1-11e6-9391-f0906c80bc1d.jpg)

![warfare](https://i.imgur.com/ZdySvOP.jpg)

_Why Densenet?_

They are very popular now because of the subsequent advantages including their ability to alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters.

![net](https://miro.medium.com/max/875/1*UgVPefF8XKR5aITCzD_5sQ.png)

 ***
 **Image Preprocessing**
 
Pytorch provides the API for loading and preprocessing raw images from the user. However, the dataset and the images in the current state aren't suitable for further stages. 
Successive transformations are introduced in train dataset including <a href="https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.RandomRotation" style="text-decoration:none">Random rotation </a>,<a href="https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.RandomSizedCrop">Random resized crop</a> ,<a href="https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.RandomVerticalFlip">Random horizontal flip</a>, <a href="https://towardsdatascience.com/how-to-improve-your-image-classifier-with-googles-autoaugment-77643f0be0c9">Imagenet policy</a> and at the end <a href="https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Normalize">Normalization</a>.

Image preprocessing effectively handles the problem when the pictures were taken in different environment background which speed up the learning pace and slightly improve the output accuracy.

Transforms for both the training and test data is defined as follows:

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),ImageNetPolicy(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

An illustration has been attached to depict the outcome of the transforms.

***
 **Methodology**

We utilized a system having the following configuration to run the whole program:
|Vendor | Asus (RoG) |
|--|--|
| Processor  | Core I7  |
| Ram  | 8 Gb  |
| GPU | Nvidia GTX - 1050ti  |

System almost spent 4-5 days to process the complicated network structure. The framework for deep learning is the latest edition of PyTorch. We applied the transfer learning method to our model which is by using the pretrained model as a <a href="https://www.kaggle.com/pytorch/densenet121/kernels">checkpoint</a> and continue to train the neural network. Reason is pretty simple, ImageNet has very large amount of dataset and is trained on precised with well equipped systems.

 - At first, use a pretrained model of DenseNet-161. Load a checkpoint from the pretrained model, which is a file that stores all the tensors after months of training for the ImageNet dataset.
 - Second step is to concentrate on the last layer of the network and recognize classes of food images. Remember to shift the classifier from ImageNet to Food-101, 101 means having 101 classes and ImageNet have 1001.
 - Here is a sample code snippet of classifier. You can observe the number of classes have modified to fit with the food dataset. We finally manipulate the softmax. Then we retrain the last layer to get a model based on the pre-trained model.

```
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 500)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, 101)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
```
- Third step is to evaluvate the result. We splitted the dataset into training, test and validation parts. The training examples take 80% of the whole datasets and the rest are for test and validation. By running the evaluvation on the last layer of the network, we obtain the training and test error.
- Finally fine tuning the network framework. We need to give the model initial parameters, and setup optimization process. Example - Dropouts are used to prevent overfitting. Adam was used as an optimizer and every optimizer have different parameters, such as learning rate and learning rate decay. To achieve the minimum loss, we have spent several days to find a sweet spot to tweak them.
Here is the snippet showing the specific betas.
```optimizer = optim.Adam(model.classifier.parameters(), lr=0.001, betas=[0.9, 0.999])```
