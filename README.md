## **Food Classification**
Food applications based on image food classification opened a new realm of challenges for computer vision. Despite of various attempts to solve the same. We believe that better results can be obtained by performing successive augmentation techniques on the data followed by a better pretrained model. To evaluate our proposed architecture, we have conducted experimental results on a benchmark dataset (Food-101-Kmader). Results demonstrate that our solution shows better performance with respect to existing approaches. (e.g Top-1 accuracy as 93.27% and Top-5 accuracy around 99.02%)

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
