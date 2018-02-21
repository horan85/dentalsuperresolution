

# Deep Learning-Based Super-Resolution Applied toDental Computed Tomography

**A github repository for enhancing dental computer tomography images**

In this repository You can find our code which impelments a superroslution neural network, descird in the paper:

*Deep Learning-Based Super-Resolution Applied toDental Computed Tomography*
Janka Hatvani, András Horváth, Jerome Michetti, Adrian Basarab, Denis Kouame and Miklós Gyongy

Submitted to:
IEEE TRANSACTIONS ON RADIATION AND PLASMA MEDICAL SCIENCES

### Prerequisites-Installing

To run our code You need to install [Python](https://www.python.org/)  (*v3.5*) and  [Tensorflow](https://www.tensorflow.org/) (v1.3.0) and that is all.

### Running our code
 Our training scripts were implemented as a single file, all You have to do to train your models on your own data is to change the data loading part (marked as *#preread images*) and run the script.
 The implementation of the U-NET based network can be found in  [Unet_train.py](https://github.com/horan85/dentalsuperresolution/blob/master/Unet_train.py)
 and the subpixel based implementation can be found in  [SubCNN_train.py](https://github.com/horan85/dentalsuperresolution/blob/master/SubCNN_train.py)

## Example
An example image showing how the subpixel based CNN netwrok can improve resoltuion and increase accuracy in root canal segmentation is displayed on the following image:
![alt text](https://github.com/horan85/dentalsuperresolution/raw/master/tooth11a_segment.png)

## Authors
**Janka Hatvani 
Andras Horvath 
Jerome Michetti 
Adrian Basarab 
Denis Kouame 
Miklos Gyong** 
