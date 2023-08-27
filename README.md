# Deep-Learning-Based-Multiresolution-Parameterization-for-Spatially-Adaptive-Model-Updating
This repository contains the codes for the paper: [**Deep Learning-Based Multiresolution Parameterization for Spatially Adaptive Model Updating**](https://onepetro.org/spersc/proceedings-abstract/23RSC/3-23RSC/D010S001R014/518346). Results are based on a set of 2D Gaussian models in a straight-ray travel-time tomography example.

## Summary
This paper presents a new deep learning-based parameterization approach for model calibration with two important properties: spatial adaptivity and multiresolution representation. The method aims to establish a spatially adaptive multiresolution latent space representation of subsurface property maps that enables local updates to property distributions at different scales. The deep learning model consists of a convolutional neural network architecture that learns successive mapping across multiple scales, from a coarse grid to increasingly finer grid representations. Once trained, the architecture learns latent spaces that encode spatial information across multiple scales. The resulting parameterization can facilitate the integration of data at different resolutions while enabling updates to the desired regions of the domain. Unlike the standard deep learning latent variables that are not localized and do not provide spatial adaptivity, the presented method enables local update capability that can be exploited to incorporate expert knowledge into assisted model updating workflows. 

## Motivation
![image](https://github.com/mahammadvaliyev/Deep-Learning-Based-Multiresolution-Parameterization-for-Spatially-Adaptive-Model-Updating/assets/68789630/f05acf2f-fad7-4acd-9a4f-c3bd89622bfa)


## Method
![image](https://github.com/mahammadvaliyev/Deep-Learning-Based-Multiresolution-Parameterization-for-Spatially-Adaptive-Model-Updating/assets/68789630/f2195080-0618-464a-aa2c-abf4cab3e4db)
</br>
</br>
</br>
![image](https://github.com/mahammadvaliyev/Deep-Learning-Based-Multiresolution-Parameterization-for-Spatially-Adaptive-Model-Updating/assets/68789630/d7f95313-57ed-426e-8cb6-2353ac7ae81f)

## Requirements
Codes were tested using the following setup:
- Windows 11
- Python 3.8.8
- tensorflow 2.6, keras 2.6
- See other required libraries in the requirements.txt file

## Data
- Gaussian data used for training architectures can be found at: https://rb.gy/vc98w
