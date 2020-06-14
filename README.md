# Kaggle-Data-Science-Bowl-2018

This repository is the implementation of paper [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) written by Olaf Ronneberger, Philipp Fischer, Thomas Brox.

## Introduction
### Spot Nuclei. Speed Cures.
Imagine speeding up research for almost every disease, from lung cancer and heart disease to rare disorders. The 2018 Data Science Bowl offers our most ambitious mission yet: create an algorithm to automate nucleus detection.

We’ve all seen people suffer from diseases like cancer, heart disease, chronic obstructive pulmonary disease, Alzheimer’s, and diabetes. Many have seen their loved ones pass away. Think how many lives would be transformed if cures came faster.

By automating nucleus detection, we could help unlock cures faster—from rare disorders to the common cold.

I know I am little late to the party but learning has no deadline or time limit. For more information visit [here](https://www.kaggle.com/c/data-science-bowl-2018).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Installation and Setup

* Fork the repo and clone it.
```
git clone https://github.com/Raghav1503/Kaggle-Data-Science-Bowl-2018.git
```
* Go in the repo and setup virtual environment using `python -m virtualenv env` or by using anaconda <br />`conda create --name env`  
* Then activate the environment using `source env/Scripts/activate` (For Windows Users using CMD or Powershell `env/Scripts/activate`) or
`conda activate env`
* Then install the necessary required packages from requirements.txt
```
pip install -r requirements.txt
```
* Then run `pre-commit install`. It will install pre-commit hook for various configurations.

## Data
You can download the dataset provided by [Kaggle](https://www.kaggle.com/c/data-science-bowl-2018/data). <br />
I am using only using data provided by Kaggle and have not added any external data in any form.

After downloading the dataset, you can run the following command
```
cd scripts
python data_sort.py
```
## Model
I am using U-Net Architecture. It is build using the fully convolutional network (FCN), which means that only convolutional layers are used and no dense or recurrent layers are used at all. 

The UNet is a ‘U’ shaped network which consists of three parts:
* The Encoder/Downsampling Path
* Bottleneck
* The Decoder/Upsampling Path

![](assets/u-net-architecture.png)

## Result
![](assets/result.gif)

| Validation Accuracy | Validation Loss | Validation IOU | Validation Precision | Validation Recall |
|:--------------------|:----------------|:---------------|:---------------------|:------------------|
|0.9438619613647461   |0.653819262981414|0.65381926298141|0.9173527359962463    |0.8266566395759583 |

## Room for Improvement : 
* I will try data augmentation techniques to improve accuracy of the model.
* I will try some other architectures for semantic segmentation.
* I will try to convert the above model into instance segmentation rather than semantic segmentation.

***You are welcome to contribute to this repo. Help is any kind is truly welcome.***

## LICENSE
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
