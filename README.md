# GenSketch
###  Ashwin Jeyaseelan, Disha Das, Naila Fatima, Tarunkumar Venkata Pasumarthi

## Introduction
Objective is to make models that sketch (vector drawings) of different classes. 
We experimented with different models based on the Sketch-RNN model [1], which is a VAE-RNN that generates sketches of images. 
The models used to generate sketches are a VAE-CNN based off of [1] but with a CNN instead of an RNN for the input, DCGAN, and Sketch-pix2seq [2].

Video Intro: https://drive.google.com/drive/folders/1za0atAODg4mE1kjNKlsja9o9kiFmFfUY 

## Dataset
We used Google’s QuickDraw dataset [5] to generate the images and train our classifier. 
The QuickDraw dataset consists of 50 million drawings and 345 categories. 
For our project, we decided to focus on 10 classes of drawings which are: tree, t-shirt, ice cream, fish, face,  car, bowtie, apple, flamingo, and sheep. We specifically worked with the simplified numpy bitmap dataset which contains the 28 by 28 grayscale images in numpy format.

## Code
Code was written in colab notebooks: https://drive.google.com/drive/folders/1za0atAODg4mE1kjNKlsja9o9kiFmFfUY 
* `classifier.ipynb` - deep convolutional network that learned to classify the ten classes of images. The purpose was to have a trained model that can help validate the accuracy of the image outputs of our generated models. 
* `Pix2Seq.ipynb` - VAE-CNN encoder - RNN decoder model. Code based off of [2] 
* `DCGAN_FINAL1.ipynb` - GAN model
* `VAE_models.ipynb` - VAE - CNN encoder - CNN decoder model
* `VAE_all_class.ipynb` - VAE-CNN but with Pix2Seq [2] loss function (omits KL divergence)

## References
1. David Ha and Douglas Eck. A neural representation of sketch drawings.arXiv preprint arXiv:1704.03477, April 2017.
2. Yajing Chen, Shikui Tu, Yuqi Yi, and Lei Xu. Sketch-pix2seq: a model to generate sketches of multiple categories. arXiv preprintarXiv:1709.04121, 2017
3. Yuncheng Wu, Yundi Fei. Sketch-RNN-GAN. Retrieved from https://1drv.ms/b/s!AkpHFm7pqfBnhbJ1PxNpLtlfx452jw. 
4. Forrest Huang, John F. Canny. Sketchforme: Composing Sketched Scenes from Text Descriptions for Interactive Applications. Retrieved from https://arxiv.org/abs/1904.04399. 
5. Dataset: https://github.com/googlecreativelab/quickdraw-dataset 
