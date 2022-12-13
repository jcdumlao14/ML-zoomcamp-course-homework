# PROJECT DESCRIPTION
This project was performed as part of the _ML Zoomcamp Course_, Capstone Project. This course is conducted by [Alexey Grigorev](https://bit.ly/3BxeAoB)

# Grapevine Leaves Classification By Neural Networks and Deep Learning
![image](https://user-images.githubusercontent.com/82657966/206971609-9a753185-19ef-4f5a-9cd0-19ebe336c98e.png)


## 1. Task Description
The Grapevine Leaves are provided a Leaf photos of 11 Vitis Vinifera varieties. I chose this dataset to apply Neural Networks and Deep Learning, check the different models and activation functions and,tune the model parameters and augmentation.  

## 2. Data

### 2.1. About the Data

This dataset contains photos of leaves of 11 grapevine varieties which collected during summer 2020 with my phone camera. The photos were put in folders each named with the corresponding variety.

List of varieties:
1. Auxerrois
2. Cabernet Franc
3. Cabernet Sauvignon
4. Chardonnay
5. Merlot
6. Müller Thurgau
7. Pinot Noir
8. Riesling
9. Sauvignon Blanc
10. Syrah
11. Tempranillo

### 2.2. Data Reference

This Model was built using [kaggle Dataset](https://www.kaggle.com/datasets/maximvlah/grapevine-leaves).

### 2.3. Downloding the Dataset from kaggle

Before starting, you need to have the opendatasets library installed in your system. If it’s not present in your system, use Python’s package manager pip and run:

```
    !pip install opendatasets
```

in a google colab Notebook cell. Python’s opendatasets library is used for downloading open datasets from platforms such as Kaggle.

The process to Download is as follows:

1. Import the opendatasets library

```
    import opendatasets as od
```

2. Now use the download function of the opendatasets library, which as the name suggests, is used to download the dataset. It takes the link to the dataset as an argument.

```
    od.download("https://www.kaggle.com/datasets/maximvlah/grapevine-leaves")
```

3. On executing the above line, it will prompt for Kaggle username. Kaggle username can be fetched from the **Account** tab of the **My Profile** section.

4. On entering the username, it will prompt for Kaggle Key. Again, go to the **Account** tab of the **My Profile** section and click on **Create New API Token**. This will download a _kaggle.json_ file.

5. On opening this file, you will find the _username_ and _key_ in it. Copy the key and paste it into the prompted Jupyter Notebook cell. The content of the downloaded file would look like this:

    `{"username":<KAGGLE USERNAME>,"key":"<KAGGLE KEY>"}`

6. A progress bar will show if the dataset is downloaded completely or not.

7. After successful completion of the download, a folder will be created in the current working directory of your google colab notebook. This folder contains our dataset.

  `[REF: https://www.analyticsvidhya.com/blog/2021/04/how-to-download-kaggle-datasets-using-jupyter-notebook/]`
  
### 2.4. Data Preparation (Split data in Train, Test, Validation)
(You can find codes in grapevine-notebook.ipynb)

For the part, I used two packages `os` and `shutil`.

`os.mkdir` is used to create the destination directories:

**--train**
* Cabernet Sauvignon
* Sauvignon Blanc
* Syrah
* Auxerrois
* Chardonnay
* Merlot
* Cabernet Franc
* Pinot Noir
* Riesling
* Muller Thurgau
* Tempranillo

**--validation**
* Cabernet Sauvignon
* Sauvignon Blanc
* Syrah
* Auxerrois
* Chardonnay
* Merlot
* Cabernet Franc
* Pinot Noir
* Riesling
* Muller Thurgau
* Tempranillo

**--test**
* Cabernet Sauvignon
* Sauvignon Blanc
* Syrah
* Auxerrois
* Chardonnay
* Merlot
* Cabernet Franc
* Pinot Noir
* Riesling
* Muller Thurgau
* Tempranillo

First, rename the files due to their classe by `os.rename`.

Then, `shutil.copy` is used to copy the file from source to destination folders as follow:

from each grapevine-leaves folder (Cabernet Sauvignon',Muller Thurgau,Auxerrois,Syrah,Sauvignon Blanc,Tempranillo,Riesling,Pinot Noir,Chardonnay,Cabernet Franc,Merlot', ...):
* The first 60% of images were copied to _grapevine-leaves/train_ folder
* The next 20% of images were copied to _grapevine-leaves/validation_ folder
* The rest were copied to _grapevine-leaves/test_ folder

### 3. Exploratory Data Analysis

Grapevine Leaves Dataset consists of 1,009 files in 11 different classes:
Viz Random Sample from each class.

![image](https://github.com/jcdumlao14/ML-zoomcamp-course-homework/blob/main/Capstone%20Project-1/viz/Viz%20Random%20Sample%20from%20each%20class.png)

|**Count of Training Sample** |**Class** | **per Count** |
|---|---|---|
|1|Cabernet Sauvignon|67|
|2|Muller Thurgau|73|
|3|Auxerrois|53|
|4|Syrah|71|
|5|Sauvignon Blanc|60|
|6|Tempranillo|42|
|7|Riesling|70|
|8|Pinot Noir|35|
|9|Chardonnay|62|
|10|Cabernet Franc|36|
|11|Merlot|37|

![image](https://github.com/jcdumlao14/ML-zoomcamp-course-homework/blob/main/Capstone%20Project-1/viz/Training%20Data.png)

## 4. Create model
(You can find codes in grapevine-notebook.ipynb)

**TensorFlow** is a library for ML and AI, and **Keras** from tensorfolw provides a Python interface for TensorFlow. In **keras.layers** you can find different layers to creat your model. more info in [keras layers](https://keras.io/api/layers/) To classify the grapevine leaves images.

### 4.1. Training Different Models

* 1. **Neural Networks (MLP)** - Multi-layer perceptron (MLP) is a supplement of a feed-forward neural network. It consists of three types of layers—the input layer, the output layer, and the hidden.
* 2. **Convolutional Neural Network** - CNN (Baseline Model)-The convolutional neural network starts with a separate temporal layer and spatial convolutional layers, followed by a pooling layer.
* 3. **Improving Baseline Model** - Create Deeper Model (Adding more convolution and max pooling layers)
* 4. **Further Improving the Baseline Model** - Reducing the overfitting of the model by using BatchNormalization and Dropout layers and Adding Global Average Pooling instead of Flatten layer.
* 5. **VGG16 model** - VGG16 is a convolutional neural network trained on a subset of the ImageNet dataset.
* 6. **Resnet model** -A residual neural network (ResNet) is an artificial neural network (ANN).

**ResNet model**

![image](https://github.com/jcdumlao14/ML-zoomcamp-course-homework/blob/main/Capstone%20Project-1/RESNET/resnet_model.png)

* 7. **MobileNetv2 model** - MobileNet-v2 is a convolutional neural network that is 53 layers deep.
```
pretrained_mobilenet_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=[height,width, 3])
pretrained_mobilenet_model.trainable=False
mobilenet_model = tf.keras.Sequential([
    pretrained_mobilenet_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(15, activation='softmax')
```
* MobileNetV2 is a general architecture and can be used for multiple use cases. Depending on the use case, it can use different input layer size and different width factors. This allows different width models to reduce the number of multiply-adds and thereby reduce inference cost on mobile devices.

* MobileNetV2 is very similar to the original MobileNet, except that it uses inverted residual blocks with bottlenecking features. It has a drastically lower parameter count than the original MobileNet. MobileNets support any input size greater than 32 x 32, with larger image sizes offering better performance.

    * input_shape -Optional shape tuple, to be specified if you would like to use a model with an input image resolution that is not (224, 224, 3). It should have exactly 3 inputs channels (256, 256, 3).
    * include_top - Boolean, whether to include the fully-connected layer at the top of the network. Defaults to False.
    * weights - String, one of None (random initialization), 'imagenet' (pre-training on ImageNet), or the path to the weights file to be loaded.
    * pooling -
       *avg means that global average pooling will be applied to the output of the last convolutional block, and thus the output of the model will be a 2D tensor.
       *max means that global max pooling will be applied.
    * classifier_activation - A str or callable. The activation function to use on the "top" layer. Ignored unless include_top=True. Set classifier_activation=None to return the logits of the "top" layer. When loading pretrained weights, classifier_activation can only be None or "softmax".
    
**This is the best model**

![image](https://github.com/jcdumlao14/ML-zoomcamp-course-homework/blob/main/Capstone%20Project-1/MobileNet/best%20model%20(mobilenet_model).png)

### 5.1. Checking Predictions with the best models -
* ResNet
    * predictions(resnet_model)
    
    ![image](https://github.com/jcdumlao14/ML-zoomcamp-course-homework/blob/main/Capstone%20Project-1/RESNET/resnet_model%20-%20prediction.png)
    ![image](https://github.com/jcdumlao14/ML-zoomcamp-course-homework/blob/main/Capstone%20Project-1/RESNET/resnet_model-prediction2.png)
    
* MobileNet
    * predictions(mobilenet_model)
    
    ![image](https://github.com/jcdumlao14/ML-zoomcamp-course-homework/blob/main/Capstone%20Project-1/MobileNet/mobilenet_model-prediction.png)
    ![image](https://github.com/jcdumlao14/ML-zoomcamp-course-homework/blob/main/Capstone%20Project-1/MobileNet/mobilenet_model-prediction2.png)

### 5.1. Data Augmentation 
* Display some Randomly Augmented Training Images
![image](https://github.com/jcdumlao14/ML-zoomcamp-course-homework/blob/main/Capstone%20Project-1/viz/randomly%20augmented%20training%20images.png)

### 6.1. Preparing Script
To use the model I prepared different script. Before using script you have to download the dataset from data folder, save it in a capstone project folder in your computer and unzip it.
#### 6.2 Train
Train the model and save the models using checkpoint. Now, we have to choose the best model manually.
#### 6.3. Predict
Predict the output of the model. It is saves using flask.
In terminal you can use this command to run it.
```
gunicorn --bind 0.0.0.0:8888 predict:app
```
#### 6.4. Predict_test and test the model by gunicorn
The test file is the same as notebook test. After runnig predict, you can use following command in new terminal:
```
python predict_test.py
```

#### 6.5. Create pipfile and pipfile.lock
Create pipfiles to have all required file for create model, because for predicting we remove tensorflow dependencies of the model. Tensorflsion ow package is a large package. We use the lighter version called _tflite_. But I prepared pipfile and pipfile.lock with numpy, tensorflow, flask and gunicorn packages.
To create pipfiles, first we need to install `pipenv`. The command is:
```
pip install pipenv
```
then we install packages as follows:
```
pipenv install numpy tensorflow flask gunicorn
```
#### 6.6. Create Lambda Function
Lambda function perform like predict file but it is used for serverless AWS deployement, however we can test it locally.

#### 6.7. Create Docker file
For creating Docker file the requirements are:
```
FROM public.ecr.aws/lambda/python:3.9

RUN pip install keras-image-helper
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl

COPY final_model.h5 .
COPY lambda_function.py .

CMD ["lambda_function.lambda_handler"]
```

#### 6.8. Containerization
follow the commands:
```
docker build -t final_model.h5 .
```
### 6.9. Deployement
#### 6..9.1. Deploy and test the model locally
Run the docker image
```
docker run -it --rm -p 8888:8888 final_model.h5:latest
```

    
