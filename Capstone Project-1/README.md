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
(You can find codes in DataPreparation.ipynb)

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

### 2.4. Exploratory Data Analysis

Grape Vine Leaves Dataset consists of 2036 pictures in 6 different folder:

|**Count of Training Sample** |**Class** | **/Count** |
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
![image](https://user-imaa-efef6283510c.png)



