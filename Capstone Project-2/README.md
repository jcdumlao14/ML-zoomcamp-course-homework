# PROJECT DESCRIPTION
This project was performed as part of the _ML Zoomcamp Course_, Capstone Project-2. This course is conducted by [Alexey Grigorev](https://bit.ly/3BxeAoB)

# Heart Disease Classification Analysis

![image](https://user-images.githubusercontent.com/82657966/213832332-73d03238-9c40-4791-9411-281cfd963c4f.png)

## Dataset Reference:
This Model was built using [kaggle Dataset](https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final).

# **About the Data**

**Heart disease** is also known as Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year which is about 32% of all deaths globally. CVDs are a group of disorders of the heart and blood vessels and include coronary heart disease, cerebrovascular disease, rheumatic heart disease, and other conditions. Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age.

We have curated this dataset by combining different datasets already available independently but not combined before. We have combined them over 11 common features which makes it the largest heart disease dataset available for research purposes. The five datasets used for its curation are:

## Data Description
This dataset consists of **11 features** and **a target variable**. It has **6 nominal variables** and **5 numeric variables**. The detailed description of all the features are as follows:

| **Features** | **Definitions** |
|---|---|
| age | Patients Age in years **(Numeric)**|
| sex | Gender of patient **(Male - 1, Female - 0) (Nominal)**|
| chest_pain_ type | Type of chest pain experienced by patient categorized into **1 typical, 2 typical angina, 3 non- anginal pain, 4 asymptomatic (Nominal)**|
| resting_bp_s | Level of blood pressure at resting mode in mm/HG **(Numerical)**|
| cholestrol| Serum cholestrol in mg/dl **(Numeric)**|
| fasting_blood_sugar| Blood sugar levels on fasting > 120 mg/dl represents as **1 in case of true** and **0 as false (Nominal)**|
| resting_ecg| Result of electrocardiogram while at rest are represented in **3 distinct values 0 : Normal 1: Abnormality in ST-T wave 2: Left ventricular hypertrophy (Nominal)**|
| max_heart_rate| Maximum heart rate achieved **(Numeric)**|
| exercise_angina| Angina induced by exercise **0 depicting NO 1 depicting Yes (Nominal)**|
| oldpeak| Exercise induced ST-depression in comparison with the state of rest **(Numeric)**|
| ST slope| ST segment measured in terms of slope during peak exercise **0: Normal 1: Upsloping 2: Flat 3: Downsloping (Nominal)**|
|Target variable|
| target| It is the target variable which we have to predict **1 means patient is suffering from heart risk** and **0 means patient is normal.(nom)**|


## Features Characteristics

**1. Features Type**
|**Categorical**|**Numerical**|
|---|---|
|sex|age|
|chest_pain_type|resting_bp_s|
|resting_ecg|cholesterol|
|st_slope|fasting_blood_sugar|	
| |max_heart_rate|	
| |exercise_angina|
| |oldpeak|

**2.Correlated Features**

**_AGE_** had the highest correaltion with **_I_**

**3.High-Risk Features**

**_CHILDREN_, _SMOKER_, _REGION_** had the highest risks, respectively.

**4.Mutual Information**

**_CHILDREN_** had the highest mutual information.

## Evaluation Metrics
**_AUC_ROC_Curve_ and _RMSE_** were used as evaluation Metrics.

# **3. Exploratory Data Analysis**
(You can find codes in heartDisease.ipynb)

![image](https://github.com/jcdumlao14/ML-zoomcamp-course-homework/blob/main/Capstone%20Project-2/image/GenderDistribution.png)

* These plot indicates that more male patients(350- normal 559- have heart Disease for male) accounts for heart disease in comparison to female(211-normal, 70-heart disease female) 

## **Age Distribution**
* Normal Patients

![image](https://github.com/jcdumlao14/ML-zoomcamp-course-homework/blob/main/Capstone%20Project-2/image/Normal%20P.png)

* Heart Disease Patients

![image](https://github.com/jcdumlao14/ML-zoomcamp-course-homework/blob/main/Capstone%20Project-2/image/Heart%20DiseaseP.png)

* The mean age for heart disease patients is around 58 to 60 years.

## Classification Models Used
We will use the following classification models to test our theory of whether to predict 1 means patient is suffering from heart risk and 0 means patient is normal, given the above fields: 

* [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
* [Decision Trees](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
* [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* [XGBoost](https://xgboost.readthedocs.io/en/latest/python/index.html)

## Prediction Model
By evaluating different models, _Random_Forest_ achieved the best result.

![image](https://github.com/jcdumlao14/ML-zoomcamp-course-homework/blob/main/Capstone%20Project-2/image/best%20model.png)

![image](https://github.com/jcdumlao14/ML-zoomcamp-course-homework/blob/main/Capstone%20Project-2/image/Val.png)


## FILE DESCRIPTION
Folder Midterm Project includes following files:

|**File Name**|**Description**|
|---|---|
|.csv|Dataset|
|heartDisease.ipynb|Data preparation and cleaning & Exploratory Data Analysis|
| |Feature important Analysis & Model selection|
|train.py|Training the final model|
| |Saved model by pickle|
|predict_test.py|Loading the model & Serving it via a web service (with Flask)|
|predict_test.py|Testing the model|
|Pipfile & Pipfile.lock|Python virtual environment, Pipenv file|
|Dockerfile|Environment management, Docker, for running file|

[RAW DATA -]()

[heartDisease.ipynb]()


# RUNNING INSTRUCTION
1. Copy scripts (train, predict and predict_test), pipenv file and Dockerfile to a folder
2. Run Windows Terminal Linux (WSL2) in the that folder
3. Install `pipenv`
   ```
   pip install pipenv
   ```
4. Install essential packages
   ```
   pipenv install numpy pandas scikit-learn==1.0 flask xgboost
   ```
5. Install Docker
 - SingUp for a DockerID in [Docker](https://hub.docker.com/)
 - Download & Intall [Docker Desktop](https://docs.docker.com/desktop/windows/install/)
 
6. In WSL2 run the following command to create the image `capstone-project'
   ```
   docker build -t capstone-project .
   ```
7. Run Docker to loading model
   ```
   docker run -it --rm -p 8888:8888 capstone-project
   ```
8. In another WSL tab run the test 
   ```
   python predict_test.py
   ```
9. The Result would be
   ```
   Is there a patient who is suffering from heart risk
   ```

# **2. Data**

**[Heart Disease Dataset (Most comprehensive)](https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final)**



# **2.2. Data References**
This model was built using [kaggle Dataset](https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final)


