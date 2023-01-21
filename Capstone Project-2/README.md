# PROJECT DESCRIPTION
This project was performed as part of the _ML Zoomcamp Course_, Capstone Project-2. This course is conducted by [Alexey Grigorev](https://bit.ly/3BxeAoB)

# Heart Disease Classification Analysis

![image](https://user-images.githubusercontent.com/82657966/213832332-73d03238-9c40-4791-9411-281cfd963c4f.png)

# **1. Task Description**

This dataset consists of **11 features** and **a target variable**. It has **6 nominal variables** and **5 numeric variables**. The detailed description of all the features are as follows:

1. **age:** Patients Age in years **(Numeric)**
2. **sex:** Gender of patient **(Male - 1, Female - 0) (Nominal)**
3. **chest_pain_ type:** Type of chest pain experienced by patient categorized into **1 typical, 2 typical angina, 3 non- anginal pain, 4 asymptomatic (Nominal)**
4. **resting_bp_s:** Level of blood pressure at resting mode in mm/HG **(Numerical)**
5. **cholestrol:** Serum cholestrol in mg/dl **(Numeric)**
6. **fasting_blood_sugar:** Blood sugar levels on fasting > 120 mg/dl represents as **1 in case of true** and **0 as false (Nominal)**
7. **resting_ecg:** Result of electrocardiogram while at rest are represented in **3 distinct values 0 : Normal 1: Abnormality in ST-T wave 2: Left ventricular hypertrophy (Nominal)
8. **max_heart_rate:** Maximum heart rate achieved **(Numeric)**
9. **exercise_angina:** Angina induced by exercise **0 depicting NO 1 depicting Yes (Nominal)**
10. **oldpeak:** Exercise induced ST-depression in comparison with the state of rest (Numeric)
11. **ST slope:** ST segment measured in terms of slope during peak exercise **0: Normal 1: Upsloping 2: Flat 3: Downsloping (Nominal)**

Target variable

12. **target:** It is the target variable which we have to predict **1 means patient is suffering from heart risk** and **0 means patient is normal.(nom)**

# **2. Data**

**[Heart Disease Dataset (Most comprehensive)](https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final)**

# **2.1. About the Data**

**Heart disease** is also known as Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year which is about 32% of all deaths globally. CVDs are a group of disorders of the heart and blood vessels and include coronary heart disease, cerebrovascular disease, rheumatic heart disease, and other conditions. Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age.

We have curated this dataset by combining different datasets already available independently but not combined before. We have combined them over 11 common features which makes it the largest heart disease dataset available for research purposes. The five datasets used for its curation are:

# **2.2. Data References**
This model was built using [kaggle Dataset](https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final)

# **3. Exploratory Data Analysis**
(You can find codes in heartDisease.ipynb)

![image](https://github.com/jcdumlao14/ML-zoomcamp-course-homework/blob/main/Capstone%20Project-2/image/GenderDistribution.png)

## **Age Distribution**
* Normal Patients

![image](https://github.com/jcdumlao14/ML-zoomcamp-course-homework/blob/main/Capstone%20Project-2/image/NormalP.png)

* Heart Disease Patients

![image](https://github.com/jcdumlao14/ML-zoomcamp-course-homework/blob/main/Capstone%20Project-2/image/HeartDiseaseP.png)

# **4. Create Model**

# **4.1 Training Different Models**
* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting

![image](https://github.com/jcdumlao14/ML-zoomcamp-course-homework/blob/main/Capstone%20Project-2/image/bestmodel.png)

![image](https://github.com/jcdumlao14/ML-zoomcamp-course-homework/blob/main/Capstone%20Project-2/image/val.png)


