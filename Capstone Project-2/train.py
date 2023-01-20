#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Parameters:

# Random Forest
n_estimators     = 50
min_samples_leaf = 5
max_depth        = 10
random_state     = 1 
t                = 0.4

# Cross-validation
n_fold =5

# save file
output_file = 'model_rf_t=0{}.bin'.format(int(t*10))


# 1. Load the Data

!wget https://raw.githubusercontent.com/jcdumlao14/ML-zoomcamp-course-homework/main/Capstone%20Project-2/heart_statlog_cleveland_hungary_final.csv
     
df = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')


# 2. Data preparation

df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')


categorical = ['sex','chest_pain_type','resting_ecg','st_slope']
numerical = ['age','resting_bp_s','cholesterol','fasting_blood_sugar',	'max_heart_rate','exercise_angina','oldpeak']


#categorical = df.select_dtypes(include=['object']).columns.tolist()
#numerical = df.select_dtypes(include=['int64','float64']).columns.tolist()
#numerical.remove('target')

# train/val/test split
df_full_train, df_test = train_test_split(df, test_size=0.20, random_state=1)

# separate the target
y_train = df_full_train.target.values
y_test = df_test.target.values

# reset indexes after splitting shuffling
df_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
del df_test['target'] # remove the target

# encode and scale
dv = DictVectorizer(sparse=False)# for the categorical features
scaler = StandardScaler() # for the numerical features

# full training dataset
train_dict = df_train[categorical].to_dict(orient='records')
X_train_cat = dv.fit_transform(train_dict) # encode the categorical features


X_train_num = df_train[numerical].values
X_train_num = scaler.fit_transform(X_train_num) # scale the numerical features

X_train = np.column_stack([X_train_num, X_train_cat]) # join the matrices

# test dataset
test_dict = df_test[categorical].to_dict(orient='records')
X_test_cat = dv.transform(test_dict) # encode the categorical features

X_test_num = df_test[numerical].values
X_test_num = scaler.transform(X_test_num) # scale the numerical features


X_test = np.column_stack([X_test_num, X_test_cat]) # join the matrices

# 3. Model training

rf = RandomForestClassifier(n_estimators=n_estimators,
                           max_depth=max_depth,
                           min_samples_leaf=min_samples_leaf,
                           random_state=1)

model = rf.fit(X_train,y_train)
y_pred = rf.predict_proba(X_test)[:,1]
acc = accuracy_score(y_test, y_pred >= t)
f1 = f1_score(y_test, y_pred >= t)
auc = roc_auc_score(y_test, y_pred)
print('For the test dataset:','ACC:', acc.round(3),'F1:', f1.round(3),'ROC,AUC:', auc.round(3))


# 4. save the model

with open(output_file,'wb')as f_out:
    pickle.dump((dv, scaler,model), f_out)

print(f'The model is saved to {output_file}')
