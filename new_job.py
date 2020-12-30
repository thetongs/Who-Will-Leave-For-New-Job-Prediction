## Problem statement
# Based on the data given for month predict the employee 
# who can leave for new job.
# 0 not looking for new job.
# 1 looking for new job.

## Libraries
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Load training data and testing dataset
#
dataset_train = pd.read_csv('aug_train.csv')
column_name = list(dataset_train.columns)

print("Training size : {}".format(len(dataset_train)))
dataset_test = pd.read_csv('aug_test.csv')
dataset_train = dataset_train.append(dataset_test, ignore_index = True)

# See 5 rows
dataset_train.head()

## Check missing value of each column
# count
dataset_train.isna().sum()

# percentage
missing_percentage = [(clm_name, dataset_train[clm_name].isna().mean() * 100) for clm_name in dataset_train]
missing_percentage = pd.DataFrame(missing_percentage, columns = ['column_names', 'percentage'])
missing_percentage

## Handle missing values
# Using frequency distribution of each category
# and select the top category which is mode.
dataset_train = dataset_train.fillna(dataset_train['gender'].value_counts().index[0])

# Check results
dataset_train.isna().sum()

dataset_train.head()

# Load training data
#
dataset_train = pd.read_csv('aug_train.csv')
column_name = list(dataset_train.columns)

print("Training size : {}".format(len(dataset_train)))
dataset_test = pd.read_csv('aug_test.csv')
dataset_train = dataset_train.append(dataset_test, ignore_index = True)

# Using SimpleImputer
# After using simple imputer the column names 
# change into numeric index
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy="most_frequent")
dataset_train = imputer.fit_transform(dataset_train)

dataset_train = pd.DataFrame(dataset_train)
dataset_train.columns = column_name
dataset_train.isna().sum()

## Check data type of each column
#
dataset_train.dtypes

## Change data type of column into respective
#
dataset_train.enrollee_id = dataset_train.enrollee_id.astype('int')

dataset_train.city_development_index = dataset_train.city_development_index.astype('int')

dataset_train.training_hours = dataset_train.training_hours.astype('float')

dataset_train.target = dataset_train.target.astype('int32')

dataset_train.city = dataset_train.city.astype('category')

# using dictionary to convert specific columns 
convert_dict = {'city': 'category',
                'gender': 'category',
                'relevent_experience':'category',
                'enrolled_university':'category',
                'education_level':'category',
                'major_discipline':'category',
                'experience':'category',
                'company_size':'category',
                'company_type':'category',
                'last_new_job': 'category'
               } 
  
dataset_train = dataset_train.astype(convert_dict) 
dataset_train.dtypes

## Handle categorical datatype columns
# Using LabelEncoder for multiple columns
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

cl_names = ['city','gender','relevent_experience','enrolled_university',
 'education_level',
 'major_discipline',
 'experience',
 'company_size',
 'company_type',
 'last_new_job']

dataset_train[cl_names] = dataset_train[cl_names].apply(encoder.fit_transform)
dataset_train.head()

## Back to test data
#
dataset_test =dataset_train[19158:]
dataset_test.head()

## Check missing value of each column
# count
dataset_test.isna().sum()

## Build model
#
independent_variables =["enrollee_id","city","city_development_index","gender","relevent_experience","enrolled_university","education_level","major_discipline","experience","company_size","company_type","last_new_job","training_hours"]

dependent_variables = 'target'

## Installation
# pip install catboost
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score, mean_squared_error

model = CatBoostRegressor(objective='RMSE')
model.fit(dataset_train[independent_variables], dataset_train[dependent_variables])

## Predictions
#
predictions = model.predict(dataset_test[independent_variables])
predictions

## Accuracy
#
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(dataset_train[dependent_variables],  model.predict(dataset_train[independent_variables]))
metrics.auc(fpr, tpr)

## Generate report
#
results = pd.DataFrame({'enroll_id':dataset_test.enrollee_id, 'result':predictions})
results.head()

# segregate result
def segregator(data):
    if(data.result > 0.5):
        return 1
    else:
        return 0
    
results['result'] = results.apply(func = segregator, axis = 'columns')

results = results.sort_values(by = 'result', ascending = False)
print("Following are enroll_id are about to get new job as per our preduction")
top10 = results[:10]
print(top10)

# Sheet
file_name = "final_results.xlsx"
results.to_excel(file_name, index = False)
print("prediction list generated and saved.")

## End
#

