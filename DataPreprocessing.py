# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 19:58:03 2022

@author: jsalem7
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.preprocessing

# read the data
data=pd.read_csv('data.csv')
print(data.shape)

# fill missing skill counts with 0 (pessimistic approach)
for i in range(1,10):
    data['occupation_skill_{}_count'.format(i)].fillna(0, inplace=True)
    
# fill in other missing data (modify this later if time)
data['candidate_attribute_1'].fillna(0, inplace=True)
data['candidate_attribute_2'].fillna(0, inplace=True)
data['candidate_demographic_variable_1'].fillna(0, inplace=True)
data['candidate_demographic_variable_2'].fillna(0, inplace=True)
data['candidate_demographic_variable_3'].fillna(0, inplace=True)
data['candidate_demographic_variable_4'].fillna(0, inplace=True)
data['ethnicity'].fillna('RNS_ethnicity', inplace=True)
data['gender'].fillna('RNS_gender', inplace=True)
data['candidate_demographic_variable_5'].fillna('RNS_cdv5', inplace=True)
data['candidate_demographic_variable_6'].fillna(0, inplace=True)
data['candidate_demographic_variable_7'].fillna(0, inplace=True)
data['candidate_demographic_variable_8'].fillna(0, inplace=True)
data['candidate_demographic_variable_9'].fillna(0, inplace=True)
data['candidate_demographic_variable_10'].fillna(0, inplace=True)
data['age'].fillna(30, inplace=True)
data['candidate_attribute_3'].fillna(0, inplace=True)
data['candidate_attribute_4'].fillna(0, inplace=True)
data['candidate_attribute_5'].fillna(0, inplace=True)
data['candidate_attribute_6'].fillna(0, inplace=True)
data['candidate_attribute_7'].fillna(0, inplace=True)
data['candidate_attribute_8'].fillna(0, inplace=True)
data['candidate_interest_1'].fillna(0, inplace=True)
data['candidate_interest_2'].fillna(0, inplace=True)
data['candidate_interest_3'].fillna(0, inplace=True)
data['candidate_interest_4'].fillna(0, inplace=True)
data['candidate_interest_5'].fillna(0, inplace=True)
data['candidate_interest_6'].fillna(0, inplace=True)
data['candidate_interest_7'].fillna(0, inplace=True)
data['candidate_interest_8'].fillna(0, inplace=True)
data['number_years_feature_1'].fillna(0, inplace=True)
data['number_years_feature_2'].fillna(0, inplace=True)
data['number_years_feature_3'].fillna(0, inplace=True)
data['number_years_feature_4'].fillna(0, inplace=True)
data['number_years_feature_5'].fillna(0, inplace=True)

data['candidate_relative_test_1'].fillna(0, inplace=True)
data['candidate_relative_test_2'].fillna(0, inplace=True)

for i in range(1,10):
    data['candidate_skill_{}_count'.format(i)].fillna(0, inplace=True)

# find all application statuses
print(data['application_status'].unique())

# map application statuses to numbers
status_map = \
{'interview' : 1,
 'hired' : 2,
 'pre-interview' : 0}

data['application_status'] = data['application_status'].map(status_map)

# change name so it doesn't conflict with another "Other"
data['ethnicity'].replace('Other','Other_ethnicity',inplace=True)

print(len(data['ethnicity'].unique()))

# drop identifiers
data = pd.DataFrame(data.drop(columns=['candidate_id','occupation_id','company_id','application_attribute_1'],axis=1))

# one hot encodings
one_hot_ethnicity = pd.get_dummies(data['ethnicity'])
data = data.drop('ethnicity',axis = 1)
data = data.join(one_hot_ethnicity)

one_hot_gender = pd.get_dummies(data['gender'])
data = data.drop('gender',axis = 1)
data = data.join(one_hot_gender)

one_hot_cdv5 = pd.get_dummies(data['candidate_demographic_variable_5'])
data = data.drop('candidate_demographic_variable_5',axis = 1)
data = data.join(one_hot_cdv5)


# check if any nan is left
print(data.isnull().any().any())

# normalize the data
print(data.columns)
features = ['number_of_employees_log',
       'occupation_skill_1_count', 'occupation_skill_2_count',
       'occupation_skill_3_count', 'occupation_skill_4_count',
       'occupation_skill_5_count', 'occupation_skill_6_count',
       'occupation_skill_7_count', 'occupation_skill_8_count',
       'occupation_skill_9_count', 'candidate_attribute_1',
       'candidate_attribute_2', 'candidate_demographic_variable_1',
       'candidate_demographic_variable_2', 'candidate_demographic_variable_3',
       'candidate_demographic_variable_4', 'candidate_demographic_variable_6',
       'candidate_demographic_variable_7', 'candidate_demographic_variable_8',
       'candidate_demographic_variable_9', 'candidate_demographic_variable_10',
       'age', 'candidate_attribute_3', 'candidate_attribute_4',
       'candidate_attribute_5', 'candidate_attribute_6',
       'candidate_attribute_7', 'candidate_interest_1', 'candidate_interest_2',
       'candidate_interest_3', 'candidate_interest_4', 'candidate_interest_5',
       'candidate_interest_6', 'candidate_interest_7', 'candidate_interest_8',
       'candidate_attribute_8', 'number_years_feature_1',
       'number_years_feature_2', 'number_years_feature_3',
       'number_years_feature_4', 'number_years_feature_5',
       'candidate_skill_1_count', 'candidate_skill_2_count',
       'candidate_skill_3_count', 'candidate_skill_4_count',
       'candidate_skill_5_count', 'candidate_skill_6_count',
       'candidate_skill_7_count', 'candidate_skill_8_count',
       'candidate_skill_9_count', 'candidate_relative_test_1',
       'candidate_relative_test_2', 'American Indian or Alaska Native',
       'Any other ethnic group Arab', 'Any other ethnic group not listed',
       'Asian', 'Asian Bangladeshi', 'Asian Chinese', 'Asian Indian',
       'Asian Pakistani', 'Asian any other background',
       'Black / African / Caribbean any other background', 'Black African',
       'Black Caribbean', 'Black or African American', 'Hispanic / Latinx',
       'Mixed ethnic White and Asian', 'Mixed ethnic White and Black African',
       'Mixed ethnic White and Black Caribbean',
       'Mixed ethnic any other background', 'Other_ethnicity', 'RNS_ethnicity',
       'Rather not say', 'White',
       'White English / Welsh / Scottish / Northern Irish / British',
       'White Gypsy / Irish Traveller', 'White Irish',
       'White any other background', 'Female', 'Male', 'Other', 'RNS_cdv5',
       'citizenship', 'international_visa', 'other_document', 'work_card',
       'work_permit']

data[['number_of_employees_log',
       'occupation_skill_1_count', 'occupation_skill_2_count',
       'occupation_skill_3_count', 'occupation_skill_4_count',
       'occupation_skill_5_count', 'occupation_skill_6_count',
       'occupation_skill_7_count', 'occupation_skill_8_count',
       'occupation_skill_9_count', 'candidate_attribute_1',
       'candidate_attribute_2', 'candidate_demographic_variable_1',
       'candidate_demographic_variable_2', 'candidate_demographic_variable_3',
       'candidate_demographic_variable_4', 'candidate_demographic_variable_6',
       'candidate_demographic_variable_7', 'candidate_demographic_variable_8',
       'candidate_demographic_variable_9', 'candidate_demographic_variable_10',
       'age', 'candidate_attribute_3', 'candidate_attribute_4',
       'candidate_attribute_5', 'candidate_attribute_6',
       'candidate_attribute_7', 'candidate_interest_1', 'candidate_interest_2',
       'candidate_interest_3', 'candidate_interest_4', 'candidate_interest_5',
       'candidate_interest_6', 'candidate_interest_7', 'candidate_interest_8',
       'candidate_attribute_8', 'number_years_feature_1',
       'number_years_feature_2', 'number_years_feature_3',
       'number_years_feature_4', 'number_years_feature_5',
       'candidate_skill_1_count', 'candidate_skill_2_count',
       'candidate_skill_3_count', 'candidate_skill_4_count',
       'candidate_skill_5_count', 'candidate_skill_6_count',
       'candidate_skill_7_count', 'candidate_skill_8_count',
       'candidate_skill_9_count', 'candidate_relative_test_1',
       'candidate_relative_test_2', 'American Indian or Alaska Native',
       'Any other ethnic group Arab', 'Any other ethnic group not listed',
       'Asian', 'Asian Bangladeshi', 'Asian Chinese', 'Asian Indian',
       'Asian Pakistani', 'Asian any other background',
       'Black / African / Caribbean any other background', 'Black African',
       'Black Caribbean', 'Black or African American', 'Hispanic / Latinx',
       'Mixed ethnic White and Asian', 'Mixed ethnic White and Black African',
       'Mixed ethnic White and Black Caribbean',
       'Mixed ethnic any other background', 'Other_ethnicity', 'RNS_ethnicity',
       'Rather not say', 'White',
       'White English / Welsh / Scottish / Northern Irish / British',
       'White Gypsy / Irish Traveller', 'White Irish',
       'White any other background', 'Female', 'Male', 'Other', 'RNS_cdv5',
       'citizenship', 'international_visa', 'other_document', 'work_card',
       'work_permit']] = pd.DataFrame(sklearn.preprocessing.scale(data[['number_of_employees_log',
       'occupation_skill_1_count', 'occupation_skill_2_count',
       'occupation_skill_3_count', 'occupation_skill_4_count',
       'occupation_skill_5_count', 'occupation_skill_6_count',
       'occupation_skill_7_count', 'occupation_skill_8_count',
       'occupation_skill_9_count', 'candidate_attribute_1',
       'candidate_attribute_2', 'candidate_demographic_variable_1',
       'candidate_demographic_variable_2', 'candidate_demographic_variable_3',
       'candidate_demographic_variable_4', 'candidate_demographic_variable_6',
       'candidate_demographic_variable_7', 'candidate_demographic_variable_8',
       'candidate_demographic_variable_9', 'candidate_demographic_variable_10',
       'age', 'candidate_attribute_3', 'candidate_attribute_4',
       'candidate_attribute_5', 'candidate_attribute_6',
       'candidate_attribute_7', 'candidate_interest_1', 'candidate_interest_2',
       'candidate_interest_3', 'candidate_interest_4', 'candidate_interest_5',
       'candidate_interest_6', 'candidate_interest_7', 'candidate_interest_8',
       'candidate_attribute_8', 'number_years_feature_1',
       'number_years_feature_2', 'number_years_feature_3',
       'number_years_feature_4', 'number_years_feature_5',
       'candidate_skill_1_count', 'candidate_skill_2_count',
       'candidate_skill_3_count', 'candidate_skill_4_count',
       'candidate_skill_5_count', 'candidate_skill_6_count',
       'candidate_skill_7_count', 'candidate_skill_8_count',
       'candidate_skill_9_count', 'candidate_relative_test_1',
       'candidate_relative_test_2', 'American Indian or Alaska Native',
       'Any other ethnic group Arab', 'Any other ethnic group not listed',
       'Asian', 'Asian Bangladeshi', 'Asian Chinese', 'Asian Indian',
       'Asian Pakistani', 'Asian any other background',
       'Black / African / Caribbean any other background', 'Black African',
       'Black Caribbean', 'Black or African American', 'Hispanic / Latinx',
       'Mixed ethnic White and Asian', 'Mixed ethnic White and Black African',
       'Mixed ethnic White and Black Caribbean',
       'Mixed ethnic any other background', 'Other_ethnicity', 'RNS_ethnicity',
       'Rather not say', 'White',
       'White English / Welsh / Scottish / Northern Irish / British',
       'White Gypsy / Irish Traveller', 'White Irish',
       'White any other background', 'Female', 'Male', 'Other', 'RNS_cdv5',
       'citizenship', 'international_visa', 'other_document', 'work_card',
       'work_permit']]), columns = features)
# print(data)
# print(data['work_card'].mean())

# check for outliers 
print(data[data < -5].count().sum())
print(data[data > 10].count().sum())
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(data[data > 5].count())



print(data['application_status'])

data.to_csv('data_preprocessed.csv', index=False)


