# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 21:43:23 2022

@author: jsalem7
"""


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

data = pd.read_csv('data_preprocessed.csv')

y = data['application_status'] # true labels
X = pd.DataFrame(data.drop(columns=['application_status'],axis=1))

predictor = LogisticRegression(random_state=0).fit(X, y)

y_hat = predictor.predict(X) # predicted labels

# print(len([i for i in y_hat if i == 0]))
# print(len([i for i in y_hat if i == 1]))
# print(len([i for i in y_hat if i == 2]))

error = (y - y_hat).to_numpy()


# print(len([i for i in error if abs(i) == 0]))
# print(len([i for i in error if abs(i) == 1]))
# print(len([i for i in error if abs(i) == 2]))




# """ Error by gender """

# indices_female = data.index[data['Female'] > 0].tolist()
# indices_male = data.index[data['Male'] > 0].tolist()
# indices_other = data.index[data['Other'] > 0].tolist()

# data_female = pd.DataFrame(data={'Female': error[indices_female]})
# data_male = pd.DataFrame(data={'Male': error[indices_male]})
# data_other = pd.DataFrame(data={'Other': error[indices_other]})
# data_to_plot = pd.concat([data_female, data_male, data_other], axis=1)
# x_positions = np.linspace(1,3,3)
# fig, ax = plt.subplots()
# _=data_to_plot.boxplot(positions=x_positions)
# plt.xticks(x_positions, labels=['Female', 'Male', 'Other'])
# ax.set_ylabel('Error', fontsize=10)
# ax.set_title('Error rate across gender')
                    
# plt.savefig('error-by-gender.png', dpi=400)
# plt.show()

# """ Error by candidate_demographic_variable_1 """

# indices_1 = data.index[data['candidate_demographic_variable_1'] < 0].tolist()
# indices_2 = data.index[data['candidate_demographic_variable_1'] > 0].tolist()

# data_1 = pd.DataFrame(data={'candidate_demographic_variable_1': error[indices_1]})
# data_2 = pd.DataFrame(data={'candidate_demographic_variable_1': error[indices_2]})
# data_to_plot = pd.concat([data_1, data_2], axis=1)
# x_positions = np.linspace(1,2,2)
# fig, ax = plt.subplots()
# _=data_to_plot.boxplot(positions=x_positions)
# plt.xticks(x_positions, labels=['Group 1', 'Group 2'])
# ax.set_ylabel('Error', fontsize=10)
# ax.set_title('Error rate across Demographic 1')
                    
# plt.savefig('error-by-d1.png', dpi=400)
# plt.show()

# """ Error by candidate_demographic_variable_2 """

# indices_1 = data.index[data['candidate_demographic_variable_2'] < 0].tolist()
# indices_2 = data.index[data['candidate_demographic_variable_2'] > 0].tolist()

# data_1 = pd.DataFrame(data={'candidate_demographic_variable_2': error[indices_1]})
# data_2 = pd.DataFrame(data={'candidate_demographic_variable_2': error[indices_2]})
# data_to_plot = pd.concat([data_1, data_2], axis=1)
# x_positions = np.linspace(1,2,2)
# fig, ax = plt.subplots()
# _=data_to_plot.boxplot(positions=x_positions)
# plt.xticks(x_positions, labels=['Group 1', 'Group 2'])
# ax.set_ylabel('Error', fontsize=10)
# ax.set_title('Error rate across Demographic 2')
                    
# plt.savefig('error-by-d2.png', dpi=400)
# plt.show()

""" Error by candidate_demographic_variable_3 """

indices_1 = data.index[data['candidate_demographic_variable_3'] < 0].tolist()
indices_2 = data.index[data['candidate_demographic_variable_3'] > 0].tolist()

data_1 = pd.DataFrame(data={'candidate_demographic_variable_3': error[indices_1]})
data_2 = pd.DataFrame(data={'candidate_demographic_variable_3': error[indices_2]})
data_to_plot = pd.concat([data_1, data_2], axis=1)
x_positions = np.linspace(1,2,2)
fig, ax = plt.subplots()
_=data_to_plot.boxplot(positions=x_positions)
plt.xticks(x_positions, labels=['Group 1', 'Group 2'])
ax.set_ylabel('Error', fontsize=10)
ax.set_title('Error rate across Demographic 3')
                    
plt.savefig('error-by-d3.png', dpi=400)
plt.show()

# """ Error by candidate_demographic_variable_4 """

# indices_1 = data.index[data['candidate_demographic_variable_4'] < 0].tolist()
# indices_2 = data.index[data['candidate_demographic_variable_4'] > 0].tolist()

# data_1 = pd.DataFrame(data={'candidate_demographic_variable_4': error[indices_1]})
# data_2 = pd.DataFrame(data={'candidate_demographic_variable_4': error[indices_2]})
# data_to_plot = pd.concat([data_1, data_2], axis=1)
# x_positions = np.linspace(1,2,2)
# fig, ax = plt.subplots()
# _=data_to_plot.boxplot(positions=x_positions)
# plt.xticks(x_positions, labels=['Group 1', 'Group 2'])
# ax.set_ylabel('Error', fontsize=10)
# ax.set_title('Error rate across Demographic 4')
                    
# plt.savefig('error-by-d4.png', dpi=400)
# plt.show()

# """ Error by candidate_demographic_variable_6 """

# indices_1 = data.index[data['candidate_demographic_variable_6'] < 0].tolist()
# indices_2 = data.index[data['candidate_demographic_variable_6'] > 0].tolist()

# data_1 = pd.DataFrame(data={'candidate_demographic_variable_6': error[indices_1]})
# data_2 = pd.DataFrame(data={'candidate_demographic_variable_6': error[indices_2]})
# data_to_plot = pd.concat([data_1, data_2], axis=1)
# x_positions = np.linspace(1,2,2)
# fig, ax = plt.subplots()
# _=data_to_plot.boxplot(positions=x_positions)
# plt.xticks(x_positions, labels=['Group 1', 'Group 2'])
# ax.set_ylabel('Error', fontsize=10)
# ax.set_title('Error rate across Demographic 6')
                    
# plt.savefig('error-by-d6.png', dpi=400)
# plt.show()

# """ Error by candidate_demographic_variable_7 """

# indices_1 = data.index[data['candidate_demographic_variable_7'] < 0].tolist()
# indices_2 = data.index[data['candidate_demographic_variable_7'] > 0].tolist()

# data_1 = pd.DataFrame(data={'candidate_demographic_variable_7': error[indices_1]})
# data_2 = pd.DataFrame(data={'candidate_demographic_variable_7': error[indices_2]})
# data_to_plot = pd.concat([data_1, data_2], axis=1)
# x_positions = np.linspace(1,2,2)
# fig, ax = plt.subplots()
# _=data_to_plot.boxplot(positions=x_positions)
# plt.xticks(x_positions, labels=['Group 1', 'Group 2'])
# ax.set_ylabel('Error', fontsize=10)
# ax.set_title('Error rate across Demographic 7')
                    
# plt.savefig('error-by-d7.png', dpi=400)
# plt.show()

# """ Error by candidate_demographic_variable_8 """

# indices_1 = data.index[data['candidate_demographic_variable_8'] < 0].tolist()
# indices_2 = data.index[data['candidate_demographic_variable_8'] > 0].tolist()

# data_1 = pd.DataFrame(data={'candidate_demographic_variable_8': error[indices_1]})
# data_2 = pd.DataFrame(data={'candidate_demographic_variable_8': error[indices_2]})
# data_to_plot = pd.concat([data_1, data_2], axis=1)
# x_positions = np.linspace(1,2,2)
# fig, ax = plt.subplots()
# _=data_to_plot.boxplot(positions=x_positions)
# plt.xticks(x_positions, labels=['Group 1', 'Group 2'])
# ax.set_ylabel('Error', fontsize=10)
# ax.set_title('Error rate across Demographic 8')
                    
# plt.savefig('error-by-d8.png', dpi=400)
# plt.show()

""" Error by candidate_demographic_variable_9 """

indices_1 = data.index[data['candidate_demographic_variable_9'] < 0].tolist()
indices_2 = data.index[data['candidate_demographic_variable_9'] > 0].tolist()

data_1 = pd.DataFrame(data={'candidate_demographic_variable_9': error[indices_1]})
data_2 = pd.DataFrame(data={'candidate_demographic_variable_9': error[indices_2]})
data_to_plot = pd.concat([data_1, data_2], axis=1)
x_positions = np.linspace(1,2,2)
fig, ax = plt.subplots()
_=data_to_plot.boxplot(positions=x_positions)
plt.xticks(x_positions, labels=['Group 1', 'Group 2'])
ax.set_ylabel('Error', fontsize=10)
ax.set_title('Error rate across Demographic 9')
                    
plt.savefig('error-by-d9.png', dpi=400)
plt.show()

# """ Error by candidate_demographic_variable_10 """

# indices_1 = data.index[data['candidate_demographic_variable_10'] < 0].tolist()
# indices_2 = data.index[data['candidate_demographic_variable_10'] > 0].tolist()

# data_1 = pd.DataFrame(data={'candidate_demographic_variable_10': error[indices_1]})
# data_2 = pd.DataFrame(data={'candidate_demographic_variable_10': error[indices_2]})
# data_to_plot = pd.concat([data_1, data_2], axis=1)
# x_positions = np.linspace(1,2,2)
# fig, ax = plt.subplots()
# _=data_to_plot.boxplot(positions=x_positions)
# plt.xticks(x_positions, labels=['Group 1', 'Group 2'])
# ax.set_ylabel('Error', fontsize=10)
# ax.set_title('Error rate across Demographic 10')
                    
# plt.savefig('error-by-d10.png', dpi=400)
# plt.show()

# """ Error by age """

# age_threshold = 1
# indices_1 = data.index[data['age'] < age_threshold].tolist()
# indices_2 = data.index[data['age'] >= age_threshold].tolist()

# data_1 = pd.DataFrame(data={'age': error[indices_1]})
# data_2 = pd.DataFrame(data={'age': error[indices_2]})
# data_to_plot = pd.concat([data_1, data_2], axis=1)
# x_positions = np.linspace(1,2,2)
# fig, ax = plt.subplots()
# _=data_to_plot.boxplot(positions=x_positions)
# plt.xticks(x_positions, labels=['Younger group', 'Older group'])
# ax.set_ylabel('Error', fontsize=10)
# ax.set_title('Error rate across age')
                    
# plt.savefig('error-by-age.png', dpi=400)
# plt.show()


""" Hires by number_years_feature_1 """

x_values = np.linspace(-2,2,15)
y_values_female = np.zeros(15)
y_values_male = np.zeros(15)
y_values_other = np.zeros(15)

for i in range(len(x_values)):
    indices = data.index[(data['number_years_feature_1'] > x_values[i]) & 
                         (data['Female'] > 0)].tolist()
    num_above_x = len(indices) + 0.0
    predictions_above_x = pd.DataFrame(data={'number_years_feature_1': y_hat[indices]})
    hired_value = predictions_above_x['number_years_feature_1'].max()
    num_hired = predictions_above_x[predictions_above_x == hired_value].count() + 0.0
    y_values_female[i] = num_hired / num_above_x
    
    indices = data.index[(data['number_years_feature_1'] > x_values[i]) & 
                         (data['Male'] > 0)].tolist()
    num_above_x = len(indices) + 0.0
    predictions_above_x = pd.DataFrame(data={'number_years_feature_1': y_hat[indices]})
    hired_value = predictions_above_x['number_years_feature_1'].max()
    num_hired = predictions_above_x[predictions_above_x == hired_value].count() + 0.0
    y_values_male[i] = num_hired / num_above_x
    
    indices = data.index[(data['number_years_feature_1'] > x_values[i]) & 
                         (data['Other'] > 0)].tolist()
    num_above_x = len(indices) + 0.0
    predictions_above_x = pd.DataFrame(data={'number_years_feature_1': y_hat[indices]})
    hired_value = predictions_above_x['number_years_feature_1'].max()
    num_hired = predictions_above_x[predictions_above_x == hired_value].count() + 0.0
    y_values_other[i] = num_hired / num_above_x
    
fig, ax = plt.subplots()

plt.plot(x_values,y_values_female, label = 'Female')
plt.plot(x_values,y_values_male, label = 'Male')
plt.plot(x_values,y_values_other, label = 'Other/NA')
ax.set_ylabel('Hire rate', fontsize=10)
ax.set_xlabel('number_years_feature_1 (normalized)', fontsize=10)
ax.set_title('Hires across number_years_feature_1 and gender')
plt.legend(loc='upper left')
plt.savefig('hires-by-years1-by-gender.png', dpi=400)
plt.show()




""" Noisy Experiments """

# noise levels
prob = [.01, .03, .05, .08, .1, .2, .3]

for p in prob:
    
    y_hat_noisy = y_hat
    # perturb predictions
    for i in range(len(y_hat_noisy)):
        r = np.random.uniform()
        if y_hat[i] == 0:
            if r < p:
                y_hat_noisy[i] = 1
        elif y_hat[i] == 1:
            if r < p:
                y_hat_noisy[i] = 0
            elif r < 2*p:
                y_hat_noisy[i] = 2
        else:
            if r < p:
                y_hat_noisy[i] = 1
    
    """ Noisy Hires by number_years_feature_1 """
    x_values = np.linspace(-2,2,15)
    y_values_female = np.zeros(15)
    y_values_male = np.zeros(15)
    y_values_other = np.zeros(15)
    
    for i in range(len(x_values)):
        indices = data.index[(data['number_years_feature_1'] > x_values[i]) & 
                              (data['Female'] > 0)].tolist()
        num_above_x = len(indices) + 0.0
        predictions_above_x = pd.DataFrame(data={'number_years_feature_1': y_hat_noisy[indices]})
        hired_value = predictions_above_x['number_years_feature_1'].max()
        num_hired = predictions_above_x[predictions_above_x == hired_value].count() + 0.0
        y_values_female[i] = num_hired / num_above_x
        
        indices = data.index[(data['number_years_feature_1'] > x_values[i]) & 
                              (data['Male'] > 0)].tolist()
        num_above_x = len(indices) + 0.0
        predictions_above_x = pd.DataFrame(data={'number_years_feature_1': y_hat_noisy[indices]})
        hired_value = predictions_above_x['number_years_feature_1'].max()
        num_hired = predictions_above_x[predictions_above_x == hired_value].count() + 0.0
        y_values_male[i] = num_hired / num_above_x
        
        indices = data.index[(data['number_years_feature_1'] > x_values[i]) & 
                              (data['Other'] > 0)].tolist()
        num_above_x = len(indices) + 0.0
        predictions_above_x = pd.DataFrame(data={'number_years_feature_1': y_hat_noisy[indices]})
        hired_value = predictions_above_x['number_years_feature_1'].max()
        num_hired = predictions_above_x[predictions_above_x == hired_value].count() + 0.0
        y_values_other[i] = num_hired / num_above_x
        
    fig, ax = plt.subplots()
    
    plt.plot(x_values,y_values_female, label = 'Female')
    plt.plot(x_values,y_values_male, label = 'Male')
    plt.plot(x_values,y_values_other, label = 'Other/NA')
    ax.set_ylabel('Hire rate', fontsize=10)
    ax.set_xlabel('number_years_feature_1 (normalized)', fontsize=10)
    ax.set_title('Hires across number_years_feature_1 and gender, p={}'.format(p))
    plt.legend(loc='upper left')
    plt.savefig('hires-by-years1-by-gender-p{}.png'.format(p), dpi=400)
    plt.show()
    
    
    """ Noisy error by candidate_demographic_variable_3 """

    indices_1 = data.index[data['candidate_demographic_variable_3'] < 0].tolist()
    indices_2 = data.index[data['candidate_demographic_variable_3'] > 0].tolist()
    
    data_1 = pd.DataFrame(data={'candidate_demographic_variable_3': (y - y_hat_noisy)[indices_1]})
    data_2 = pd.DataFrame(data={'candidate_demographic_variable_3': (y - y_hat_noisy)[indices_2]})
    data_to_plot = pd.concat([data_1, data_2], axis=1)
    x_positions = np.linspace(1,2,2)
    fig, ax = plt.subplots()
    _=data_to_plot.boxplot(positions=x_positions)
    plt.xticks(x_positions, labels=['Group 1', 'Group 2'])
    ax.set_ylabel('Error', fontsize=10)
    ax.set_title('Error rate across Demographic 3, p={}'.format(p))
                        
    plt.savefig('error-by-d3-p{}.png'.format(p), dpi=400)
    plt.show()
    
    """ Noisy error by candidate_demographic_variable_9 """

    indices_1 = data.index[data['candidate_demographic_variable_9'] < 0].tolist()
    indices_2 = data.index[data['candidate_demographic_variable_9'] > 0].tolist()
    
    data_1 = pd.DataFrame(data={'candidate_demographic_variable_9': (y - y_hat_noisy)[indices_1]})
    data_2 = pd.DataFrame(data={'candidate_demographic_variable_9': (y - y_hat_noisy)[indices_2]})
    data_to_plot = pd.concat([data_1, data_2], axis=1)
    x_positions = np.linspace(1,2,2)
    fig, ax = plt.subplots()
    _=data_to_plot.boxplot(positions=x_positions)
    plt.xticks(x_positions, labels=['Group 1', 'Group 2'])
    ax.set_ylabel('Error', fontsize=10)
    ax.set_title('Error rate across Demographic 9, p={}'.format(p))
                        
    plt.savefig('error-by-d9-p{}.png'.format(p), dpi=400)
    plt.show()


def predict(p):
    data = pd.read_csv('data_preprocessed.csv')

    y = data['application_status'] # true labels
    X = pd.DataFrame(data.drop(columns=['application_status'],axis=1))
    
    predictor = LogisticRegression(random_state=0).fit(X, y)
    
    y_hat = predictor.predict(X) # predicted labels
    
    y_hat_noisy = y_hat
    # perturb predictions
    for i in range(len(y_hat_noisy)):
        r = np.random.uniform()
        if y_hat[i] == 0:
            if r < p:
                y_hat_noisy[i] = 1
        elif y_hat[i] == 1:
            if r < p:
                y_hat_noisy[i] = 0
            elif r < 2*p:
                y_hat_noisy[i] = 2
        else:
            if r < p:
                y_hat_noisy[i] = 1
    return y_hat_noisy
