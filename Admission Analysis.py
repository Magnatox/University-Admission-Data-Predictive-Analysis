#%%
# Loading the data

# Import the libraries

import pandas as pd
import numpy as np
admission_df=pd.read_csv('adm_data.csv',index_col=0)

#%%
print(admission_df.head())
#%%
print(admission_df.info())
#%%
print(admission_df.describe())

#%%
# EDA
# Import the visualization libraries

import matplotlib.pyplot as plt
import seaborn as sns

# Setting style and the color palette
sns.set_style('darkgrid')
sns.set_palette("hls")

#%%
# Heatmap showcasing the correlation among the parameters of the data

admission_df.corr()
sns.heatmap(admission_df.corr(),linecolor='white')

#%%
# Pairplot 

# Map to upper,lower and diagonal
plot = sns.PairGrid(admission_df)
plot.map_diag(plt.hist)
plot.map_upper(plt.scatter)
plot.map_lower(sns.kdeplot)

#%%
# Plots showcasing the distribution of GRE Score

plt.figure(figsize=(13,4))

plt.subplot(1,2,1)
sns.histplot(x="GRE Score", data=admission_df,color='#288BA8',kde=True,lw=1)

plt.subplot(1,2,2)
sns.violinplot(x="GRE Score", data=admission_df,color='#FF7F50',saturation=0.9)

#%%
# Plots showcasing the distribution of TOEFL Score

plt.figure(figsize=(13,4))

plt.subplot(1,2,1)
sns.histplot(x="TOEFL Score", data=admission_df,color='#288BA8',kde=True,lw=1)

plt.subplot(1,2,2)
sns.violinplot(x="TOEFL Score", data=admission_df,color='#B22222',saturation=0.8)

#%%
# Countplot for the University Rating

sns.countplot(x=admission_df['University Rating'],saturation=1)

#%%
# Countplot for the Research parameter

sns.countplot(x='Research',data=admission_df,saturation=5, color='navy')

#%%
# Countplot for the Research & University Rating respective of each other

sns.countplot(x=admission_df['Research'],hue=admission_df['University Rating'],saturation=1)
plt.title("Research count plot taking University Rating into consideration")

#%%
sns.countplot(x=admission_df['University Rating'],hue=admission_df['Research'],saturation=0.8,color='green')
plt.title("University Rating counts taking 'Research' into consideration")

#%%
# Countplot for SOP parameter

sns.countplot(x='SOP',data=admission_df)

#%%
# Countplot of LOR parameter

sns.countplot(x='LOR ',data=admission_df)

#%%
# Regression Plot (lmplot) for showcasing the corr between 'Chance of Admit' & other paramaters taking 'Research' into consideration

# These plots will provide the rough idea about dependency of desire parameter ('Chance of Admit ') on other parameters.

sns.lmplot(x='GRE Score',y='Chance of Admit ',data=admission_df,hue='Research')
plt.title('Chance of Admit vs GRE score')

#%%
sns.lmplot(x='TOEFL Score',y='Chance of Admit ',data=admission_df,hue='Research',palette='coolwarm')
plt.title('Chance of Admit vs TOEFL score')
#%%
sns.lmplot(x='CGPA',y='Chance of Admit ',data=admission_df,hue='Research')
plt.title('Chance of Admit vs CGPA')

#%%
# Predictive Analysis

# Import the libraries

from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor


# Split the data

X=admission_df.drop(['Chance of Admit '],axis=1)
y=admission_df['Chance of Admit ']

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression

# create a linear regression object
lr=LinearRegression()
# fit the model
lr.fit(X_train,y_train)
# predict the values for test data
pred1 = lr.predict(X_test)

# Model Evaluation

# Regression Score of the model
print('Linear Regression')
print('Score For Train Data : {}'.format(lr.score(X_train,y_train)))
print('Score For Test Data : {}'.format(lr.score(X_test,y_test)))

print('The mean absolute error:', metrics.mean_absolute_error(y_test, pred1))
print('The mean squared error:', metrics.mean_squared_error(y_test, pred1))
print('The root mean squared error:', np.sqrt(metrics.mean_squared_error(y_test, pred1)))
print('\n')

# Plot showacasing the how well model fitted on testing data
sns.scatterplot(x=y_test, y=pred1)
plt.xlabel('y_test')
plt.ylabel('predictions')
plt.title('Actual test data vs Model predictions - Linear Regression')
plt.show()


# Decision Tree Regression

# create a regressor object
DTR = DecisionTreeRegressor() 
# fit the model
DTR.fit(X_train,y_train)
# predict the values for test data
pred2 = DTR.predict(X_test)

# Model Evaluation

# Regression Score of the model
print('Decision Tree Regression')
print('Score For Train Data : {}'.format(DTR.score(X_train,y_train)))
print('Score For Test Data : {}'.format(DTR.score(X_test,y_test)))

print('The mean absolute error:', metrics.mean_absolute_error(y_test, pred2))
print('The mean squared error:', metrics.mean_squared_error(y_test, pred2))
print('The root mean squared error:', np.sqrt(metrics.mean_squared_error(y_test, pred2)))
print('\n')

# Plot showacasing the how well model fitted on testing data
sns.scatterplot(x=y_test, y=pred2)
plt.xlabel('y_test')
plt.ylabel('predictions')
plt.title('Actual test data vs Model predictions - Decision Tree Regression')
plt.show()

# Random Forest Regression

# create regressor object
RFreg = RandomForestRegressor(n_estimators = 100)

# fit the regressor with x and y data
RFreg.fit(X_train,y_train) 
# predict the values for test data
pred3 = RFreg.predict(X_test)

# Model Evaluation

# Regression Score of the model
print('Random Forest Regression')
print('Score For Train Data : {}'.format(RFreg.score(X_train,y_train)))
print('Score For Test Data : {}'.format(RFreg.score(X_test,y_test)))

print('The mean absolute error:', metrics.mean_absolute_error(y_test, pred3))
print('The mean squared error:', metrics.mean_squared_error(y_test, pred3))
print('The root mean squared error:', np.sqrt(metrics.mean_squared_error(y_test, pred3)))
print('\n')

# Plot showacasing the how well model fitted on testing data
sns.scatterplot(x=y_test, y=pred3)
plt.xlabel('y_test')
plt.ylabel('predictions')
plt.title('Actual test data vs Model predictions - Random Forest Regression')
plt.show()

# KNN

from sklearn.neighbors import KNeighborsRegressor
# create KKNeighborsRegressor object with initial value of n =2
KNneigh = KNeighborsRegressor(n_neighbors=2)

# fit the regressor with x and y data
KNneigh.fit(X_train,y_train)
# predict the values for test data
pred4 = KNneigh.predict(X_test)

# Model Evaluation

# Regression Score of the model
print('KNN')
print('Score For Train Data : {}'.format(KNneigh.score(X_train,y_train)))
print('Score For Test Data : {}'.format(KNneigh.score(X_test,y_test)))

print('The mean absolute error:', metrics.mean_absolute_error(y_test, pred4))
print('The mean squared error:', metrics.mean_squared_error(y_test, pred4))
print('The root mean squared error:', np.sqrt(metrics.mean_squared_error(y_test, pred4)))
print('\n')

# Plot showacasing the how well model fitted on testing data
sns.scatterplot(x=y_test, y=pred4)
plt.xlabel('y_test')
plt.ylabel('predictions')
plt.title('Actual test data vs Model predictions - KNN')
plt.show()

# XGBoost Regression

import xgboost as xg
# create xgboost object with initial value of n estimators =10
xgb_reg = xg.XGBRegressor(objective ='reg:linear', n_estimators = 10)

# fit the regressor with x and y data
xgb_reg.fit(X_train,y_train)
# predict the values for test data
pred5 = xgb_reg.predict(X_test)

# Regression Score of the model
print('XGBoost Regression')
print('Score For Train Data : {}'.format(xgb_reg.score(X_train,y_train)))
print('Score For Test Data : {}'.format(xgb_reg.score(X_test,y_test)))

print('The mean absolute error:', metrics.mean_absolute_error(y_test, pred5))
print('The mean squared error:', metrics.mean_squared_error(y_test, pred5))
print('The root mean squared error:', np.sqrt(metrics.mean_squared_error(y_test, pred5)))
print('\n')

# Plot showacasing the how well model fitted on testing data
sns.scatterplot(x=y_test, y=pred5)
plt.xlabel('y_test')
plt.ylabel('predictions')
plt.title('Actual test data vs Model predictions - XGBoost Regression')
plt.show()


# %%
