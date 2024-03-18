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
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xg


# Split the data

X=admission_df.drop(['Chance of Admit '],axis=1)
y=admission_df['Chance of Admit ']



#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model initialization
models = [
    ("KNN", KNeighborsRegressor(n_neighbors=3)),
    ("Linear Regression", LinearRegression()),
    ("Decision Tree", DecisionTreeRegressor()),
    ("Random Forest", RandomForestRegressor(n_estimators = 100)),
    ("XGBoost", xg.XGBRegressor(objective ='reg:linear', n_estimators = 10))
]


# Model training and evaluation


results = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score_For_Train_Data = round(model.score(X_train,y_train)* 100,2)
    score_For_Test_Data = round(model.score(X_test,y_test)*100,2)
    improvement_rate = (score_For_Test_Data/score_For_Train_Data)-1
    mean_absolute_error = metrics.mean_absolute_error(y_test,y_pred)
    mean_squared_error = metrics.mean_squared_error(y_test,y_pred)
    root_mean_squared_error = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
    print(name)
    print('Score For Train Data : {}'.format(model.score(X_train,y_train)))
    print('Score For Test Data : {}'.format(model.score(X_test,y_test)))
    print('The mean absolute error:', metrics.mean_absolute_error(y_test, y_pred))
    print('The mean squared error:', metrics.mean_squared_error(y_test, y_pred))
    print('The root mean squared error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('\n')
    results.append((name,score_For_Train_Data,score_For_Test_Data,improvement_rate,mean_absolute_error,mean_squared_error,root_mean_squared_error))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel('y_test')
    plt.ylabel('predictions')
    plt.title('Actual test data vs Model predictions - {}'.format(name))
    plt.show()

results_df = pd.DataFrame(results, columns=["Model","Train_Data_Score","Test_Data_Score","improvement_rate","Mean Absolute Error","Mean Squared Error","Root Mean Squared Error"])
results_df = results_df.sort_values(by='Test_Data_Score', ascending=False)

# Plotting results
plt.figure(figsize=(12, 8))
ax = sns.barplot(x='Model', y="Train_Data_Score", data=results_df)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.title("Model Train Data Score Comparison")
plt.xlabel("Model")
plt.ylabel("Train Data Score")
plt.xticks(rotation=45)
plt.show()

#%%

plt.figure(figsize=(12, 8))
ax = sns.barplot(x='Model', y="Test_Data_Score", data=results_df)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.title("Model Test Data Score Comparison")
plt.xlabel("Model")
plt.ylabel("Test Data Score")
plt.xticks(rotation=45)
plt.show()

#%%
plt.figure(figsize=(12, 8))
sns.barplot(data=results_df, x="Model", y="improvement_rate")
plt.title('Improvement Rate of Regression Models')
plt.xlabel('Model')
plt.ylabel('Improvement Rate')
plt.xticks(rotation=45)
plt.show()

# %%
