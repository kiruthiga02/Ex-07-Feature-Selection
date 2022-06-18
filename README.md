# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE

```
Program developed by:Kiruthiga M
Reg no:212219040061
```
```
from sklearn.datasets import load_boston
boston_data=load_boston()
import pandas as pd
boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
boston['MEDV'] = boston_data.target
dummies = pd.get_dummies(boston.RAD)
boston = boston.drop(columns='RAD').merge(dummies,left_index=True,right_index=True)
X = boston.drop(columns='MEDV')
y = boston.MEDV
boston.head(10)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from math import sqrt

cv = KFold(n_splits=10, random_state=None, shuffle=False)
classifier_pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10))
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),2)))
print("R_squared: " + str(round(r2_score(y,y_pred),2)))
```
## Filter Features by Correlation
```
import seaborn as sn
import matplotlib.pyplot as plt
fig_dims = (12, 8)
fig, ax = plt.subplots(figsize=fig_dims)
sn.heatmap(boston.corr(), ax=ax)
plt.show()
abs(boston.corr()["MEDV"])
abs(boston.corr()["MEDV"][abs(boston.corr()["MEDV"])>0.5].drop('MEDV')).index.tolist()
vals = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
for val in vals:
    features = abs(boston.corr()["MEDV"][abs(boston.corr()["MEDV"])>val].drop('MEDV')).index.tolist()
    
    X = boston.drop(columns='MEDV')
    X=X[features]
    
    print(features)

    y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
    print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),2)))
    print("R_squared: " + str(round(r2_score(y,y_pred),2)))
```
## Feature Selection Using a Wrapper
```
boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
boston['MEDV'] = boston_data.target
boston['RAD'] = boston['RAD'].astype('category')
dummies = pd.get_dummies(boston.RAD)
boston = boston.drop(columns='RAD').merge(dummies,left_index=True,right_index=True)
X = boston.drop(columns='MEDV')
y = boston.MEDV

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

sfs1 = SFS(classifier_pipeline, 
           k_features=1, 
           forward=False, 
           scoring='neg_mean_squared_error',
           cv=cv)

X = boston.drop(columns='MEDV')
sfs1.fit(X,y)
sfs1.subsets_

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT']]
y = boston['MEDV']
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))
print("R_squared: " + str(round(r2_score(y,y_pred),3)))

boston[['CRIM','RM','PTRATIO','LSTAT','MEDV']].corr()

boston['RM*LSTAT']=boston['RM']*boston['LSTAT']

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT']]
y = boston['MEDV']
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))
print("R_squared: " + str(round(r2_score(y,y_pred),3)))

sn.pairplot(boston[['CRIM','RM','PTRATIO','LSTAT','MEDV']])

boston = boston.drop(boston[boston['MEDV']==boston['MEDV'].max()].index.tolist())

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT','RM*LSTAT']]
y = boston['MEDV']
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))
print("R_squared: " + str(round(r2_score(y,y_pred),3)))

boston['LSTAT_2']=boston['LSTAT']**2

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT']]
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))
print("R_squared: " + str(round(r2_score(y,y_pred),3)))
```




# OUPUT
![image](https://user-images.githubusercontent.com/98682825/174454470-b42d3a71-f00f-4735-9ce1-43b74242a0e9.png)
![image](https://user-images.githubusercontent.com/98682825/174454473-6a4f7617-1cca-4589-9959-f7cbdc07ab80.png)
![image](https://user-images.githubusercontent.com/98682825/174454476-885384ba-cc99-46c3-9360-bd5a420dea7d.png)
![image](https://user-images.githubusercontent.com/98682825/174454484-e690dc36-36ee-48b4-aa3b-946385928645.png)
![image](https://user-images.githubusercontent.com/98682825/174454487-2624c320-76a2-4400-8571-e187eff04ae3.png)
![image](https://user-images.githubusercontent.com/98682825/174454493-2f865af6-1490-4bf2-8a46-8103111aa70e.png)
![image](https://user-images.githubusercontent.com/98682825/174454501-64be4c9b-8611-4eac-9f14-faac823f8043.png)
![image](https://user-images.githubusercontent.com/98682825/174454511-53c424e7-2444-4bdd-bdcc-134b208a83ad.png)
![image](https://user-images.githubusercontent.com/98682825/174454519-525fbd20-f7d3-4a84-910b-d5c487e460c0.png)
![image](https://user-images.githubusercontent.com/98682825/174454524-b275b523-9f81-47d5-ae9f-1b3e643e59b0.png)
![image](https://user-images.githubusercontent.com/98682825/174454543-2e3b4d6f-da3a-4670-9a66-937c36ffe118.png)
![image](https://user-images.githubusercontent.com/98682825/174454546-8006b42a-814f-402a-89c6-b8d291c1aa59.png)
![image](https://user-images.githubusercontent.com/98682825/174454560-ad7f7a76-b082-4eb4-87bb-7308e3390507.png)
![image](https://user-images.githubusercontent.com/98682825/174454574-a3949224-0e01-4f6e-8bdf-68a361f52d78.png)
![image](https://user-images.githubusercontent.com/98682825/174454567-e576351d-558d-4fd6-a9dd-3664d9eb7276.png)
![image](https://user-images.githubusercontent.com/98682825/174454658-67d4d3a0-c8ea-4748-8bae-9c263c78e44a.png)


## RESULT
    The various feature selection techniques has been performed on a dataset and saved the data to a file.


