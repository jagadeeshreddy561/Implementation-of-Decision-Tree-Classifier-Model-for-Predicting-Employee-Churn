# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
### Step1
Import the required libraries.
### Step2
Upload and read the dataset.
### Step3
Check for any null values using the isnull() function.
### Step4
From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
### Step5
Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
## Developed by: jagadeeshreddy
## RegisterNumber: 212222240059  
```python

import pandas as pd
data=pd.read_csv("/content/Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## output:

## Data head():


![ml-6 1](https://github.com/jagadeeshreddy561/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120623104/299785d3-a7c4-4d63-a52c-dc14ed37f7d6)


## Data set info():


![ml-6 2](https://github.com/jagadeeshreddy561/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120623104/8fdc4d0d-4efe-4f8e-b6d7-03700d768460)



## Null dataset:


![ml-6 3](https://github.com/jagadeeshreddy561/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120623104/79742369-6260-4d97-95dc-f1abdb3d1078)

## Values count():

![ml-6 4](https://github.com/jagadeeshreddy561/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120623104/6991be4f-8b2b-47bb-9365-ec31c45496a3)



## Data head() for salary:



![ml-6 5](https://github.com/jagadeeshreddy561/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120623104/cd86d7e4-83e4-466a-bbaf-4631eb990a04)



## x.head():


![ml-6 6](https://github.com/jagadeeshreddy561/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120623104/e0ff3127-84f5-4787-80d0-5ed63b38c080)


## Accuracy value:

![ml-6 7](https://github.com/jagadeeshreddy561/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120623104/4c251b32-0721-4b38-8b56-a2c249bd9bff)



## Data prediction:


![ml-6 8](https://github.com/jagadeeshreddy561/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/120623104/504c5ecb-f63c-4589-8b9a-2d86ba783327)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
