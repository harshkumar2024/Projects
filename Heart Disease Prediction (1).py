#!/usr/bin/env python
# coding: utf-8

# # NAIVE BAYE's 

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'E:\Study Material\4th Sem\ML LAB\heart.csv')
X = dataset.iloc[:, 0:13].values

y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.24, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
ac = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(ac)


# In[3]:


#Checking Model
input_data = (58,0,0,100,248,0,0,122,0,1,1,0,2)

input_data_array = np.asarray(input_data)
input_data_reshape = input_data_array.reshape(1,-1)
prediction = classifier.predict(input_data_reshape)
print (prediction)
if (prediction[0] == 0  ):
    print("The person does not have the heart disease")
else :
    print("The person have the heart disease")


# # KNN algorithm

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'E:\Study Material\4th Sem\ML LAB\heart.csv')
X = dataset.iloc[:, 0:13].values

y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.18, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 4, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test,y_pred)
print(cm)
print(ac)


# In[5]:


#Checking Model
input_data = (58,0,0,100,248,0,0,122,0,1,1,0,2)

input_data_array = np.asarray(input_data)
input_data_reshape = input_data_array.reshape(1,-1)
prediction = classifier.predict(input_data_reshape)

print (prediction)
if (prediction[0] == 0  ):
    print("The person does not have the heart disease")
else :
    print("The person have the heart disease")


# # LOGISTIC REGRESSION
# 

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score

dataset = pd.read_csv(r'E:\Study Material\4th Sem\ML LAB\heart.csv')
X = dataset.iloc[:, 0:13].values

y = dataset.iloc[:, -1].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20 ,random_state = 99)

#Fit the Logistic Regression Model
model = LogisticRegression() 
model.fit(X_train, y_train)
X_train_prediction= model.predict(X_train)
training_data_accuracy= accuracy_score(X_train_prediction, y_train)

print("Accuracy on Training data :",training_data_accuracy)


# In[8]:


##Checking Model
input_data = (58,0,0,100,248,0,0,122,0,1,1,0,2)

input_data_array = np.asarray(input_data)
input_data_reshape = input_data_array.reshape(1,-1)
prediction = model.predict(input_data_reshape)

print (prediction)
if (prediction[0] == 0  ):
    print("The person does not have the heart disease")
else :
    print("The person have the heart disease")


# In[ ]:




