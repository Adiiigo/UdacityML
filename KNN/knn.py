
# coding: utf-8

# KNN - predict whether a person will have diabeted or not

# In[2]:


#powerful dataframe
import pandas as pd
#number array
import numpy as np

#To split the current dataset to train and test 
from sklearn.model_selection import train_test_split

#to normalize the dataset - not to skew results
from sklearn.preprocessing import StandardScaler
#library for KNN
from sklearn.neighbors import KNeighborsClassifier

#to test our trained model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# In[10]:


dataset = pd.read_csv("diabetes.csv")

print ( "Names of the heading in the dataset" )
columns = list( dataset.head(0) )
print ( columns , "\n" )

print ("Total length of the dataset")
print ( len(dataset) , "\n" )

print ("First 5 data instances of the dataset")
print( dataset.head(5) )


# In[11]:


#replace zeros
zero_not_accepted = ["Glucose" , "BloodPressure" , "SkinThickness" , "BMI" , "Insulin"]


# In[12]:


print (zero_not_accepted)


# In[14]:


for column in zero_not_accepted :
    dataset[column] = dataset[column].replace(0,np.NaN)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.NaN , mean)


# In[15]:


#splitting the dataset
instances = dataset.iloc[:,0:8]
outcomes = dataset.iloc[:,8]
X_train , X_test , Y_train , Y_test = train_test_split(instances , outcomes , random_state = 0 , test_size = 0.3)


# In[17]:


#feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[18]:


print (len(X_train))
print (len(Y_train))
print (len(X_test))
print (len(Y_test))


# import math
# math.sqrt(len(X_train))
# 

# In[23]:


math.sqrt(len(X_test))


# In[24]:


#define the model
classifier = KNeighborsClassifier(n_neighbors=11 , p=2 , metric = 'euclidean')


# In[26]:


classifier.fit(X_train , Y_train)
y_pred = classifier.predict(X_test)


# In[27]:


cm = confusion_matrix(Y_test , y_pred)


# In[29]:


print (cm)


# In[30]:


print ( f1_score(Y_test , y_pred))


# In[31]:


print (accuracy_score(Y_test,y_pred))

