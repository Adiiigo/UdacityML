#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

#Imports from Aditi Goyal
from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#Initializing Class Variable
clf = svm.SVC(C=10000 , kernel="rbf", gamma='auto')

#to Decrease the time 
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

#Training the model through Data
t0=time()
clf.fit(features_train , labels_train)
print 'Training Time = ' , round(time()-t0,3) , 's'

#Predicting the Labels of test features
t1 = time()
labels_pred = clf.predict(features_test)
print 'Test Time = ' , round(time()-t1,3) , 's'

#Counting Chris' Mail (Value as 1)
i = 1
count = 0 
for i in labels_pred:
	if (labels_pred[i] == 1):
		count += 1
		i += 1

print 'Chris\' mail count = ' , count

#Finding the accuracy of the model
acc = accuracy_score(labels_test,labels_pred)
print 'Accuracy = ', acc

#########################################################


