#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

#import - Aditi Goyal
from sklearn import tree
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#min_smaple_split = minimum number of samples(NOS) required to split the leaf node
#if the NOS is less than 40 then splitting will not happen
clf = tree.DecisionTreeClassifier(min_samples_split=40)

clf.fit(features_train , labels_train)

labels_pred = clf.predict(features_test)

acc = accuracy_score(labels_test,labels_pred)
print "Accuracy = " , acc 

#########################################################


